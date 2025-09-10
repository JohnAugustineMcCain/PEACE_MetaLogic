#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gb_auto.py — Self-tuning subtractor-only Goldbach engine (deterministic)
#
# What it does
# ------------
# - Uses a subtractor-only search (q = n - p) with:
#     • Wheel residue filter (skip p that force q divisible by small primes)
#     • Small-prime pre-sieve on q (before BPSW)
#     • Learned band ordering over ascending subtractor primes (zero hot-loop cost)
#     • Early break when p >= n-2
# - Auto-tunes parameters with deterministic strategies:
#     • sha     : Successive Halving (racing) — fast exploitation, minimal exploration
#     • greedy  : Always run best-so-far; only replace if a challenger is strictly better
#     • hybrid  : ε-greedy that decays exploration to zero
# - Persists knowledge:
#     • subs_learn_cache.json : per-digit band CTRs for learned ordering
#     • gb_auto_best.json     : best-per-digit config (used to bias future runs)
#
# CLI output per iteration prints ONLY:
#     rate=...  avg_checks=...  avg_ms=...
# (no bars, no parameter echo)
#
# At the end, prints exactly two sample Goldbach decompositions found with the best config.
#
# (c) 2025 — MIT-style permissive for this prototype

from __future__ import annotations
import argparse, math, random, time, json, os, csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ----------------------------- Utilities ---------------------------------

def random_even_with_digits(D: int, rng: random.Random) -> int:
    if D < 2: raise ValueError("digits must be >= 2")
    lo = 10**(D-1); hi = 10**D - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2: n += 1
    if n > hi: n -= 2
    if n < lo: n = lo + (1 if lo % 2 else 0)
    if n % 2: n += 1
    return n

def sieve_upto(limit: int) -> List[int]:
    if limit < 2: return []
    sieve = bytearray(b"\x01")*(limit+1)
    sieve[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for p in range(2, r+1):
        if sieve[p]:
            start = p*p; step = p
            sieve[start:limit+1:step] = b"\x00" * (((limit - start)//step) + 1)
    return [i for i, b in enumerate(sieve) if b]

# -------------------------- BPSW primality test --------------------------

_SMALL_PRIMES: List[int] = [2,3,5,7,11,13,17,19,23,29,31,37]

def _is_perfect_square(n: int) -> bool:
    if n < 0: return False
    r = math.isqrt(n); return r*r == n

def _mr_strong_base(n: int, a: int) -> bool:
    if n % 2 == 0: return n == 2
    d = n - 1; s = 0
    while d % 2 == 0: d //= 2; s += 1
    x = pow(a % n, d, n)
    if x == 1 or x == n - 1: return True
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1: return True
    return False

def _mr_strong_base2(n: int) -> bool:
    return _mr_strong_base(n, 2)

def _jacobi(a: int, n: int) -> int:
    assert n > 0 and n % 2 == 1
    a %= n; result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3,5): result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3: result = -result
        a %= n
    return result if n == 1 else 0

def _lucas_strong_prp(n: int) -> bool:
    if n == 2: return True
    if n < 2 or n % 2 == 0 or _is_perfect_square(n): return False
    D = 5
    while True:
        j = _jacobi(D, n)
        if j == -1: break
        if j == 0: return False
        D = -(abs(D) + 2) if D > 0 else abs(D) + 2
    P = 1; Q = (1 - D) // 4
    d = n + 1; s = 0
    while d % 2 == 0: d //= 2; s += 1
    def _lucas_uv_mod(k: int) -> Tuple[int,int]:
        U, V = 0, 2; qk = 1
        bits = bin(k)[2:]; inv2 = pow(2, -1, n)
        for b in bits:
            U2 = (U*V) % n
            V2 = (V*V - 2*qk) % n
            qk = (qk*qk) % n
            if b == '0':
                U, V = U2, V2
            else:
                U = ((P*U2 + V2) * inv2) % n
                V = ((D*U2 + P*V2) * inv2) % n
                qk = (qk * Q) % n
        return U, V
    Ud, Vd = _lucas_uv_mod(d)
    if Vd % n == 0: return True
    for r in range(1, s + 1):
        Vd = (Vd*Vd - 2 * pow(Q, d * (1 << (r - 1)), n)) % n
        if Vd % n == 0: return True
    return False

def is_probable_prime(n: int) -> bool:
    if n < 2: return False
    for p in _SMALL_PRIMES:
        if n == p: return True
        if n % p == 0: return False
    if not _mr_strong_base2(n): return False
    if not _lucas_strong_prp(n): return False
    return True

# ---------------------- Residue filter & pre-sieve -----------------------

def build_wheel_residues_for_n(n: int, wheel_primes: List[int]) -> Dict[int, int]:
    # For candidate subtractor p, if p % r == n % r => q = n - p ≡ 0 (mod r) => reject p
    return {r: (n % r) for r in wheel_primes if r != 2}

def q_pre_sieve(q: int, pre_primes: List[int]) -> bool:
    # Quick rejection of q divisible by small primes before BPSW
    for r in pre_primes:
        if r >= q: break
        if q % r == 0:
            return False
    return True

# --------------------- Learned band ordering state -----------------------

def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)

def init_band_stats(num_bands: int) -> Dict[str, List[int]]:
    return {"trials": [0]*num_bands, "successes": [0]*num_bands}

def apply_decay(stats: Dict[str, List[int]], decay: float) -> None:
    decay = max(0.0, min(1.0, decay))
    if decay >= 0.9999: return
    for k in ("trials","successes"):
        stats[k] = [int(round(v*decay)) for v in stats[k]]

def band_scores(trials: List[int], successes: List[int], smoothing: float) -> List[float]:
    s = max(0.0, smoothing)
    # Expected CTR with Laplace smoothing
    return [(successes[i] + s) / (trials[i] + 2.0*s) for i in range(len(trials))]

# -------------------------- Core GB search (q=n-p) -----------------------

@dataclass(frozen=True)
class ParamSet:
    B: int       # band size
    L: int       # pre-sieve limit for q
    M: int       # max q-checks per n (post-wheel)
    W: Tuple[int, ...]  # wheel primes (tuple for hashing)

@dataclass
class Metrics:
    rate: float
    avg_checks: Optional[float]
    avg_ms: Optional[float]

def make_prime_assets(subs_ceiling: int, wheel: List[int], pre_limit: int):
    subtractor_primes = [p for p in sieve_upto(subs_ceiling) if p % 2 == 1 and p != 1]
    pre_primes = [p for p in sieve_upto(pre_limit) if p >= 3]
    wheel_list = [v for v in wheel if v > 2]
    # residue cache: for each wheel prime r, compute [p % r] for all p once
    res_cache: Dict[int, List[int]] = {r: [p % r for p in subtractor_primes] for r in wheel_list}
    return subtractor_primes, pre_primes, wheel_list, res_cache

def ensure_digit_stats(cache: Dict[str, Dict[str, List[int]]],
                       D: int, num_bands: int) -> Dict[str, List[int]]:
    key = str(D)
    if key not in cache or "trials" not in cache[key] or "successes" not in cache[key]:
        cache[key] = init_band_stats(num_bands)
    for k in ("trials","successes"):
        arr = cache[key][k]
        if len(arr) < num_bands:
            arr.extend([0]*(num_bands - len(arr)))
        elif len(arr) > num_bands:
            cache[key][k] = arr[:num_bands]
    return cache[key]

def band_order_for_digits(stats: Dict[str, List[int]], decay: float, smoothing: float) -> List[int]:
    apply_decay(stats, decay)
    sc = band_scores(stats["trials"], stats["successes"], smoothing)
    order = list(range(len(sc)))
    order.sort(key=lambda b: (-sc[b], b))
    return order

def search_one_n(n: int,
                 subtractor_primes: List[int],
                 pre_primes: List[int],
                 wheel_list: List[int],
                 res_cache: Dict[int, List[int]],
                 band_size: int,
                 band_order: List[int],
                 max_checks: int,
                 stats_to_update: Dict[str, List[int]]) -> Tuple[bool, int, Optional[int], Optional[Tuple[int,int]]]:
    """
    Core learned-band search for a single n.
    Returns (found, checks_used, hit_band, pair_if_found)
    """
    if n % 2 or n < 4:
        return False, 0, None, None

    n_mod = build_wheel_residues_for_n(n, wheel_list)
    total_items = len(subtractor_primes)
    checks_total = 0
    pair = None

    for b in band_order:
        start = b * band_size
        end   = min(start + band_size, total_items)
        if start >= end: continue
        band_checks = 0

        for idx in range(start, end):
            p = subtractor_primes[idx]
            if p >= n - 2:
                # update visited count for this band and early-exit — remaining q<=2
                if band_checks:
                    stats_to_update["trials"][b] += band_checks
                return False, checks_total + band_checks, None, None

            # wheel residue filter (using cached p%r)
            ok = True
            for r, nr in n_mod.items():
                if res_cache[r][idx] == nr:
                    ok = False; break
            if not ok:
                continue

            q = n - p
            # pre-sieve
            if not q_pre_sieve(q, pre_primes):
                band_checks += 1
                if checks_total + band_checks >= max_checks:
                    stats_to_update["trials"][b] += band_checks
                    return False, checks_total + band_checks, None, None
                continue

            # full BPSW check
            band_checks += 1
            if is_probable_prime(q):
                stats_to_update["trials"][b] += band_checks
                stats_to_update["successes"][b] += 1
                pair = (min(p,q), max(p,q))
                return True, checks_total + band_checks, b, pair

            if checks_total + band_checks >= max_checks:
                stats_to_update["trials"][b] += band_checks
                return False, checks_total + band_checks, None, None

        # finished this band
        checks_total += band_checks

    return False, checks_total, None, None

# ---------------------------- Auto-tuner ---------------------------------

def candidate_grid() -> List[ParamSet]:
    wheels = [
        [3,5,7],
        [3,5,7,11],
        [3,5,7,11,13],
        [3,5,7,11,13,17],
    ]
    band_sizes = [32, 64, 128, 256]
    pre_limits = [5000, 10000, 20000, 40000, 80000]
    max_checks = [250, 500, 750, 1000, 1500]
    grid = []
    for B in band_sizes:
        for L in pre_limits:
            for M in max_checks:
                for W in wheels:
                    grid.append(ParamSet(B=B, L=L, M=M, W=tuple(W)))
    return grid

def run_iteration(D: int, params: ParamSet, samples: int, rng: random.Random,
                  subs_ceiling: int, cache_file: str, decay: float, smoothing: float,
                  best_examples_sink: Optional[List[Tuple[int,int,int]]] = None) -> Metrics:
    """
    Evaluate one parameter set for `samples` draws of n with D digits.
    Updates the learned band cache on disk.
    Returns hit rate, avg checks, avg ms.
    Optionally collects up to two example decompositions in best_examples_sink.
    """
    # Build assets for this param set
    subtractor_primes, pre_primes, wheel_list, res_cache = make_prime_assets(subs_ceiling, list(params.W), params.L)
    total_items = len(subtractor_primes)
    band_size = max(1, int(params.B))
    num_bands = (total_items + band_size - 1) // band_size

    # Load/ensure cache
    cache: Dict[str, Dict[str, List[int]]] = load_json(cache_file, {})
    stats = ensure_digit_stats(cache, D, num_bands)
    order = band_order_for_digits(stats, decay, smoothing)

    hits = 0
    checks_sum = 0
    times_ms: List[float] = []
    ex_collected = 0

    for _ in range(samples):
        n = random_even_with_digits(D, rng)
        t0 = time.perf_counter()
        found, checks, hit_band, pair = search_one_n(
            n, subtractor_primes, pre_primes, wheel_list, res_cache,
            band_size, order, params.M, stats
        )
        dt_ms = (time.perf_counter() - t0)*1000.0
        times_ms.append(dt_ms)
        if found:
            hits += 1
            checks_sum += checks
            if best_examples_sink is not None and ex_collected < 2 and pair is not None:
                ex_collected += 1
                # store (p, q, n)
                best_examples_sink.append((pair[0], pair[1], n))
        # persist cache occasionally to avoid data loss in long runs
        # (cheap write; safe)
    save_json(cache_file, cache)

    rate = hits / samples if samples else 0.0
    avg_checks = (checks_sum / hits) if hits else None
    avg_ms = (sum(times_ms) / len(times_ms)) if times_ms else None
    return Metrics(rate=rate, avg_checks=avg_checks, avg_ms=avg_ms)

# ------------- Deterministic selection strategies (tuner) ----------------

def metric_key(m: Metrics):
    # Primary objective: minimize avg_checks, then avg_ms, then maximize rate
    return (m.avg_checks if m.avg_checks is not None else 1e18,
            m.avg_ms if m.avg_ms is not None else 1e18,
            -m.rate)

def successive_halving(D: int, grid: List[ParamSet], sha_rungs: int, keep_ratio: float,
                       base_budget: int, rng: random.Random, subs_ceiling: int,
                       cache_file: str, decay: float, smoothing: float) -> Tuple[ParamSet, Metrics]:
    cohort = list(grid)
    budget = max(1, base_budget)
    winner_p, winner_m = None, None
    for _ in range(max(1, sha_rungs)):
        scored: List[Tuple[ParamSet, Metrics]] = []
        for p in cohort:
            m = run_iteration(D, p, budget, rng, subs_ceiling, cache_file, decay, smoothing)
            scored.append((p, m))
        scored.sort(key=lambda pm: metric_key(pm[1]))
        keep = max(1, int(len(scored) * max(0.05, min(0.95, keep_ratio))))
        cohort = [p for (p, _) in scored[:keep]]
        winner_p, winner_m = scored[0]
        budget *= 2
    return winner_p, winner_m

def greedy_driver(D: int, grid: List[ParamSet], iters: int, base_budget: int,
                  rng: random.Random, subs_ceiling: int, cache_file: str,
                  decay: float, smoothing: float,
                  log_csv: Optional[str], examples_out: List[Tuple[int,int,int]]):
    # Seed with a quick SHA to avoid a terrible starting point
    best_p, best_m = successive_halving(D, grid, sha_rungs=2, keep_ratio=0.5, base_budget=base_budget,
                                        rng=rng, subs_ceiling=subs_ceiling, cache_file=cache_file,
                                        decay=decay, smoothing=smoothing)
    for it in range(1, iters+1):
        m = run_iteration(D, best_p, args.samples, rng, subs_ceiling, cache_file, decay, smoothing,
                          best_examples_sink=examples_out)
        print(f"rate={m.rate:.3f}  avg_checks={(m.avg_checks if m.avg_checks is not None else float('nan')):.1f}  avg_ms={(m.avg_ms if m.avg_ms is not None else float('nan')):.2f}")
        if log_csv:
            append_log(log_csv, it, D, m)
        # challenger from a single quick rung
        cand_p, cand_m = successive_halving(D, grid, sha_rungs=1, keep_ratio=0.25, base_budget=base_budget,
                                            rng=rng, subs_ceiling=subs_ceiling, cache_file=cache_file,
                                            decay=decay, smoothing=smoothing)
        if metric_key(cand_m) < metric_key(best_m):
            best_p, best_m = cand_p, cand_m
    return best_p, best_m

def sha_driver(D: int, grid: List[ParamSet], iters: int, sha_rungs: int, keep_ratio: float,
               base_budget: int, rng: random.Random, subs_ceiling: int, cache_file: str,
               decay: float, smoothing: float, log_csv: Optional[str], examples_out: List[Tuple[int,int,int]]):
    # Each iteration, run a full SHA and emit its winner
    last_p, last_m = None, None
    for it in range(1, iters+1):
        p, m = successive_halving(D, grid, sha_rungs, keep_ratio, base_budget,
                                  rng, subs_ceiling, cache_file, decay, smoothing)
        print(f"rate={m.rate:.3f}  avg_checks={(m.avg_checks if m.avg_checks is not None else float('nan')):.1f}  avg_ms={(m.avg_ms if m.avg_ms is not None else float('nan')):.2f}")
        if log_csv:
            append_log(log_csv, it, D, m)
        # Run once more with full samples to gather example decompositions
        _ = run_iteration(D, p, max(1, int(0.25*args.samples)), rng, subs_ceiling, cache_file, decay, smoothing,
                          best_examples_sink=examples_out)
        last_p, last_m = p, m
    return last_p, last_m

def hybrid_driver(D: int, grid: List[ParamSet], iters: int, eps_start: float, eps_final: float,
                  eps_iters: int, base_budget: int, rng: random.Random, subs_ceiling: int,
                  cache_file: str, decay: float, smoothing: float,
                  log_csv: Optional[str], examples_out: List[Tuple[int,int,int]]):
    eps = eps_start
    best_p, best_m = successive_halving(D, grid, sha_rungs=2, keep_ratio=0.5, base_budget=base_budget,
                                        rng=rng, subs_ceiling=subs_ceiling, cache_file=cache_file,
                                        decay=decay, smoothing=smoothing)
    for it in range(1, iters+1):
        if rng.random() < eps:
            # explore a random config briefly; do not replace best here
            cand = rng.choice(grid)
            _ = run_iteration(D, cand, base_budget, rng, subs_ceiling, cache_file, decay, smoothing)
        m = run_iteration(D, best_p, args.samples, rng, subs_ceiling, cache_file, decay, smoothing,
                          best_examples_sink=examples_out)
        print(f"rate={m.rate:.3f}  avg_checks={(m.avg_checks if m.avg_checks is not None else float('nan')):.1f}  avg_ms={(m.avg_ms if m.avg_ms is not None else float('nan')):.2f}")
        if log_csv:
            append_log(log_csv, it, D, m)
        if it <= eps_iters:
            eps = eps_start + (eps_final - eps_start) * (it / max(1, eps_iters))
    return best_p, m

# ------------------------------- Logging ---------------------------------

def init_log(path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter","digits","rate","avg_checks","avg_ms"])

def append_log(path: str, it: int, D: int, m: Metrics):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            it, D,
            f"{m.rate:.6f}",
            "" if m.avg_checks is None else f"{m.avg_checks:.3f}",
            "" if m.avg_ms is None else f"{m.avg_ms:.3f}",
        ])

# ------------------------------- CLI ------------------------------------

def parse_digits_list(digits: Optional[int], sweep: Optional[str]) -> List[int]:
    if digits is not None:
        return [digits]
    a,b,c = [int(x) for x in sweep.split(":")]
    step = c if c != 0 else 1
    if step > 0:  return list(range(a, b+1, step))
    else:         return list(range(a, b-1, step))

parser = argparse.ArgumentParser(prog="gb_auto",
    description="Self-tuning subtractor-only Goldbach engine (deterministic).")
grp = parser.add_mutually_exclusive_group(required=True)
grp.add_argument("--digits", type=int, help="Single digit size to auto-tune.")
grp.add_argument("--digits-sweep", type=str, help="Sweep 'A:B:C' (inclusive).")
parser.add_argument("--samples", type=int, default=300, help="Samples per evaluation (default 300).")
parser.add_argument("--iterations", type=int, default=20, help="Tuning iterations per digit (default 20).")
parser.add_argument("--seed", type=int, default=2025, help="PRNG seed.")
parser.add_argument("--subs-ceiling", type=int, default=300000, help="Subtractor primes ceiling.")
parser.add_argument("--decay", type=float, default=1.0, help="Per-use band stats decay (1.0 = none).")
parser.add_argument("--smoothing", type=float, default=1.0, help="Laplace smoothing for CTR.")
parser.add_argument("--tuner", choices=["greedy","sha","hybrid"], default="sha", help="Selection strategy.")
parser.add_argument("--epsilon-start", type=float, default=0.10, help="(hybrid) start epsilon.")
parser.add_argument("--epsilon-final", type=float, default=0.00, help="(hybrid) final epsilon.")
parser.add_argument("--epsilon-iters", type=int, default=8, help="(hybrid) iterations to decay epsilon.")
parser.add_argument("--sha-rungs", type=int, default=3, help="SHA rungs.")
parser.add_argument("--sha-keep-ratio", type=float, default=0.5, help="SHA keep ratio per rung.")
parser.add_argument("--sha-base-budget", type=int, default=250, help="SHA base samples per config.")
parser.add_argument("--cache-file", type=str, default="subs_learn_cache.json", help="Band CTR cache path.")
parser.add_argument("--best-file", type=str, default="gb_auto_best.json", help="Best-per-digit config path.")
parser.add_argument("--log-csv", type=str, default=None, help="Optional iteration CSV log.")
parser.add_argument("--examples", type=int, default=2, help="Number of decompositions to print at end (default 2).")
args = parser.parse_args()

# ------------------------------- Main ------------------------------------

rng = random.Random(args.seed)
grid = candidate_grid()
digits_list = parse_digits_list(args.digits, args.digits_sweep)
if args.log_csv: init_log(args.log_csv)

# Warm-start with previously best configs (if any)
best_store = load_json(args.best_file, {})  # key=str(D) -> {"B":..,"L":..,"M":..,"W":[..]}
def best_for_digit(D: int) -> Optional[ParamSet]:
    rec = best_store.get(str(D))
    if not rec: return None
    try:
        return ParamSet(B=int(rec["B"]), L=int(rec["L"]), M=int(rec["M"]), W=tuple(int(x) for x in rec["W"]))
    except Exception:
        return None

def promote_best_to_front(g: List[ParamSet], best_p: Optional[ParamSet]) -> List[ParamSet]:
    if best_p is None: return g
    # Stable order; move exact match to front if present
    for i, p in enumerate(g):
        if p == best_p:
            return [p] + g[:i] + g[i+1:]
    return [best_p] + g

# Run per digit
final_examples: List[Tuple[int,int,int]] = []  # (p,q,n) up to 2 per last digit processed
for D in digits_list:
    maybe_best = best_for_digit(D)
    use_grid = promote_best_to_front(grid, maybe_best)

    # choose driver
    examples_out: List[Tuple[int,int,int]] = []
    if args.tuner == "greedy":
        winner_p, winner_m = greedy_driver(D, use_grid, args.iterations, args.sha_base_budget,
                                           rng, args.subs_ceiling, args.cache_file,
                                           args.decay, args.smoothing, args.log_csv, examples_out)
    elif args.tuner == "sha":
        winner_p, winner_m = sha_driver(D, use_grid, args.iterations, args.sha_rungs, args.sha_keep_ratio,
                                        args.sha_base_budget, rng, args.subs_ceiling, args.cache_file,
                                        args.decay, args.smoothing, args.log_csv, examples_out)
    else:
        winner_p, winner_m = hybrid_driver(D, use_grid, args.iterations, args.epsilon_start, args.epsilon_final,
                                           args.epsilon_iters, args.sha_base_budget, rng, args.subs_ceiling,
                                           args.cache_file, args.decay, args.smoothing, args.log_csv, examples_out)

    # Persist best config for this digit
    if winner_p is not None:
        best_store[str(D)] = {"B": winner_p.B, "L": winner_p.L, "M": winner_p.M, "W": list(winner_p.W)}
        save_json(args.best_file, best_store)

    # Save up to args.examples examples from this digit pass for final display
    for tup in examples_out:
        if len(final_examples) < args.examples:
            final_examples.append(tup)
        else:
            break

# Print exactly two (or args.examples) Goldbach decompositions found
# If we didn't collect enough during tuning (rare), generate a couple now using the last digit & best config.
if len(final_examples) < args.examples and digits_list:
    D = digits_list[-1]
    rec = best_store.get(str(D))
    if rec:
        best_p = ParamSet(B=int(rec["B"]), L=int(rec["L"]), M=int(rec["M"]), W=tuple(int(x) for x in rec["W"]))
        subtractor_primes, pre_primes, wheel_list, res_cache = make_prime_assets(args.subs_ceiling, list(best_p.W), best_p.L)
        total_items = len(subtractor_primes)
        band_size = max(1, int(best_p.B))
        num_bands = (total_items + band_size - 1) // band_size
        cache: Dict[str, Dict[str, List[int]]] = load_json(args.cache_file, {})
        stats = ensure_digit_stats(cache, D, num_bands)
        order = band_order_for_digits(stats, args.decay, args.smoothing)
        # produce decompositions
        tries = 0
        while len(final_examples) < args.examples and tries < 1000:
            n = random_even_with_digits(D, rng)
            _stats_shadow = {"trials":[0]*num_bands, "successes":[0]*num_bands}  # don't perturb cache here
            found, checks, hit_band, pair = search_one_n(
                n, subtractor_primes, pre_primes, wheel_list, res_cache,
                band_size, order, best_p.M, _stats_shadow
            )
            if found and pair:
                final_examples.append((pair[0], pair[1], n))
            tries += 1

# Exactly N example lines
for (p, q, n) in final_examples[:args.examples]:
    print(f"{p} + {q} = {n}")
