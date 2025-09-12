#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gbopt.py — Subtractor-only Goldbach finder (per-n adaptive)
#   - Per-n (signature) adaptive wheel strength k
#   - Per-n (signature) adaptive pre-sieve length L + globally learned sieve order
#   - Deterministic ascending subtractors (p increasing)
#   - Pre-sieve q with small primes before BPSW
#   - Early break at midpoint: one-direction search (stop when 2*p > n)
#   - Precomputed residue cache for p % r to avoid repeated mod
#   - ZERO hot-loop overhead learning (choices made once per n)
#   - Outputs per-digit avg_checks_to_hit and avg_ms_per_hit (compatible with old CLI)
#
# (c) 2025 — MIT-style permissive for this prototype

from __future__ import annotations
import argparse, math, random, csv, json, os, time, hashlib
from typing import List, Tuple, Optional, Dict, Any

# ----------------------------- Small helpers -----------------------------

def _wilson_ci(h: int, n: int, z: float = 1.96) -> Tuple[float,float]:
    if n == 0: return (0.0, 1.0)
    p = h/n; denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (max(0.0, center-half), min(1.0, center+half))

def random_even_with_digits(D: int, rng: random.Random) -> int:
    if D < 2: raise ValueError("digits must be >= 2")
    lo = 10**(D-1); hi = 10**D - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2: n += 1
    if n > hi: n -= 2
    if n < lo: n = lo + (1 if lo % 2 else 0)
    if n % 2: n += 1
    return n

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

# ---------------------- Sieve & residue utilities -----------------------

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

def build_wheel_residues_for_n(n: int, wheel_primes: List[int]) -> Dict[int, int]:
    return {r: (n % r) for r in wheel_primes if r != 2}

def q_pre_sieve(q: int, pre_primes: List[int]) -> Tuple[bool, Optional[int]]:
    """Return (passes, killer_prime_or_None)."""
    for r in pre_primes:
        if r >= q: break
        if q % r == 0:
            return (False, r)
    return (True, None)

# ------------------------- Persistence helpers --------------------------

def load_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)

# -------------------------- Learned band ordering -----------------------

def init_band_stats(num_bands: int) -> Dict[str, List[int]]:
    return {"trials":[0]*num_bands, "successes":[0]*num_bands}

def band_scores(trials: List[int], successes: List[int], smoothing: float = 1.0) -> List[float]:
    return [(successes[b]+smoothing)/((trials[b]+2.0*smoothing)) for b in range(len(trials))]

def apply_decay_vec(vec: List[float], decay: float, as_int: bool=False) -> List[Any]:
    if decay >= 0.9999: return vec
    if as_int:
        return [int(round(v*decay)) for v in vec]
    else:
        return [v*decay for v in vec]

# --------------------- Adaptive wheels (k for p; L for sieve) ------------

def make_k_options(L: int) -> List[int]:
    # Menu of subtractor wheel strengths (how many wheel primes to use)
    return sorted(set([0, min(2,L), min(4,L), L]))

def make_L_options(max_pre: int) -> List[int]:
    # Menu of pre-sieve lengths; cap by available primes
    base = [0, 16, 32, 64, 128, 256]
    return [x for x in base if x <= max_pre] or [0]

def choose_arm(trials: List[int], ms_sum: List[float], arms: List[int], rng: random.Random, explore_eps: float=0.05) -> int:
    if rng.random() < explore_eps:
        return rng.choice(arms)
    best_arm, best = arms[0], float('inf')
    for i,_ in enumerate(arms):
        t = trials[i]
        m = (ms_sum[i]/t) if t>0 else (1e6 - 1e3*i)  # encourage unseen early; slight bias to smaller index
        if m < best: best, best_arm = m, arms[i]
    return best_arm

# ------------------------ Signature bucketing (per-n) --------------------

def signature_of_n(n: int, sig_primes: List[int], extra_bits: int=10) -> str:
    """
    Cheap per-n signature:
      - residues mod selected small primes (>=3)
      - low bits modulo 2**extra_bits (captures some power-of-two structure)
    Returned as a compact string key; can be feature-hashed if desired.
    """
    parts = [f"m{p}={n%p}" for p in sig_primes if p>=3]
    mask = (1<<extra_bits)-1
    parts.append(f"b2={n & mask}")
    raw = "|".join(parts)
    # Feature-hash to keep key space modest and JSON-friendly
    h = hashlib.blake2b(raw.encode(), digest_size=6).hexdigest()
    return f"sig:{h}"

# ------------------------------ Core program -----------------------------

def rationale() -> str:
    return (
        "\n*** Goldbach (per-n adaptive wheels) ***\n\n"
        "Given even n, try ascending subtractor primes p so that q=n-p is prime.\n"
        "Speedups:\n"
        "  • Per-n adaptive wheel strength (k) chosen by a residue signature of n.\n"
        "  • Pre-sieve has its own adaptive 'wheel': per-n choice of length L and\n"
        "    a globally learned order of small primes ranked by observed kill-rate.\n"
        "  • Early break at midpoint: stop when 2*p > n (one-direction only).\n"
        "  • Learned band ordering with per-digit CTR stats (zero hot-loop overhead).\n"
        "Any hit is validated by BPSW (Baillie–PSW primality test).\n"
    )

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="gbopt",
        description="Goldbach decompositions with per-n adaptive wheels and learned ordering."
    )
    ap.add_argument("--digits", type=int, help="Single digit size (mutually exclusive with --sweep).")
    ap.add_argument("--count", type=int, default=10, help="How many n for single-run (default 10).")
    ap.add_argument("--sweep", type=lambda s: _parse_sweep(s), default=None, help="Range 'start:end:step' for digit sweep.")
    ap.add_argument("--samples", type=int, default=100, help="How many n per digit in sweep (default 100).")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed (default 2025).")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-n prints; show only summary rows.")
    ap.add_argument("--csv", type=str, default=None, help="Write sweep CSV summary.")
    ap.add_argument("--ci", action="store_true", help="Show Wilson 95% CI for hit rate.")
    ap.add_argument("--why", action="store_true", help="Print a short rationale and exit.")

    # Subtractor config
    ap.add_argument("--subs-ceiling", type=int, default=300000,
                    help="Upper bound for subtractor primes (default 300000).")
    ap.add_argument("--subs-max-checks", type=int, default=2000,
                    help="Max subtractor primes to examine per n (after wheel filtering) (default 2000).")

    # Wheels & sieve config
    ap.add_argument("--wheel-primes", type=str, default="3,5,7,11,13,17,19,23",
                    help="Comma-separated small primes for residue wheel.")
    ap.add_argument("--pre-sieve-limit", type=int, default=20000,
                    help="Pre-sieve q with primes up to this value before BPSW (default 20000).")
    ap.add_argument("--sieve-max-order", type=int, default=4096,
                    help="Max number of pre-sieve primes whose order we track globally (default 4096).")

    # Learned band ordering (per-digit, as before)
    ap.add_argument("--band-size", type=int, default=64,
                    help="Contiguous band size for learned ordering (default 64).")

    # Cache / learning parameters
    ap.add_argument("--cache-file", type=str, default="subs_learn_cache.json",
                    help="Path to JSON cache (will be extended for per-n buckets).")
    ap.add_argument("--decay", type=float, default=1.0,
                    help="Per-run multiplicative decay for stats (e.g., 0.99). Default 1.0 = no decay.")
    ap.add_argument("--smoothing", type=float, default=1.0,
                    help="Laplace smoothing for band scores (default 1.0).")
    ap.add_argument("--explore-eps", type=float, default=0.05,
                    help="Exploration probability for k/L bandits (default 0.05).")
    ap.add_argument("--sig-extra-bits", type=int, default=10,
                    help="Extra low bits included in per-n signature (default 10).")

    args = ap.parse_args()
    if args.why:
        print(rationale()); return

    rng = random.Random(args.seed)

    # Prepare subtractor primes
    subtractor_primes = [p for p in sieve_upto(args.subs_ceiling) if p % 2 == 1 and p != 1]

    # Build wheel list
    wheel_list: List[int] = []
    for tok in args.wheel_primes.split(","):
        tok = tok.strip()
        if not tok: continue
        v = int(tok)
        if v <= 2: continue
        wheel_list.append(v)
    W = len(wheel_list)
    k_opts = make_k_options(W)  # e.g., [0,2,4,8] for default list length 8

    # Precompute residues p % r once for each wheel prime
    res_cache: Dict[int, List[int]] = {r: [p % r for p in subtractor_primes] for r in wheel_list}

    # Pre-sieve base primes (exclude 2; keep odd primes), and initialize global order/stats
    pre_primes_full = [p for p in sieve_upto(args.pre_sieve_limit) if p >= 3]
    max_track = min(args.sieve_max_order, len(pre_primes_full))
    # global learned order state lives in cache["global_sieve"]
    band_size = max(1, args.band_size)
    total_items = len(subtractor_primes)
    num_bands = (total_items + band_size - 1) // band_size

    # Load / init cache
    cache: Dict[str, Any] = load_cache(args.cache_file)
    if "global_sieve" not in cache:
        cache["global_sieve"] = {
            "order": pre_primes_full[:max_track],       # start with natural order
            "trials": [0]*max_track,                    # how often each prime tested
            "kills": [0]*max_track,                     # how often it divided q
        }
    else:
        # sanitize if pre_sieve_limit changed
        g = cache["global_sieve"]
        # rebuild order vector to be a subset of current pre_primes_full, preserving learned ranking
        current_set = set(pre_primes_full)
        learned = [p for p in g.get("order", []) if p in current_set]
        tail = [p for p in pre_primes_full if p not in learned]
        new_order = (learned + tail)[:max_track]
        # re-dimension stats
        old_index = {p:i for i,p in enumerate(g.get("order", []))}
        trials = [0]*len(new_order); kills = [0]*len(new_order)
        for i,p in enumerate(new_order):
            j = old_index.get(p)
            if j is not None:
                if j < len(g.get("trials", [])): trials[i] = int(g["trials"][j])
                if j < len(g.get("kills", [])):  kills[i]  = int(g["kills"][j])
        cache["global_sieve"] = {"order": new_order, "trials": trials, "kills": kills}

    # Digit-level band stats (kept as before)
    if "digits" not in cache: cache["digits"] = {}
    def get_digit_stats(D: int) -> Dict[str, Any]:
        dkey = str(D)
        entry = cache["digits"].setdefault(dkey, {})
        # bands
        if "trials" not in entry or "successes" not in entry:
            entry["trials"] = [0]*num_bands
            entry["successes"] = [0]*num_bands
        for k in ("trials","successes"):
            arr = entry[k]
            if len(arr) < num_bands: arr.extend([0]*(num_bands-len(arr)))
            elif len(arr) > num_bands: entry[k] = arr[:num_bands]
        return entry

    # Signature-level (per-n bucket) stats
    if "sig_buckets" not in cache: cache["sig_buckets"] = {}
    def get_sig_stats(sig: str, k_opts: List[int], L_opts: List[int]) -> Dict[str, Any]:
        sb = cache["sig_buckets"].setdefault(sig, {})
        # k bandit
        if sb.get("k_opts") != k_opts:
            sb["k_opts"] = list(k_opts)
            sb["k_trials"] = [0]*len(k_opts)
            sb["k_ms_sum"] = [0.0]*len(k_opts)
            sb["k_checks_sum"] = [0]*len(k_opts)
        else:
            if "k_trials" not in sb: sb["k_trials"] = [0]*len(k_opts)
            if "k_ms_sum" not in sb: sb["k_ms_sum"] = [0.0]*len(k_opts)
            if "k_checks_sum" not in sb: sb["k_checks_sum"] = [0]*len(k_opts)
        # L bandit
        if sb.get("L_opts") != L_opts:
            sb["L_opts"] = list(L_opts)
            sb["L_trials"] = [0]*len(L_opts)
            sb["L_ms_sum"] = [0.0]*len(L_opts)
            sb["L_checks_sum"] = [0]*len(L_opts)
        else:
            if "L_trials" not in sb: sb["L_trials"] = [0]*len(L_opts)
            if "L_ms_sum" not in sb: sb["L_ms_sum"] = [0.0]*len(L_opts)
            if "L_checks_sum" not in sb: sb["L_checks_sum"] = [0]*len(L_opts)
        return sb

    def decay_all(decay: float) -> None:
        d = max(0.0, min(1.0, decay))
        if d >= 0.9999: return
        # digit bands
        for entry in cache["digits"].values():
            entry["trials"] = apply_decay_vec(entry["trials"], d, as_int=True)
            entry["successes"] = apply_decay_vec(entry["successes"], d, as_int=True)
        # signature buckets
        for sb in cache["sig_buckets"].values():
            for name in ("k_trials","k_checks_sum","L_trials","L_checks_sum"):
                if name in sb: sb[name] = apply_decay_vec(sb[name], d, as_int=True)
            for name in ("k_ms_sum","L_ms_sum"):
                if name in sb: sb[name] = apply_decay_vec(sb[name], d, as_int=False)
        # global sieve
        g = cache["global_sieve"]
        g["trials"] = apply_decay_vec(g["trials"], d, as_int=True)
        g["kills"]  = apply_decay_vec(g["kills"], d, as_int=True)

    # Pre-sieve L options depend on available primes
    L_opts = make_L_options(len(cache["global_sieve"]["order"]))

    def band_scores_for(D: int) -> Tuple[List[int], Dict[str, Any]]:
        entry = get_digit_stats(D)
        # decay once per outer call
        decay_all(args.decay)
        sc = band_scores(entry["trials"], entry["successes"], smoothing=max(0.0, args.smoothing))
        order = list(range(num_bands))
        order.sort(key=lambda b: (-sc[b], b))
        return order, entry

    def record_band_updates(entry: Dict[str, Any], visited_band_trials: Dict[int, int], hit_band: Optional[int]) -> None:
        for b, t in visited_band_trials.items():
            entry["trials"][b] += int(t)
        if hit_band is not None:
            entry["successes"][hit_band] += 1

    # --------------------------- Per-digit or sweep -----------------------

    def run_for_digits(D: int, trials: int) -> Tuple[int,int,List[int],List[float]]:
        hits = 0
        checks_on_hits: List[int] = []
        ms_on_hits: List[float] = []

        band_order, digit_entry = band_scores_for(D)

        for _ in range(trials):
            n = random_even_with_digits(D, rng)

            # ----- Per-n choices (made once; zero hot-loop cost) -----
            sig = signature_of_n(n, wheel_list, extra_bits=args.sig_extra_bits)
            sb = get_sig_stats(sig, k_opts, L_opts)

            k = choose_arm(sb["k_trials"], sb["k_ms_sum"], k_opts, rng, args.explore_eps)
            L = choose_arm(sb["L_trials"], sb["L_ms_sum"], L_opts, rng, args.explore_eps)

            use_wheel = k > 0
            n_mod = build_wheel_residues_for_n(n, wheel_list[:k]) if use_wheel else {}

            # snapshot the first L primes from global learned order
            g = cache["global_sieve"]
            sieve_order = g["order"][:L]

            visited_band_trials: Dict[int, int] = {}
            checks_total = 0
            found = False
            hit_band = None
            t0 = time.perf_counter()

            if n % 2 == 0 and n >= 4:
                for b in band_order:
                    start = b * band_size
                    end   = min(start + band_size, total_items)
                    if start >= end: continue
                    band_checks = 0
                    broke_mid = False
                    for idx in range(start, end):
                        p = subtractor_primes[idx]
                        if p * 2 > n:
                            broke_mid = True
                            break
                        # Wheel prefilter (subtractor side)
                        if use_wheel:
                            wheel_ok = True
                            for r, nr in n_mod.items():
                                if res_cache[r][idx] == nr:
                                    wheel_ok = False; break
                            if not wheel_ok:
                                continue

                        q = n - p
                        # Pre-sieve using current learned order (first L primes)
                        if L > 0:
                            passes, killer = q_pre_sieve(q, sieve_order)
                            # update global sieve stats for primes we actually touched:
                            # we approximate by crediting 1 trial to each tested prime up to the killer (or all L if passes)
                            if killer is not None:
                                # trials credited to primes up to and incl. killer
                                for i, rp in enumerate(sieve_order):
                                    gi = g["order"].index(rp)
                                    g["trials"][gi] += 1
                                    if rp == killer:
                                        g["kills"][gi] += 1
                                        break
                                # pre-sieve failed; no BPSW; count a check and continue
                                band_checks += 1
                                if checks_total + band_checks >= args.subs_max_checks:
                                    visited_band_trials[b] = visited_band_trials.get(b, 0) + band_checks
                                    checks_total += band_checks
                                    record_band_updates(digit_entry, visited_band_trials, None)
                                    found = False; hit_band = None
                                    break
                                continue
                            else:
                                # passed: all L primes were tested
                                for rp in sieve_order:
                                    gi = g["order"].index(rp)
                                    g["trials"][gi] += 1

                        # BPSW
                        band_checks += 1
                        if is_probable_prime(q):
                            visited_band_trials[b] = visited_band_trials.get(b, 0) + band_checks
                            checks_total += band_checks
                            record_band_updates(digit_entry, visited_band_trials, b)
                            found = True; hit_band = b
                            break
                        if checks_total + band_checks >= args.subs_max_checks:
                            visited_band_trials[b] = visited_band_trials.get(b, 0) + band_checks
                            checks_total += band_checks
                            record_band_updates(digit_entry, visited_band_trials, None)
                            found = False; hit_band = None
                            break

                    # band finished or broke
                    if found or checks_total >= args.subs_max_checks or (broke_mid and start < end):
                        if not found and broke_mid:
                            # if we broke on midpoint, still record the partial band checks
                            if band_checks:
                                visited_band_trials[b] = visited_band_trials.get(b, 0) + band_checks
                                checks_total += band_checks
                                record_band_updates(digit_entry, visited_band_trials, None)
                        break

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Update per-n bucket bandits
            ki = k_opts.index(k)
            sb["k_trials"][ki] += 1
            sb["k_ms_sum"][ki] += elapsed_ms
            sb["k_checks_sum"][ki] += checks_total

            Li = L_opts.index(L)
            sb["L_trials"][Li] += 1
            sb["L_ms_sum"][Li] += elapsed_ms
            sb["L_checks_sum"][Li] += checks_total

            # Periodically refresh global sieve order by utility = kills / (trials+1)
            # (very cheap; keeps front of the list useful)
            # Note: we do not change 'order' length here; only reorder within tracked window
            if sum(g["trials"]) % 256 == 0 and sum(g["trials"]) > 0:
                util = [(g["kills"][i] / (g["trials"][i] + 1.0), i) for i in range(len(g["order"]))]
                util.sort(key=lambda x: -x[0])
                new_order = [g["order"][i] for _,i in util]
                new_trials = [g["trials"][i] for _,i in util]
                new_kills  = [g["kills"][i]  for _,i in util]
                g["order"], g["trials"], g["kills"] = new_order, new_trials, new_kills

            if found:
                hits += 1
                checks_on_hits.append(checks_total)
                ms_on_hits.append(elapsed_ms)

        # persist cache updates for this D
        save_cache(args.cache_file, cache)
        return hits, trials, checks_on_hits, ms_on_hits

    # ------------------------------ Sweep mode ------------------------------
    if args.sweep is not None:
        digits_space = args.sweep
        rows: List[Dict[str, object]] = []
        if not args.quiet:
            print("\n# Learned subtractor-only sweep (per-n adaptive)")
            print(f"# subs_ceiling={args.subs_ceiling} | subs_max_checks={args.subs_max_checks}")
            print(f"# wheel={wheel_list} | pre_sieve_limit={args.pre_sieve_limit} | band_size={args.band_size} | cache={args.cache_file} | decay={args.decay} | smoothing={args.smoothing}")
            print(f"# samples per digit={args.samples} | seed={args.seed}\n")

        for D in digits_space:
            hits, total, checks_on_hits, ms_on_hits = run_for_digits(D, args.samples)
            rate = hits/total if total else 0.0
            avg_checks = (sum(checks_on_hits)/len(checks_on_hits)) if checks_on_hits else None
            avg_ms = (sum(ms_on_hits)/len(ms_on_hits)) if ms_on_hits else None

            show_hit = round(rate, 3) < 1.000
            if args.ci and show_hit:
                lo95, hi95 = _wilson_ci(hits, total)
                ci_txt = f"  CI95=[{lo95:.3f},{hi95:.3f}]"
            else:
                ci_txt = ""

            hit_str = f"hit_rate={rate:6.3f}{ci_txt}  " if show_hit else ""
            avg_checks_txt = ('%.1f' % avg_checks) if avg_checks is not None else '—'
            avg_ms_txt = ('%.1f' % avg_ms) if avg_ms is not None else '—'
            print(f"{D:>3}d  {hit_str}avg_checks_to_hit={avg_checks_txt:>6}  avg_ms_per_hit={avg_ms_txt:>6}")

            row = {
                "digits": D,
                "samples": total,
                "hits": hits,
                "misses": total - hits,
                "hit_rate": round(rate, 6),
                "avg_checks_to_hit": round(avg_checks, 3) if avg_checks is not None else None,
                "avg_ms_per_hit": round(avg_ms, 3) if avg_ms is not None else None,
                "subs_ceiling": args.subs_ceiling,
                "subs_max_checks": args.subs_max_checks,
                "wheel_primes": ",".join(map(str, wheel_list)),
                "pre_sieve_limit": args.pre_sieve_limit,
                "band_size": args.band_size,
                "cache_file": args.cache_file,
                "decay": args.decay,
                "smoothing": args.smoothing,
                "seed": args.seed,
            }
            if args.ci and show_hit:
                row["hit_rate_ci95_low"] = round(lo95, 6)
                row["hit_rate_ci95_high"] = round(hi95, 6)
            rows.append(row)

        if args.csv and rows:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader(); writer.writerows(rows)
            print(f"[saved] CSV -> {args.csv}")
        return

    # ----------------------------- Single-run mode --------------------------
    if args.digits is None:
        print(rationale()); raise SystemExit("Provide --digits or --sweep. Use --why for the rationale.")

    D = args.digits
    print(f"# Goldbach | D={D} | count={args.count} | seed={args.seed}")
    print(f"# subs_ceiling={args.subs_ceiling} | subs_max_checks={args.subs_max_checks} "
          f"| wheel={wheel_list} | pre_sieve_limit={args.pre_sieve_limit} | band_size={args.band_size} | cache={args.cache_file}")
    hits, total, checks_on_hits, ms_on_hits = run_for_digits(D, args.count)
    rate = hits/total if total else 0.0
    avg_checks = (sum(checks_on_hits)/len(checks_on_hits)) if checks_on_hits else None
    avg_ms = (sum(ms_on_hits)/len(ms_on_hits)) if ms_on_hits else None

    show_hit = round(rate, 3) < 1.000
    avg_checks_txt = ('%.1f' % avg_checks) if avg_checks is not None else '—'
    avg_ms_txt = ('%.1f' % avg_ms) if avg_ms is not None else '—'

    if show_hit:
        print(f"hit_rate={rate:.3f}  avg_checks_to_hit={avg_checks_txt}  avg_ms_per_hit={avg_ms_txt}")
    else:
        print(f"avg_checks_to_hit={avg_checks_txt}  avg_ms_per_hit={avg_ms_txt}")

# ------------------------------- Utilities --------------------------------

def _parse_sweep(s: str) -> List[int]:
    try:
        a, b, c = [int(x) for x in s.split(":")]
        if c == 0: raise ValueError
        rng = range(a, b + (1 if c > 0 else -1), c)
        out = [v for v in rng if v > 0]
        if not out: raise ValueError
        return out
    except Exception:
        raise argparse.ArgumentTypeError(f"--sweep must be 'start:end:step' (got {s!r})")

if __name__ == "__main__":
    main()
