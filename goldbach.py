#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# subs_gb_learn.py — Subtractor-only Goldbach finder
# (learned band ordering + self-tuning wheel; one-direction search)
#
# - Deterministic ascending subtractors (p increasing)
# - Self-tuning wheel residue prefilter (per digit, learns whether using the wheel is faster)
# - Pre-sieve q with small primes before BPSW
# - Early break at midpoint: only search one direction (stop when 2*p > n)
# - Precomputed residue cache for p % r to avoid repeated mod
# - ZERO hot-loop overhead "learning": per-digit band scores cached on disk
# - Outputs per-digit avg_checks_to_hit and avg_ms_per_hit
# - Hit rate & CI are hidden when the display would be 1.000
#
# (c) 2025 — MIT-style permissive for this prototype

from __future__ import annotations
import argparse, math, random, csv, json, os, time
from typing import List, Tuple, Optional, Dict

# ----------------------------- Small helpers -----------------------------

def _wilson_ci(h: int, n: int, z: float = 1.96) -> Tuple[float,float]:
    if n == 0: return (0.0, 1.0)
    p = h/n; denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (max(0.0, center-half), min(1.0, center+half))

def _ascii_bar(frac: float, width: int = 40) -> str:
    n = max(0, min(width, int(round(frac * width))))
    return "█"*n + "·"*(width-n)

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

def _median(vals: List[int]) -> Optional[int]:
    if not vals: return None
    s = sorted(vals); m = len(s)//2
    return s[m] if len(s)%2 else (s[m-1]+s[m])//2

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
    """
    Precompute n mod r for each small prime r in the 'wheel'.
    For a candidate subtractor p, if p % r == n % r, then q=n-p ≡ 0 (mod r), so reject p.
    """
    return {r: (n % r) for r in wheel_primes if r != 2}

def q_pre_sieve(q: int, pre_primes: List[int]) -> bool:
    """
    Quick small-prime filter on q before BPSW.
    Accept if q is itself one of the pre_primes, or coprime to all of them.
    """
    for r in pre_primes:
        if r >= q: break
        if q % r == 0:
            return False
    return True

# ------------------------- Learned band ordering -------------------------

def load_cache(path: str) -> Dict[str, Dict[str, List[int]]]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(path: str, data: Dict[str, Dict[str, List[int]]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)

def init_band_stats(num_bands: int) -> Dict[str, List[int]]:
    # We track two integer arrays: trials[b], successes[b]
    return {
        "trials":   [0]*num_bands,
        "successes":[0]*num_bands
    }

def band_scores(trials: List[int], successes: List[int], smoothing: float = 1.0) -> List[float]:
    # Expected CTR with Laplace smoothing: (succ + s)/(trials + 2s)
    return [ (successes[b] + smoothing) / ( (trials[b] + 2.0*smoothing) ) for b in range(len(trials)) ]

def apply_decay(stats: Dict[str, List[int]], decay: float) -> None:
    if decay >= 0.9999: return
    for k in ("trials","successes"):
        for i, v in enumerate(stats[k]):
            stats[k][i] = int(round(v * decay))

# ------------------------- Core search procedure ------------------------

def search_subtractor_learned(
    n: int,
    primes: List[int],
    *,
    max_checks: int,
    wheel_primes: List[int],
    pre_primes: List[int],
    band_size: int,
    band_order: List[int],
    res_cache: Dict[int, List[int]],   # r -> [p % r for all p]
    rng: random.Random
) -> Tuple[bool, int, Optional[int]]:
    """
    (Kept for reference.) Band-ordered subtractor loop with wheel filter.
    Returns (found, checks_used, hit_band_index or None).
    """
    if n % 2 or n < 4:
        return False, 0, None

    n_mod = build_wheel_residues_for_n(n, wheel_primes)
    checks = 0
    total_items = len(primes)

    for b in band_order:
        start = b * band_size
        end   = min(start + band_size, total_items)
        if start >= end:
            continue
        for idx in range(start, end):
            p = primes[idx]
            # one-direction early break: stop once p > q  <=>  2*p > n
            if p * 2 > n:
                return False, checks, None
            # wheel prefilter
            wheel_ok = True
            for r, nr in n_mod.items():
                if res_cache[r][idx] == nr:
                    wheel_ok = False; break
            if not wheel_ok:
                continue
            q = n - p
            # presieve
            if not q_pre_sieve(q, pre_primes):
                checks += 1
                if checks >= max_checks:
                    return False, checks, None
                continue
            # BPSW
            checks += 1
            if is_probable_prime(q):
                return True, checks, b
            if checks >= max_checks:
                return False, checks, None

    return False, checks, None

# ------------------------------- CLI bits --------------------------------

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

def rationale() -> str:
    return (
        "\n*** Subtractor-only Goldbach (learned bands + self-tuning wheel + one-direction) ***\n\n"
        "Given even n, we try ascending subtractor primes p so that q=n-p is prime.\n"
        "Speedups:\n"
        "  • Self-tuning wheel residue prefilter (richer default) learns per-digit whether to use it.\n"
        "  • Pre-sieve q by many small primes before BPSW.\n"
        "  • One-direction early break at midpoint (stop when 2*p > n).\n"
        "  • Precomputed residues p % r once per wheel prime.\n"
        "  • Learned band ordering (per-digit).\n"
        "Any reported hit is validated by BPSW (Baillie–PSW primality test).\n"
    )

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="subs-gb-learn",
        description="Subtractor-only Goldbach with learned ordering + self-tuning wheel + one-direction search."
    )
    ap.add_argument("--digits", type=int, help="Single digit size (mutually exclusive with --sweep).")
    ap.add_argument("--count", type=int, default=10, help="How many n for single-run (default 10).")
    ap.add_argument("--sweep", type=_parse_sweep, default=None, help="Range 'start:end:step' for digit sweep.")
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

    # Optimizations (richer default wheel)
    ap.add_argument("--wheel-primes", type=str, default="3,5,7,11,13,17,19,23",
                    help="Comma-separated small primes for residue wheel (richer default).")
    ap.add_argument("--pre-sieve-limit", type=int, default=20000,
                    help="Pre-sieve q with primes up to this value before BPSW (default 20000).")

    # Learned band ordering
    ap.add_argument("--band-size", type=int, default=64,
                    help="Contiguous band size for learned ordering (default 64).")
    ap.add_argument("--cache-file", type=str, default="subs_learn_cache.json",
                    help="Path to JSON cache with per-digit band stats (default subs_learn_cache.json).")
    ap.add_argument("--decay", type=float, default=1.0,
                    help="Per-run multiplicative decay for stats (e.g., 0.99). Default 1.0 = no decay.")
    ap.add_argument("--smoothing", type=float, default=1.0,
                    help="Laplace smoothing for band scores (default 1.0).")

    args = ap.parse_args()
    if args.why:
        print(rationale()); return

    rng = random.Random(args.seed)

    # Prepare subtractor primes (ascending, deterministic, odd)
    subtractor_primes = [p for p in sieve_upto(args.subs_ceiling) if p % 2 == 1 and p != 1]

    # Precompute residues p % r once for each wheel prime
    wheel_list: List[int] = []
    for tok in args.wheel_primes.split(","):
        v = int(tok.strip())
        if v <= 2: continue
        wheel_list.append(v)
    res_cache: Dict[int, List[int]] = {r: [p % r for p in subtractor_primes] for r in wheel_list}

    # Prepare pre-sieve primes for q (exclude 2; keep odd primes)
    pre_primes = [p for p in sieve_upto(args.pre_sieve_limit) if p >= 3]

    # Band partition info
    band_size = max(1, args.band_size)
    total_items = len(subtractor_primes)
    num_bands = (total_items + band_size - 1) // band_size

    # Load / init cache
    cache = load_cache(args.cache_file)

    def get_stats_for_digits(D: int) -> Dict[str, List[int]]:
        key = str(D)
        if key not in cache:
            cache[key] = {}
        if "trials" not in cache[key] or "successes" not in cache[key]:
            cache[key]["trials"] = [0]*num_bands
            cache[key]["successes"] = [0]*num_bands
        # Resize band arrays if needed
        for k in ("trials","successes"):
            arr = cache[key][k]
            if len(arr) < num_bands:
                arr.extend([0]*(num_bands - len(arr)))
            elif len(arr) > num_bands:
                cache[key][k] = arr[:num_bands]
        # Self-tuning wheel stats (index 0=no-wheel, 1=wheel)
        if "wheel_trials" not in cache[key]:
            cache[key]["wheel_trials"] = [0, 0]
        if "wheel_ms_sum" not in cache[key]:
            cache[key]["wheel_ms_sum"] = [0.0, 0.0]
        if "wheel_checks_sum" not in cache[key]:
            cache[key]["wheel_checks_sum"] = [0, 0]
        return cache[key]

    def apply_decay_all(stats: Dict[str, List[int]]) -> None:
        # Decay band stats
        apply_decay({"trials": stats["trials"], "successes": stats["successes"]}, max(0.0, min(1.0, args.decay)))
        # Decay wheel stats smoothly
        d = max(0.0, min(1.0, args.decay))
        if d < 0.9999:
            for i in (0,1):
                stats["wheel_trials"][i] = int(round(stats["wheel_trials"][i] * d))
                stats["wheel_ms_sum"][i] *= d
                stats["wheel_checks_sum"][i] = int(round(stats["wheel_checks_sum"][i] * d))

    def band_scores_for(D: int) -> Tuple[List[int], Dict[str, List[int]]]:
        stats = get_stats_for_digits(D)
        apply_decay_all(stats)
        sc = band_scores(stats["trials"], stats["successes"], smoothing=max(0.0, args.smoothing))
        order = list(range(num_bands))
        order.sort(key=lambda b: (-sc[b], b))
        return order, stats

    def record_band_updates(stats: Dict[str, List[int]], visited_band_trials: Dict[int, int], hit_band: Optional[int]) -> None:
        for b, t in visited_band_trials.items():
            stats["trials"][b] += int(t)
        if hit_band is not None:
            stats["successes"][hit_band] += 1

    def choose_wheel(stats: Dict[str, List[int]], rng: random.Random, explore_eps: float = 0.05) -> int:
        """
        Return 0 for no-wheel, 1 for wheel.
        Chooses the option with lower observed avg ms per trial (fallback to checks),
        with small epsilon exploration to keep learning.
        """
        if rng.random() < explore_eps:
            return rng.choice([0,1])

        wt = stats["wheel_trials"]
        ms = stats["wheel_ms_sum"]
        ck = stats["wheel_checks_sum"]

        def metric(i: int) -> float:
            if wt[i] > 0:
                return ms[i] / wt[i]
            # If no timing yet, use checks as a proxy; larger is worse
            if wt[1-i] > 0:
                return (ck[i] / max(1, wt[i])) if wt[i] > 0 else float('inf')
            # No data at all: prefer wheel initially (usually beneficial)
            return 0.0 if i == 1 else 1e9

        m0 = metric(0)
        m1 = metric(1)
        return 0 if m0 < m1 else 1

    def run_for_digits(D: int, trials: int) -> Tuple[int,int,List[int],List[float]]:
        hits = 0
        checks_on_hits: List[int] = []
        ms_on_hits: List[float] = []

        band_order, stats = band_scores_for(D)

        for _ in range(trials):
            n = random_even_with_digits(D, rng)
            visited_band_trials: Dict[int, int] = {}

            # Decide per-n whether to use wheel (self-tuning)
            use_wheel = choose_wheel(stats, rng)  # 0=no-wheel, 1=wheel
            n_mod = build_wheel_residues_for_n(n, wheel_list) if use_wheel else {}

            checks_total = 0
            found = False
            hit_band = None
            t0 = time.perf_counter()

            if n % 2 == 0 and n >= 4:
                for b in band_order:
                    start = b * band_size
                    end   = min(start + band_size, total_items)
                    if start >= end:
                        continue
                    band_checks = 0
                    for idx in range(start, end):
                        p = subtractor_primes[idx]
                        # one-direction early break: stop once p > q  <=>  2*p > n
                        if p * 2 > n:
                            if band_checks:
                                visited_band_trials[b] = visited_band_trials.get(b, 0) + band_checks
                            checks_total += band_checks
                            record_band_updates(stats, visited_band_trials, None)
                            found = False
                            hit_band = None
                            break

                        # Wheel prefilter if enabled
                        if use_wheel:
                            wheel_ok = True
                            for r, nr in n_mod.items():
                                if res_cache[r][idx] == nr:
                                    wheel_ok = False; break
                            if not wheel_ok:
                                continue

                        q = n - p
                        # Pre-sieve
                        if not q_pre_sieve(q, pre_primes):
                            band_checks += 1
                            if checks_total + band_checks >= args.subs_max_checks:
                                visited_band_trials[b] = visited_band_trials.get(b, 0) + band_checks
                                checks_total += band_checks
                                record_band_updates(stats, visited_band_trials, None)
                                found = False; hit_band = None
                                break
                            continue
                        # BPSW
                        band_checks += 1
                        if is_probable_prime(q):
                            visited_band_trials[b] = visited_band_trials.get(b, 0) + band_checks
                            checks_total += band_checks
                            record_band_updates(stats, visited_band_trials, b)
                            found = True; hit_band = b
                            break
                        if checks_total + band_checks >= args.subs_max_checks:
                            visited_band_trials[b] = visited_band_trials.get(b, 0) + band_checks
                            checks_total += band_checks
                            record_band_updates(stats, visited_band_trials, None)
                            found = False; hit_band = None
                            break
                    # band finished or broke
                    if found or checks_total >= args.subs_max_checks or (start < end and subtractor_primes[start] * 2 > n):
                        break

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Update self-tuning stats (count every trial, hit or miss)
            mode = 1 if use_wheel else 0
            stats["wheel_trials"][mode] += 1
            stats["wheel_ms_sum"][mode] += elapsed_ms
            stats["wheel_checks_sum"][mode] += checks_total

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
            print("\n# Learned subtractor-only sweep")
            print(f"# subs_ceiling={args.subs_ceiling} | subs_max_checks={args.subs_max_checks}")
            print(f"# wheel={wheel_list} | pre_sieve_limit={args.pre_sieve_limit}")
            print(f"# band_size={band_size} | cache={args.cache_file} | decay={args.decay} | smoothing={args.smoothing}")
            print(f"# samples per digit={args.samples} | seed={args.seed}\n")

        for D in digits_space:
            hits, total, checks_on_hits, ms_on_hits = run_for_digits(D, args.samples)
            rate = hits/total if total else 0.0
            avg_checks = (sum(checks_on_hits)/len(checks_on_hits)) if checks_on_hits else None
            avg_ms = (sum(ms_on_hits)/len(ms_on_hits)) if ms_on_hits else None

            # Hide hit rate and CI if displayed value would be 1.000
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
                "band_size": band_size,
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
    print(f"# Subtractor-only (LEARNED + AUTO-WHEEL + ONE-DIR) | D={D} | count={args.count} | seed={args.seed}")
    print(f"# subs_ceiling={args.subs_ceiling} | subs_max_checks={args.subs_max_checks} "
          f"| wheel={wheel_list} | pre_sieve_limit={args.pre_sieve_limit} | band_size={band_size} | cache={args.cache_file}")
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


if __name__ == "__main__":
    main()
