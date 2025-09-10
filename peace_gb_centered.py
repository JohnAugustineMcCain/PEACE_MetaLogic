#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# peace_gb_centered.py
#
# Random-probe Goldbach sampler with multi-phase strategy:
#   Phase A: centered near n/2
#   Phase B: centered near n/3 (so q≈2n/3)
#   Phase C: subtractor primes p; test q=n-p
#
# + Sweep mode with Wilson CIs & Bayes-factor summary
# + BPSW primality (+ optional extra MR rounds)
# + Example decompositions and phase labels
#
# (c) 2025 John Augustine McCain — MIT-style permissive for this prototype

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, Iterable, Set
import argparse, random, json, csv, math

# === Global knob: extra MR rounds AFTER BPSW ===
_EXTRA_MR_ROUNDS: int = 0

# ---------------- Small primes -----------------
_SMALL_PRIMES: List[int] = [2,3,5,7,11,13,17,19,23,29,31,37]

def _decompose(n: int) -> Tuple[int, int]:
    d = n - 1; s = 0
    while d % 2 == 0:
        d //= 2; s += 1
    return d, s

def _mr_witness(a: int, d: int, n: int, s: int) -> bool:
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True  # a is a witness (composite)

# ------------- BPSW (+ optional MR) ------------
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

def is_probable_prime(n: int, rng: Optional[random.Random] = None) -> bool:
    if n < 2: return False
    for p in _SMALL_PRIMES:
        if n == p: return True
        if n % p == 0: return False
    if not _mr_strong_base2(n): return False
    if not _lucas_strong_prp(n): return False
    rounds = max(0, int(_EXTRA_MR_ROUNDS))
    if rounds > 0:
        if rng is None: rng = random.Random(0xC0FFEE)
        d, s = _decompose(n)
        for _ in range(rounds):
            a = 2 if n <= 4 else rng.randrange(2, n - 1)
            if _mr_witness(a, d, n, s): return False
    return True

# ---------------- Utilities --------------------
def random_even_with_digits(D: int, rng: random.Random) -> int:
    if D < 2: raise ValueError("digits must be >= 2")
    lo = 10**(D-1); hi = 10**D - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2: n += 1
    if n > hi: n -= 2
    if n < lo: n = lo + (1 if lo % 2 else 0)
    if n % 2: n += 1
    return n

def _ascii_bar(frac: float, width: int = 40) -> str:
    n = max(0, min(width, int(round(frac * width))))
    return "█"*n + "·"*(width-n)

def _wilson_ci(h: int, n: int, z: float = 1.96) -> Tuple[float,float]:
    if n == 0: return (0.0, 1.0)
    p = h/n; denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (max(0.0, center-half), min(1.0, center+half))

def _log10_binom_likelihood(h: int, n: int, p: float) -> float:
    if p <= 0 or p >= 1:
        return float("-inf") if (h>0 and h<n) else 0.0
    logC = math.lgamma(n+1) - math.lgamma(h+1) - math.lgamma(n-h+1)
    ll = logC + h*math.log(p) + (n-h)*math.log(1-p)
    return ll / math.log(10)

# ------------- Search primitives ---------------
def _search_around_target(
    n: int,
    target_p: int,
    *,
    rng: random.Random,
    trials: int,
    window: Optional[int]
) -> Tuple[bool, Optional[int], int, Optional[Tuple[int,int]]]:
    """Probe p near target_p; return (found, |p - n/2|, tries_used, (p,q))"""
    half = n // 2
    if window is None:
        # heuristic adaptive window
        window = max(50_000, min(5_000_000, (n.bit_length() ** 2) * 25))
    tries = 0
    for _ in range(trials):
        delta = rng.randint(-window, window)
        p = target_p + delta
        if p <= 2: tries += 1; continue
        if p % 2 == 0:
            p += 1 if delta <= 0 else -1
            if p <= 2: tries += 1; continue
        q = n - p
        if q <= 2: tries += 1; continue
        if is_probable_prime(p, rng) and is_probable_prime(q, rng):
            pair = (p, q) if p <= q else (q, p)
            return True, abs(p - half), tries + 1, pair
        tries += 1
    return False, None, tries, None

def _sieve_upto(limit: int) -> List[int]:
    if limit < 2: return []
    sieve = bytearray(b"\x01")*(limit+1)
    sieve[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for p in range(2, r+1):
        if sieve[p]:
            step = p
            start = p*p
            sieve[start:limit+1:step] = b"\x00" * (((limit - start)//step) + 1)
    return [i for i, b in enumerate(sieve) if b]

def _search_by_subtractors(
    n: int,
    *,
    rng: random.Random,
    ceiling: int,
    max_checks: int
) -> Tuple[bool, Optional[int], int, Optional[Tuple[int,int]]]:
    """Try random prime subtractors p ≤ ceiling; return first q=n-p prime."""
    half = n // 2
    primes = _sieve_upto(ceiling)
    rng.shuffle(primes)
    tries = 0
    for p in primes:
        if max_checks is not None and tries >= max_checks:
            break
        q = n - p
        if q > 2 and is_probable_prime(q, rng):
            pair = (p, q) if p <= q else (q, p)
            return True, abs(pair[0] - half), tries + 1, pair
        tries += 1
    return False, None, tries, None

# ------------- Multi-phase first-hit -----------
def first_hit_multiphase(
    n: int,
    *,
    rng: random.Random,
    # Phase A (center)
    trials_center: int,
    window_center: Optional[int],
    # Phase B (third)
    trials_third: int,
    window_third: Optional[int],
    # Phase C (subtractors)
    subs_ceiling: int,
    subs_max_checks: int
) -> Tuple[bool, Optional[int], int, Optional[Tuple[int,int]], str]:
    """
    Returns (found, delta_from_half, total_tries, pair, phase)
    phase ∈ {"center","third","subs","miss"}
    """
    if n % 2 != 0 or n < 4:
        return False, None, 0, None, "miss"

    # Quick p=2 shortcut
    q0 = n - 2
    if is_probable_prime(q0, rng):
        return True, abs(2 - (n//2)), 0, (2, q0), "center"

    total_tries = 0

    # Phase A: around n/2
    found, d, t, pair = _search_around_target(
        n, n//2, rng=rng, trials=trials_center, window=window_center
    )
    total_tries += t
    if found: return True, d, total_tries, pair, "center"

    # Phase B: around n/3
    found, d, t, pair = _search_around_target(
        n, n//3, rng=rng, trials=trials_third, window=window_third
    )
    total_tries += t
    if found: return True, d, total_tries, pair, "third"

    # Phase C: subtractor primes
    found, d, t, pair = _search_by_subtractors(
        n, rng=rng, ceiling=subs_ceiling, max_checks=subs_max_checks
    )
    total_tries += t
    if found: return True, d, total_tries, pair, "subs"

    return False, None, total_tries, None, "miss"

# ---------------- Rationale --------------------
def rationale() -> str:
    return (
        "\n*** Goldbach — Multi-phase Probing & Rising Certainty ***\n\n"
        "Phase A samples near n/2 where valid pairs concentrate (density ~1/(log n)^2). "
        "If that misses within a fixed budget, Phase B samples near n/3 (so q≈2n/3), adding a distinct ridge "
        "in the additive landscape. If both miss, Phase C scans randomized prime subtractors p and tests q=n−p.\n\n"
        "With fixed budgets, observing hit-rates that rise (or stay high) as digits grow indicates miss probability "
        "shrinks roughly like exp(−const/(log n)^2), so empirical certainty in Goldbach increases with scale "
        "(up to ranges where computation ceases to be informative). Primality is BPSW (+ optional MR)."
    )

# ---------------- CLI parsing ------------------
def _positive_int(name: str, s: str) -> int:
    try:
        v = int(s)
        if v <= 0: raise ValueError
        return v
    except Exception:
        raise argparse.ArgumentTypeError(f"{name} must be a positive integer (got {s!r})")

def _parse_digits_list(s: str) -> List[int]:
    try:
        out = []
        for part in s.split(","):
            part = part.strip()
            if part:
                v = int(part)
                if v <= 0: raise ValueError
                out.append(v)
        if not out: raise ValueError
        return out
    except Exception:
        raise argparse.ArgumentTypeError(f"--digits-list must be like '12,16,20' (got {s!r})")

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

# ---------------------- Main -------------------
def main() -> None:
    global _EXTRA_MR_ROUNDS

    ap = argparse.ArgumentParser(
        prog="peace-gb-centered",
        description="Goldbach sampler with multi-phase strategy: center → third → subtractors; sweep with CIs/Bayes."
    )
    # Single-run basics
    ap.add_argument("--digits", type=lambda s: _positive_int("--digits", s),
                    help="Digits for single-run (mutually exclusive with sweep).")
    ap.add_argument("--count", type=lambda s: _positive_int("--count", s), default=10,
                    help="How many random n in single-run (default 10).")
    ap.add_argument("--pairs-only", action="store_true", help="Only print 'p + q = n' lines (single-run).")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed (default 2025).")
    ap.add_argument("--why", action="store_true", help="Print the one-page rationale and exit.")

    # Phase budgets (defaults chosen to mirror your earlier runs)
    ap.add_argument("--per-n-trials", type=lambda s: _positive_int("--per-n-trials", s), default=60_000,
                    help="TOTAL trials budget if you only care about Phase A (legacy).")
    ap.add_argument("--window", type=lambda s: _positive_int("--window", s), default=None,
                    help="Window around target p for Phase A (default adaptive).")

    ap.add_argument("--trials-center", type=int, default=None,
                    help="Trials for Phase A (center). Default: --per-n-trials.")
    ap.add_argument("--window-center", type=int, default=None,
                    help="Window for Phase A. Default: --window (adaptive if None).")

    ap.add_argument("--trials-third", type=int, default=20_000,
                    help="Trials for Phase B (third). Default 20000.")
    ap.add_argument("--window-third", type=int, default=None,
                    help="Window for Phase B. Default adaptive.")

    ap.add_argument("--subs-ceiling", type=int, default=200_000,
                    help="Prime sieve ceiling for Phase C (default 200k).")
    ap.add_argument("--subs-max-checks", type=int, default=20_000,
                    help="Max subtractor primes to try in Phase C (default 20000).")

    # Primality hardness knob
    ap.add_argument("--extra-mr-rounds", type=int, default=0,
                    help="Extra random MR rounds after BPSW (default 0).")

    # Sweep
    ap.add_argument("--digits-list", type=_parse_digits_list, default=None,
                    help="Comma-separated digit sizes to sweep, e.g. 18,22,26.")
    ap.add_argument("--sweep", type=_parse_sweep, default=None,
                    help="Range sweep 'start:end:step', e.g. 18:60:4.")
    ap.add_argument("--samples", type=lambda s: _positive_int("--samples", s), default=100,
                    help="How many n per digit in sweep (default 100).")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-n prints in sweep.")
    ap.add_argument("--csv", type=str, default=None, help="Write sweep summary CSV.")
    ap.add_argument("--jsonl", type=str, default=None, help="Write per-row JSONL.")
    ap.add_argument("--examples", type=int, default=2, help="Random example decompositions to show at end (default 2).")

    # Stats outputs
    ap.add_argument("--ci", action="store_true", help="Print Wilson 95% CI for hit-rate.")
    ap.add_argument("--bayes-eps", type=float, default=0.1,
                    help="Null miss-rate epsilon (p0=1-eps). Default 0.1 → p0=0.9.")
    ap.add_argument("--bayes-summary", action="store_true",
                    help="Print cumulative log10 Bayes factor vs fixed p0.")

    args = ap.parse_args()
    if args.why:
        print(rationale()); return

    _EXTRA_MR_ROUNDS = max(0, int(args.extra_mr_rounds))
    rng = random.Random(args.seed)

    # Fill phase defaults
    trials_center = args.trials_center if args.trials_center is not None else args.per_n_trials
    window_center = args.window_center if args.window_center is not None else args.window
    trials_third  = max(0, int(args.trials_third))
    window_third  = args.window_third

    # ---------- SWEEP MODE ----------
    in_sweep = (args.digits_list is not None) or (args.sweep is not None)
    if in_sweep or (args.digits is None):
        digits_space: List[int] = []
        if args.digits_list: digits_space.extend(args.digits_list)
        if args.sweep: digits_space.extend(args.sweep)
        if not digits_space:
            # If neither provided, require --digits for single-run
            if args.digits is None:
                ap.error("Provide --digits OR --digits-list/--sweep. Use --why to read the rationale.")
        else:
            digits_space = sorted(set(digits_space))

    if in_sweep:
        if not args.quiet:
            print("\n# Multi-phase Goldbach sweep (center → third → subtractors)")
            print(f"# center_trials={trials_center} | third_trials={trials_third} | subs_ceiling={args.subs_ceiling} | subs_max_checks={args.subs_max_checks}")
            print(f"# samples per digit={args.samples} | seed={args.seed} | extraMR={_EXTRA_MR_ROUNDS}\n")

        p0 = max(0.0, min(1.0, 1.0 - args.bayes_eps))
        sum_log10_BF = 0.0
        total_h = 0; total_n = 0
        summary_rows: List[Dict[str, Any]] = []
        jsonl_rows: List[Dict[str, Any]] = []
        example_pool: List[Dict[str, Any]] = []

        for D in digits_space:
            hits = 0
            tries_on_hits: List[int] = []
            deltas_on_hits: List[int] = []
            phase_counts = {"center":0, "third":0, "subs":0, "miss":0}
            local_examples: List[Dict[str, Any]] = []

            for _ in range(args.samples):
                n = random_even_with_digits(D, rng)
                found, delta, tries_used, pair, phase = first_hit_multiphase(
                    n,
                    rng=rng,
                    trials_center=trials_center, window_center=window_center,
                    trials_third=trials_third,   window_third=window_third,
                    subs_ceiling=args.subs_ceiling, subs_max_checks=args.subs_max_checks
                )
                phase_counts[phase] += 1
                if found:
                    hits += 1
                    if delta is not None: deltas_on_hits.append(delta)
                    tries_on_hits.append(tries_used)
                    if pair and len(local_examples) < 3:
                        p, q = pair
                        local_examples.append({
                            "digits": D, "n": str(n), "p": str(p), "q": str(q),
                            "delta": delta, "tries": tries_used, "phase": phase
                        })

                if args.jsonl:
                    jsonl_rows.append({
                        "digits": D, "n": str(n), "found": found,
                        "delta_from_half": delta, "tries_used": tries_used,
                        "phase": phase,
                        "center_trials": trials_center, "third_trials": trials_third,
                        "subs_ceiling": args.subs_ceiling, "subs_max_checks": args.subs_max_checks,
                        "seed": args.seed, "extra_mr_rounds": _EXTRA_MR_ROUNDS
                    })

            total = args.samples; total_h += hits; total_n += total
            hit_rate = hits/total if total else 0.0
            avg_tries = (sum(tries_on_hits)/len(tries_on_hits)) if tries_on_hits else None
            median_delta = None
            if deltas_on_hits:
                srt = sorted(deltas_on_hits); m = len(srt)//2
                median_delta = srt[m] if len(srt)%2==1 else (srt[m-1]+srt[m])//2

            lo95, hi95 = _wilson_ci(hits, total) if args.ci else (None, None)

            if args.bayes_summary and total > 0:
                p_hat = min(1 - 1e-12, max(1e-12, hit_rate))
                ll1 = _log10_binom_likelihood(hits, total, p_hat)
                ll0 = _log10_binom_likelihood(hits, total, p0)
                sum_log10_BF += (ll1 - ll0)

            bar = _ascii_bar(hit_rate)
            phases_txt = f"  [center:{phase_counts['center']} third:{phase_counts['third']} subs:{phase_counts['subs']} miss:{phase_counts['miss']}]"
            ci_txt = f"  CI95=[{lo95:.3f},{hi95:.3f}]" if args.ci else ""
            print(f"{D:>3}d  hit_rate={hit_rate:6.3f}{ci_txt}  avg_trials_to_hit={('%.1f'%avg_tries) if avg_tries is not None else '—':>6}  "
                  f"medianΔ={median_delta if median_delta is not None else '—':>8}  {bar}{phases_txt}")

            row = {
                "digits": D, "samples": total, "hits": hits, "misses": total - hits,
                "hit_rate": round(hit_rate, 6),
                "avg_trials_to_hit": round(avg_tries, 3) if avg_tries is not None else None,
                "median_delta_from_half": median_delta,
                "phase_center": phase_counts["center"],
                "phase_third": phase_counts["third"],
                "phase_subs": phase_counts["subs"],
                "phase_miss": phase_counts["miss"],
                "center_trials": trials_center, "third_trials": trials_third,
                "subs_ceiling": args.subs_ceiling, "subs_max_checks": args.subs_max_checks,
                "window_center": window_center, "window_third": window_third,
                "seed": args.seed, "extra_mr_rounds": _EXTRA_MR_ROUNDS
            }
            if args.ci:
                row["hit_rate_ci95_low"] = round(lo95, 6)
                row["hit_rate_ci95_high"] = round(hi95, 6)
            summary_rows.append(row)
            example_pool.extend(local_examples)

        if args.csv and summary_rows:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader(); writer.writerows(summary_rows)
            print(f"[saved] CSV -> {args.csv}")

        if args.jsonl and jsonl_rows:
            with open(args.jsonl, "w") as f:
                for r in jsonl_rows: f.write(json.dumps(r) + "\n")
            print(f"[saved] JSONL -> {args.jsonl}")

        if args.bayes_summary:
            print(f"\n# Bayes summary vs fixed p0={1-args.bayes_eps:.3f}")
            print(f"log10 Bayes factor (cumulative): {sum_log10_BF:.3f}")
            print("Interpretation:", "favoring Goldbach-friendly model" if sum_log10_BF>0 else ("favoring fixed-p0 null" if sum_log10_BF<0 else "indifferent"))

        print("\n# Meta-summary")
        overall_rate = total_h/total_n if total_n else 0.0
        print(f"overall hit_rate={overall_rate:.3f} across {total_n} trials | seed={args.seed} | extraMR={_EXTRA_MR_ROUNDS}")

        if args.examples and example_pool:
            k = min(args.examples, len(example_pool))
            rnd = random.Random(args.seed ^ 0xA11CE)
            picks = rnd.sample(example_pool, k)
            print("\n# Example decompositions (random selection from this sweep)")
            for ex in picks:
                print(f"{ex['digits']:>3}d phase={ex['phase']:<6} Δ={ex['delta']:<7} tries={ex['tries']:<5}  {ex['p']} + {ex['q']} = {ex['n']}")
        return

    # ---------- SINGLE-RUN ----------
    D = args.digits
    if D is None:
        ap.error("Provide --digits for single-run OR use sweep flags. Use --why for the rationale.")

    if args.pairs_only:
        for i in range(args.count):
            n = random_even_with_digits(D, rng)
            found, _, _, pair, _ = first_hit_multiphase(
                n,
                rng=rng,
                trials_center=trials_center, window_center=window_center,
                trials_third=trials_third,   window_third=window_third,
                subs_ceiling=args.subs_ceiling, subs_max_checks=args.subs_max_checks
            )
            if found and pair:
                p, q = pair; print(f"{p} + {q} = {n}")
            else:
                print(f"# miss for n={n}")
    else:
        for i in range(args.count):
            n = random_even_with_digits(D, rng)
            found, d, tries, pair, phase = first_hit_multiphase(
                n,
                rng=rng,
                trials_center=trials_center, window_center=window_center,
                trials_third=trials_third,   window_third=window_third,
                subs_ceiling=args.subs_ceiling, subs_max_checks=args.subs_max_checks
            )
            if found and pair:
                p, q = pair
                print(f"[{i+1:>3}/{args.count}] phase={phase:<6} tries={tries:<6} Δ={d if d is not None else '—':>8}  ->  {p} + {q} = {n}")
            else:
                print(f"[{i+1:>3}/{args.count}] phase=miss   tries={tries:<6} Δ=       —  ->  # miss for n={n}")

if __name__ == "__main__":
    main()
