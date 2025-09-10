#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# subs_gb.py — Deterministic subtractor-only Goldbach engine
#
# What it does
#   • For each even n, iterate primes p = 2, 3, 5, 7, … up to --subs-ceiling in ascending order (NO SHUFFLE).
#   • For each p, test q = n - p for primality; the first prime q gives a Goldbach decomposition.
#   • Stops after --subs-max-checks primes (or when p exceeds ceiling).
#
# Notes
#   • Primality is Baillie–PSW + optional extra Miller–Rabin rounds (belt-and-suspenders).
#   • Supports single-run and sweep (across digit sizes) with hit-rate, optional Wilson CI, CSV/JSONL logs.
#   • Optional --log-misses to write ONLY misses (one per line: digits,n).
#
# (c) 2025 — permissive prototype

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import argparse, random, math, csv, json

# ===== Small helpers =====

_SMALL_PRIMES: List[int] = [2,3,5,7,11,13,17,19,23,29,31,37]
_EXTRA_MR_ROUNDS: int = 0  # configurable via CLI

def _positive_int(name: str, s: str) -> int:
    try:
        v = int(s)
        if v <= 0: raise ValueError
        return v
    except Exception:
        raise argparse.ArgumentTypeError(f"{name} must be a positive integer (got {s!r})")

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

def _wilson_ci(h: int, n: int, z: float = 1.96) -> Tuple[float,float]:
    if n == 0: return (0.0, 1.0)
    p = h/n; denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (max(0.0, center-half), min(1.0, center+half))

def _ascii_bar(frac: float, width: int = 40) -> str:
    n = max(0, min(width, int(round(frac * width))))
    return "█"*n + "·"*(width-n)

# ===== Random even n with D digits =====

def random_even_with_digits(D: int, rng: random.Random) -> int:
    if D < 2: raise ValueError("digits must be >= 2")
    lo = 10**(D-1); hi = 10**D - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2: n += 1
    if n > hi: n -= 2
    if n < lo: n = lo + (1 if lo % 2 else 0)
    if n % 2: n += 1
    return n

# ===== BPSW + optional MR =====

def _decompose(n: int) -> Tuple[int, int]:
    d = n - 1; s = 0
    while d % 2 == 0: d //= 2; s += 1
    return d, s

def _mr_witness(a: int, d: int, n: int, s: int) -> bool:
    x = pow(a, d, n)
    if x == 1 or x == n - 1: return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1: return False
    return True

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

# ===== Sieve and subtractor search (DETERMINISTIC ORDER) =====

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

def subtractor_first_hit(
    n: int,
    *,
    primes: List[int],
    max_checks: int,
    rng: Optional[random.Random] = None
) -> Tuple[bool, int]:
    """
    Deterministic (unshuffled) subtractor pass.
    Returns (found, checks_used). Does NOT return the pair (p,q); this engine logs only misses by default.
    """
    checks = 0
    for p in primes:
        if max_checks is not None and checks >= max_checks:
            break
        q = n - p
        # for even n≥4, p=2 gives q even; we still test uniformly (deterministic policy)
        if q > 2 and is_probable_prime(q, rng):
            return True, checks + 1
        checks += 1
    return False, checks

# ===== CLI =====

def rationale() -> str:
    return (
        "\n*** Subtractor-only Goldbach engine (deterministic) ***\n\n"
        "For each even n, we iterate primes p in ascending order (no shuffle) and test q=n−p for primality.\n"
        "This models a fully sequential subtractor strategy and lets you study hit-rates and miss structure\n"
        "without the randomness of reordering the subtractors.\n"
        "Primality uses Baillie–PSW with optional extra Miller–Rabin rounds.\n"
    )

def main() -> None:
    global _EXTRA_MR_ROUNDS

    ap = argparse.ArgumentParser(
        prog="subs-gb",
        description="Deterministic subtractor-only Goldbach engine (no shuffle)."
    )

    # Modes
    ap.add_argument("--digits", type=lambda s: _positive_int("--digits", s),
                    help="Digits for single-run (mutually exclusive with sweep).")
    ap.add_argument("--count", type=lambda s: _positive_int("--count", s), default=10,
                    help="How many random n to test in single-run (default 10).")
    ap.add_argument("--sweep", type=_parse_sweep, default=None,
                    help="Range sweep 'start:end:step', e.g. 18:60:4.")
    ap.add_argument("--samples", type=lambda s: _positive_int("--samples", s), default=100,
                    help="How many n per digit in sweep (default 100).")

    # Subtractor params
    ap.add_argument("--subs-ceiling", type=int, default=200000,
                    help="Prime sieve ceiling for subtractors (default 200k).")
    ap.add_argument("--subs-max-checks", type=int, default=1000,
                    help="Max subtractor primes to try (default 1000).")

    # Output controls
    ap.add_argument("--ci", action="store_true", help="Print Wilson 95% CI for hit-rate (sweep mode).")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-n prints; show summary lines only.")
    ap.add_argument("--csv", type=str, default=None, help="Write sweep summary CSV.")
    ap.add_argument("--jsonl", type=str, default=None, help="Write per-row JSONL.")
    ap.add_argument("--log-misses", type=str, default=None, help="Append ONLY misses as 'digits,n' lines to this file.")
    ap.add_argument("--why", action="store_true", help="Print rationale and exit.")
    ap.add_argument("--seed", type=int, default=2025, help="RNG seed for sampling n (default 2025).")
    ap.add_argument("--extra-mr-rounds", type=int, default=0, help="Extra random MR rounds after BPSW (default 0).")

    args = ap.parse_args()
    if args.why:
        print(rationale()); return

    _EXTRA_MR_ROUNDS = max(0, int(args.extra_mr_rounds))
    rng = random.Random(args.seed)

    # Build deterministic prime list once
    primes = sieve_upto(args.subs_ceiling)

    # Single-run
    if args.sweep is None and args.digits is not None:
        D = args.digits
        hits = 0
        checks_on_hits: List[int] = []
        for i in range(args.count):
            n = random_even_with_digits(D, rng)
            found, checks_used = subtractor_first_hit(
                n, primes=primes, max_checks=args.subs_max_checks, rng=rng
            )
            if not args.quiet:
                if found:
                    print(f"[{i+1:>3}/{args.count}] hit  checks={checks_used}  n={n}")
                else:
                    print(f"[{i+1:>3}/{args.count}] MISS checks={checks_used}  n={n}")
            if found:
                hits += 1
                checks_on_hits.append(checks_used)
            else:
                if args.log_misses:
                    with open(args.log_misses, "a") as f:
                        f.write(f"{D},{n}\n")

        total = args.count
        rate = hits/total if total else 0.0
        avg_checks = (sum(checks_on_hits)/len(checks_on_hits)) if checks_on_hits else None
        print(f"\n{D}d  hit_rate={rate:.3f}  avg_checks_to_hit={('%.1f' % avg_checks) if avg_checks is not None else '—'}")
        return

    # Sweep
    if args.sweep is None and args.digits is None:
        ap.error("Provide --digits for single-run OR --sweep for sweep mode.")

    digits_space = args.sweep if args.sweep is not None else [args.digits]
    if not args.quiet:
        print("\n# Deterministic subtractor-only sweep")
        print(f"# subs_ceiling={args.subs_ceiling} | subs_max_checks={args.subs_max_checks}")
        print(f"# samples per digit={args.samples} | seed={args.seed} | extraMR={_EXTRA_MR_ROUNDS}\n")

    summary_rows: List[Dict[str, Any]] = []
    jsonl_rows: List[Dict[str, Any]] = []

    for D in digits_space:
        hits = 0
        checks_on_hits: List[int] = []
        for _ in range(args.samples):
            n = random_even_with_digits(D, rng)
            found, checks_used = subtractor_first_hit(
                n, primes=primes, max_checks=args.subs_max_checks, rng=rng
            )
            if found:
                hits += 1
                checks_on_hits.append(checks_used)
            else:
                if args.log_misses:
                    with open(args.log_misses, "a") as f:
                        f.write(f"{D},{n}\n")
            if args.jsonl:
                jsonl_rows.append({
                    "digits": D, "n": str(n), "found": found,
                    "checks_used": checks_used,
                    "subs_ceiling": args.subs_ceiling,
                    "subs_max_checks": args.subs_max_checks,
                    "seed": args.seed, "extra_mr_rounds": _EXTRA_MR_ROUNDS
                })

        total = args.samples
        rate = hits/total if total else 0.0
        avg_checks = (sum(checks_on_hits)/len(checks_on_hits)) if checks_on_hits else None
        bar = _ascii_bar(rate)
        if args.ci:
            lo, hi = _wilson_ci(hits, total)
            ci_txt = f"  CI95=[{lo:.3f},{hi:.3f}]"
        else:
            ci_txt = ""

        print(f"{D:>3}d  hit_rate={rate:6.3f}{ci_txt}  avg_checks_to_hit={('%.1f'%avg_checks) if avg_checks is not None else '—':>6}  {bar}")

        row = {
            "digits": D, "samples": total, "hits": hits, "misses": total - hits,
            "hit_rate": round(rate, 6),
            "avg_checks_to_hit": round(avg_checks, 3) if avg_checks is not None else None,
            "subs_ceiling": args.subs_ceiling, "subs_max_checks": args.subs_max_checks,
            "seed": args.seed, "extra_mr_rounds": _EXTRA_MR_ROUNDS
        }
        if args.ci:
            row["hit_rate_ci95_low"] = round(lo, 6)
            row["hit_rate_ci95_high"] = round(hi, 6)
        summary_rows.append(row)

    if args.csv and summary_rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader(); writer.writerows(summary_rows)
        print(f"[saved] CSV -> {args.csv}")

    if args.jsonl and jsonl_rows:
        with open(args.jsonl, "w") as f:
            for r in jsonl_rows: f.write(json.dumps(r) + "\n")
        print(f"[saved] JSONL -> {args.jsonl}")

if __name__ == "__main__":
    main()
