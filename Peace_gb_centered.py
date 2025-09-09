#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# peace_gb_centered.py
#
# Random-probe Goldbach sampler centered at n/2
# + Sweep mode to measure how hit-rate (and thus heuristic certainty) scales with digits.
#
# (c) 2025 John Augustine McCain — MIT-style permissive for this prototype

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Iterable, Set
import argparse, random, json, csv

# =========================
#  Lightweight MR primality
# =========================

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

def _mr_rounds_for_digits(d: int) -> int:
    # A few rounds suffice for randomized evidence (we're sampling, not certifying)
    if d <= 18: return 8
    if d <= 24: return 6
    if d <= 34: return 5
    return 4

def is_probable_prime(n: int, rng: random.Random) -> bool:
    if n < 2: return False
    for p in _SMALL_PRIMES:
        if n == p: return True
        if n % p == 0: return False
    d, s = _decompose(n)
    rounds = _mr_rounds_for_digits(len(str(n)))
    max_base = n - 2
    for _ in range(rounds):
        a = rng.randrange(2, max_base + 1) if max_base >= 2 else 2
        if _mr_witness(a, d, n, s):
            return False
    return True

# =========================
#  Random even generator
# =========================

def random_even_with_digits(D: int, rng: random.Random) -> int:
    if D < 2:
        raise ValueError("digits must be >= 2")
    lo = 10 ** (D - 1); hi = 10 ** D - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2: n += 1
    if n > hi: n -= 2
    if n < lo:
        n = lo + (1 if lo % 2 == 1 else 0)
    if n % 2: n += 1
    return n

# =========================
#  Centered search around n/2
# =========================

def centered_pairs_iter(
    n: int,
    *,
    rng: random.Random,
    per_n_trials: int,
    window: Optional[int] = None,
) -> Iterable[Tuple[int, int]]:
    """
    Yield DISTINCT unordered prime pairs (p,q) with p+q=n, prioritizing p≈n/2.

    Strategy: random ring-expansion around n/2.
      - draw delta in [-window, window], set p = n//2 + delta (forced odd)
      - test p and q=n-p with MR
      - avoid duplicates; stop after per_n_trials draws
    """
    if n % 2 != 0 or n < 4:
        return
    half = n // 2
    if window is None:
        # heuristic window scales with bit-length (small but effective)
        window = max(50_000, min(5_000_000, (n.bit_length() ** 2) * 25))

    seen: Set[Tuple[int, int]] = set()

    # quick p=2 shortcut (q big)
    q0 = n - 2
    if is_probable_prime(q0, rng):
        pair = (2, q0)
        pair = (pair[0], pair[1]) if pair[0] <= pair[1] else (pair[1], pair[0])
        if pair not in seen:
            seen.add(pair)
            yield pair

    for _ in range(per_n_trials):
        delta = rng.randint(-window, window)
        p = half + delta
        if p <= 2:
            continue
        # force odd
        if p % 2 == 0:
            p += 1 if delta <= 0 else -1
            if p <= 2:
                continue
        q = n - p
        if q <= 2:
            continue
        # test both
        if is_probable_prime(p, rng) and is_probable_prime(q, rng):
            pair = (p, q) if p <= q else (q, p)
            if pair not in seen:
                seen.add(pair)
                yield pair

# =========================
#  First-hit probe (for sweep)
# =========================

def first_hit_centered(
    n: int,
    *,
    rng: random.Random,
    per_n_trials: int,
    window: Optional[int] = None,
) -> Tuple[bool, Optional[int], int]:
    """
    Returns (found, delta_from_half, tries_used).
    'tries_used' counts actual candidate draws (p choices).
    """
    if n % 2 != 0 or n < 4:
        return (False, None, 0)

    half = n // 2
    if window is None:
        window = max(50_000, min(5_000_000, (n.bit_length() ** 2) * 25))

    # quick p=2 shortcut
    q0 = n - 2
    tries_used = 0
    if is_probable_prime(q0, rng):
        return (True, abs(2 - half), tries_used)

    for _ in range(per_n_trials):
        delta = rng.randint(-window, window)
        p = half + delta
        if p <= 2:
            tries_used += 1
            continue
        if p % 2 == 0:
            p += 1 if delta <= 0 else -1
            if p <= 2:
                tries_used += 1
                continue
        q = n - p
        if q <= 2:
            tries_used += 1
            continue
        if is_probable_prime(p, rng) and is_probable_prime(q, rng):
            return (True, abs(p - half), tries_used + 1)
        tries_used += 1

    return (False, None, tries_used)

# =========================
#  Friendly rationale (updated)
# =========================

def rationale() -> str:
    return (
        "\n*** Goldbach at Huge Scales — Centered Random Probing and Rising Certainty ***\n\n"
        "Goldbach predicts every even n>2 is p+q with p,q prime. For large n, the expected number of such pairs grows like ~ n/(log n)^2.\n"
        "Prime density is ~1/log n, and we need TWO primes (p and q=n−p), so a random split lands on 'both prime' with chance ~1/(log n)^2.\n"
        "Combinatorially, valid pairs concentrate near the balanced split p≈q≈n/2. Thus, sampling near n/2 first witnesses decompositions fastest.\n\n"
        "Why this supports increasing certainty (if the sweep behaves as expected):\n"
        "• Fix a per-n budget (same trials per n). As digits grow, the heuristic hit-rate typically INCREASES.\n"
        "  Intuition: the expected count of Goldbach pairs grows like n/(log n)^2, so there are simply MORE targets clustered around n/2.\n"
        "• When a fixed-effort sampler succeeds more often at larger n, its empirical success probability rises with scale.\n"
        "• Therefore, conditional on the observed trend, we become MORE CERTAIN (in the evidential sense) that Goldbach holds for typical large n,\n"
        "  up to the point where numbers become so astronomically large that our finite probe budget ceases to be informative.\n\n"
        "This is not a formal proof. It is experimental mathematics: a meta-level argument where rising hit-rates under fixed budget provide\n"
        "increasing empirical support for the conjecture across growing scales (before computational limits dominate).\n"
    )

# =========================
#  CLI helpers
# =========================

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
                if v <= 0:
                    raise ValueError
                out.append(v)
        if not out:
            raise ValueError
        return out
    except Exception:
        raise argparse.ArgumentTypeError(f"--digits-list must be like '12,16,20' (got {s!r})")

def _parse_sweep(s: str) -> List[int]:
    try:
        a, b, c = [int(x) for x in s.split(":")]
        if c == 0:
            raise ValueError
        rng = range(a, b + (1 if c > 0 else -1), c)
        out = [v for v in rng if v > 0]
        if not out:
            raise ValueError
        return out
    except Exception:
        raise argparse.ArgumentTypeError(f"--sweep must be 'start:end:step' (got {s!r})")

def _ascii_bar(frac: float, width: int = 40) -> str:
    n = max(0, min(width, int(round(frac * width))))
    return "█" * n + "·" * (width - n)

# =========================
#  Main (single-run or sweep)
# =========================

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="peace-gb-centered",
        description="Random Goldbach sampler centered at n/2. Single-run or sweep to measure accuracy vs digits."
    )
    # single-run (original behavior)
    ap.add_argument("--digits", type=lambda s: _positive_int("--digits", s),
                    help="Digits for single-run mode (mutually exclusive with sweep).")
    ap.add_argument("--count", type=lambda s: _positive_int("--count", s), default=10,
                    help="How many random n to sample in single-run mode (default 10).")
    ap.add_argument("--per-n-trials", type=lambda s: _positive_int("--per-n-trials", s), default=60_000,
                    help="Candidate draws around n/2 per n (default 60000).")
    ap.add_argument("--per-n-count", type=lambda s: _positive_int("--per-n-count", s), default=1,
                    help="Max pairs to print per n in single-run mode (default 1).")
    ap.add_argument("--window", type=lambda s: _positive_int("--window", s), default=None,
                    help="Search window around n/2 (default: adaptive).")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed (default 2025).")
    ap.add_argument("--pairs-only", action="store_true",
                    help="Only print 'p + q = n' lines (quiet mode) for single-run.")
    ap.add_argument("--why", action="store_true",
                    help="Explain why rising hit-rate with fixed budget increases empirical certainty with scale.")

    # sweep mode
    ap.add_argument("--digits-list", type=_parse_digits_list, default=None,
                    help="Comma-separated list of digit sizes to sweep, e.g. 12,16,20,24.")
    ap.add_argument("--sweep", type=_parse_sweep, default=None,
                    help="Range sweep 'start:end:step' over digits, e.g. 12:40:4.")
    ap.add_argument("--samples", type=lambda s: _positive_int("--samples", s), default=100,
                    help="How many n to sample per digit size in sweep (default 100).")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-n output (sweep prints only summary rows).")
    ap.add_argument("--csv", type=str, default=None,
                    help="When set (e.g., results.csv), write sweep summary to CSV.")
    ap.add_argument("--jsonl", type=str, default=None,
                    help="When set (e.g., rows.jsonl), write per-row sweep results to JSONL.")

    args = ap.parse_args()

    if args.why:
        print(rationale())
        return

    # Determine mode
    in_sweep = (args.digits_list is not None) or (args.sweep is not None)
    if (args.digits is None) and not in_sweep:
        ap.error("Provide --digits for single-run OR --digits-list / --sweep for sweep mode. Use --why for the explanation page.")

    rng = random.Random(args.seed)

    # ---------- SWEEP MODE ----------
    if in_sweep:
        digits_space: List[int] = []
        if args.digits_list:
            digits_space.extend(args.digits_list)
        if args.sweep:
            digits_space.extend(args.sweep)
        digits_space = sorted(set(digits_space))

        summary_rows: List[Dict[str, Any]] = []
        jsonl_rows: List[Dict[str, Any]] = []

        if not args.quiet:
            print("\n# Measuring centered-heuristic accuracy vs. digits")
            print("# (hit = find at least one pair within per-n budget)")
            print(f"# per_n_trials={args.per_n_trials} | samples per digit={args.samples} | seed={args.seed}\n")

        for D in digits_space:
            hits = 0
            tries_on_hits: List[int] = []
            deltas_on_hits: List[int] = []
            for _ in range(args.samples):
                n = random_even_with_digits(D, rng)
                found, delta, tries_used = first_hit_centered(
                    n, rng=rng, per_n_trials=args.per_n_trials, window=args.window
                )
                if found:
                    hits += 1
                    tries_on_hits.append(tries_used)
                    deltas_on_hits.append(delta if delta is not None else 0)

                if args.jsonl:
                    jsonl_rows.append({
                        "digits": D, "n": str(n), "found": found,
                        "delta_from_half": delta, "tries_used": tries_used,
                        "per_n_trials": args.per_n_trials, "seed": args.seed
                    })

            total = args.samples
            hit_rate = hits / total if total else 0.0
            avg_tries = (sum(tries_on_hits) / len(tries_on_hits)) if tries_on_hits else None
            median_delta = None
            if deltas_on_hits:
                srt = sorted(deltas_on_hits)
                m = len(srt) // 2
                median_delta = (srt[m] if len(srt) % 2 == 1 else (srt[m - 1] + srt[m]) // 2)

            bar = _ascii_bar(hit_rate)
            print(f"{D:>3}d  hit_rate={hit_rate:6.3f}  avg_trials_to_hit={('%.1f'%avg_tries) if avg_tries is not None else '—':>6}  "
                  f"medianΔ={median_delta if median_delta is not None else '—':>8}  {bar}")

            summary_rows.append({
                "digits": D, "samples": total, "hits": hits, "misses": total - hits,
                "hit_rate": round(hit_rate, 6),
                "avg_trials_to_hit": round(avg_tries, 3) if avg_tries is not None else None,
                "median_delta_from_half": median_delta,
                "per_n_trials": args.per_n_trials, "window": args.window, "seed": args.seed
            })

        if args.csv and summary_rows:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"[saved] CSV -> {args.csv}")

        if args.jsonl and jsonl_rows:
            with open(args.jsonl, "w") as f:
                for r in jsonl_rows:
                    f.write(json.dumps(r) + "\n")
            print(f"[saved] JSONL -> {args.jsonl}")

        return

    # ---------- SINGLE-RUN MODE (original behavior) ----------
    for i in range(args.count):
        n = random_even_with_digits(args.digits, rng)
        printed = 0
        for (p, q) in centered_pairs_iter(
            n, rng=rng, per_n_trials=args.per_n_trials, window=args.window
        ):
            if args.pairs_only:
                print(f"{p} + {q} = {n}")
            else:
                dist = abs(p - (n // 2))
                print(f"[{i+1:>3}/{args.count}] n({args.digits}d)  |  p≈n/2? Δ={dist}  ->  {p} + {q} = {n}")
            printed += 1
            if printed >= args.per_n_count:
                break
        if printed == 0 and not args.pairs_only:
            print(f"# miss for n={n} within budget (trials={args.per_n_trials}, window={args.window})")

if __name__ == "__main__":
    main()
