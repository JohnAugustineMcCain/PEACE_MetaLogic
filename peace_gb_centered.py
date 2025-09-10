#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# peace_gb_centered.py  —  Clean CLI (no Δ by default)
#
# Three-strategy Goldbach sampler:
#   (digits ≤ 18):   Center → Adaptive → Subtractors
#   (digits > 18):   Subtractors → Adaptive → Center   <-- NEW ORDER
#
# + Sweep mode (per-digit hit-rate, Wilson CIs, Bayes summary)
# + BPSW primality (+ optional extra MR rounds)
# + Shows decompositions from the HIGHEST digit size via --examples
# + Δ metrics hidden by default; enable with --show-delta
#
# Defaults (per your request):
#   --trials-center = 1000
#   --trials-adapt  = 1000
#   --subs-max-checks = 1000
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

def _median(lst: List[int]) -> Optional[int]:
    if not lst: return None
    s = sorted(lst); m = len(s)//2
    return s[m] if len(s)%2==1 else (s[m-1]+s[m])//2

# ------------- Generators for examples ----------
def centered_pairs_iter(
    n: int,
    *,
    rng: random.Random,
    per_n_trials: int,
    window: Optional[int] = None,
) -> Iterable[Tuple[int, int]]:
    """Yield DISTINCT unordered prime pairs (p,q) with p+q=n, prioritizing p≈n/2."""
    if n % 2 != 0 or n < 4: return
    half = n // 2
    if window is None:
        window = max(50_000, min(5_000_000, (n.bit_length() ** 2) * 25))
    seen: Set[Tuple[int, int]] = set()

    # quick p=2
    q0 = n - 2
    if is_probable_prime(q0, rng):
        pair = (2, q0)
        pair = (pair[0], pair[1]) if pair[0] <= pair[1] else (pair[1], pair[0])
        if pair not in seen:
            seen.add(pair); yield pair

    tries = 0
    while tries < per_n_trials:
        delta = rng.randint(-window, window)
        p = half + delta
        if p <= 2:
            tries += 1; continue
        if p % 2 == 0:
            p += 1 if delta <= 0 else -1
            if p <= 2:
                tries += 1; continue
        q = n - p
        if q <= 2:
            tries += 1; continue
        if is_probable_prime(p, rng) and is_probable_prime(q, rng):
            pair = (p, q) if p <= q else (q, p)
            if pair not in seen:
                seen.add(pair); yield pair
        tries += 1

def subtractor_pairs_iter(
    n: int,
    *,
    rng: random.Random,
    ceiling: int,
    max_checks: int
) -> Iterable[Tuple[int,int]]:
    """Yield prime pairs via random subtractor primes p ≤ ceiling with q=n-p prime."""
    if n % 2 != 0 or n < 4: return
    primes = _sieve_upto(ceiling)
    rng.shuffle(primes)
    seen: Set[Tuple[int,int]] = set()
    checks = 0
    for p in primes:
        if max_checks is not None and checks >= max_checks: break
        q = n - p
        if q > 2 and is_probable_prime(q, rng):
            pair = (p, q) if p <= q else (q, p)
            if pair not in seen:
                seen.add(pair); yield pair
        checks += 1

# ------------- Core search primitives ----------
def _search_around_target(
    n: int,
    target_p: int,
    *,
    rng: random.Random,
    trials: int,
    window: Optional[int]
) -> Tuple[bool, Optional[Tuple[int,int]], int, Optional[Tuple[int,int]]]:
    """
    Probe p near target_p; return (found, (Δ_half, Δ_phase), tries_used, (p,q))
    (Δ values returned for optional diagnostics; CLI hides them by default.)
    """
    half = n // 2
    if window is None:
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
            delta_half   = abs(pair[0] - half)
            delta_phase  = abs(pair[0] - target_p)
            return True, (delta_half, delta_phase), tries + 1, pair
        tries += 1
    return False, None, tries, None

def _sieve_upto(limit: int) -> List[int]:
    if limit < 2: return []
    sieve = bytearray(b"\x01")*(limit+1)
    sieve[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for p in range(2, r+1):
        if sieve[p]:
            start = p*p; step = p
            sieve[start:limit+1:step] = b"\x00" * (((limit - start)//step) + 1)
    return [i for i, b in enumerate(sieve) if b]

def _search_by_subtractors(
    n: int,
    *,
    rng: random.Random,
    ceiling: int,
    max_checks: int
) -> Tuple[bool, Optional[Tuple[int,int]], int, Optional[Tuple[int,int]]]:
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
            delta_half = abs(p - half)  # measured from chosen subtractor p
            delta_phase = 0
            return True, (delta_half, delta_phase), tries + 1, pair
        tries += 1
    return False, None, tries, None

# ------------- First-hit wrappers --------------
def first_hit_center(
    n: int, *, rng: random.Random, trials_center: int, window_center: Optional[int]
) -> Tuple[bool, Optional[Tuple[int,int]], int, Optional[Tuple[int,int]], str]:
    """Center-only first-hit."""
    if n % 2 != 0 or n < 4:
        return False, None, 0, None, "miss"
    q0 = n - 2
    if is_probable_prime(q0, rng):
        return True, (abs(2 - (n//2)), 0), 0, (2, q0), "center"
    ok, deltas, t, pair = _search_around_target(n, n//2, rng=rng, trials=trials_center, window=window_center)
    return ok, deltas, t, pair, ("center" if ok else "miss")

def first_hit_adaptive(
    n: int,
    *,
    rng: random.Random,
    total_trials: int,
    window: Optional[int],
    fractions: List[float],
    mini_batch: int,
    subs_ceiling: int,
    subs_max_checks: int,
    allow_subs_fallback: bool = True  # NEW: so we don't double-run subtractors in subs-first order
) -> Tuple[bool, Optional[Tuple[int,int]], int, Optional[Tuple[int,int]], str, float]:
    """
    Thompson-sampling over target fractions p≈f*n.
    Returns (found, (Δ_half, Δ_phase), total_tries, pair, phase, f_used)
    phase ∈ {"adapt","subs","miss"}
    """
    if n % 2 != 0 or n < 4:
        return False, None, 0, None, "miss", 0.5

    q0 = n - 2
    if is_probable_prime(q0, rng):
        return True, (abs(2 - (n//2)), abs(2 - int(0.5*n))), 0, (2, q0), "adapt", 0.5

    alpha = [1.0]*len(fractions)
    beta  = [1.0]*len(fractions)
    tries_total = 0

    def _mini_batch_at(target_p: int) -> Tuple[bool, Optional[Tuple[int,int]], int, Optional[Tuple[int,int]]]:
        return _search_around_target(n, target_p, rng=rng, trials=mini_batch, window=window)

    while tries_total < total_trials:
        thetas = [rng.betavariate(alpha[i], beta[i]) for i in range(len(fractions))]
        i = max(range(len(fractions)), key=lambda k: thetas[k])
        f = fractions[i]
        target_p = max(3, int(round(f * n)))

        found, deltas, t, pair = _mini_batch_at(target_p)
        tries_total += t

        if found:
            alpha[i] += 1.0
            return True, deltas, tries_total, pair, "adapt", f
        else:
            beta[i] += 1.0

        if tries_total + mini_batch > total_trials:
            break

    if allow_subs_fallback:
        found, deltas, t, pair = _search_by_subtractors(
            n, rng=rng, ceiling=subs_ceiling, max_checks=subs_max_checks
        )
        tries_total += t
        if found:
            return True, deltas, tries_total, pair, "subs", -1.0

    return False, None, tries_total, None, "miss", -1.0

# ---------------- Rationale --------------------
def rationale() -> str:
    return (
        "\n*** Goldbach — Center, Adaptive Bandit, and Subtractor Probing ***\n\n"
        "We test p+q=n by sampling near the balanced split p≈n/2 where valid pairs concentrate, "
        "adaptively sampling near p≈f*n (f∈{0.5, 2/3, 1/3, 0.6, 0.4}) via Thompson sampling, and trying "
        "random prime subtractors p≤C with q=n−p. For large n we may prefer subtractors first to quickly jump to a valid q=n−p.\n\n"
        "Primality is checked by Baillie–PSW (practically deterministic) with optional extra Miller–Rabin rounds."
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

def _parse_fractions(s: str) -> List[float]:
    out = []
    for tok in s.split(","):
        v = float(tok.strip())
        if not (0.0 < v < 1.0):
            raise argparse.ArgumentTypeError(f"fractions must be in (0,1): {tok}")
        out.append(v)
    if 0.5 not in out:
        out.insert(0, 0.5)
    return out

# ---------------------- Main -------------------
def main() -> None:
    global _EXTRA_MR_ROUNDS

    ap = argparse.ArgumentParser(
        prog="peace-gb-centered",
        description="Goldbach sampler: (≤18d) center→adaptive→subs ; (>18d) subs→adaptive→center. Sweep with CIs/Bayes."
    )
    # Single-run basics
    ap.add_argument("--digits", type=lambda s: _positive_int("--digits", s),
                    help="Digits for single-run (mutually exclusive with sweep).")
    ap.add_argument("--count", type=lambda s: _positive_int("--count", s), default=10,
                    help="How many random n in single-run (default 10).")
    ap.add_argument("--pairs-only", action="store_true", help="Only print 'p + q = n' lines (single-run).")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed (default 2025).")
    ap.add_argument("--why", action="store_true", help="Print the one-page rationale and exit.")

    # Center budget/window (DEFAULT 1000)
    ap.add_argument("--trials-center", type=int, default=1000,
                    help="Trials for center (n/2) phase (default 1000).")
    ap.add_argument("--window-center", type=int, default=None, help="Window around n/2 (default adaptive).")

    # Adaptive controller (DEFAULT total 1000)
    ap.add_argument("--adaptive", action="store_true", help="Enable adaptive bandit phase.")
    ap.add_argument("--fractions", type=str, default="0.5,0.6666667,0.3333333,0.6,0.4",
                    help="Comma-separated fractions f for targets p≈f*n.")
    ap.add_argument("--mini-batch", type=int, default=500,
                    help="Trials per mini-batch in adaptive phase (default 500).")
    ap.add_argument("--trials-adapt", type=int, default=1000,
                    help="Total trials budget for adaptive phase (default 1000).")
    ap.add_argument("--window-adapt", type=int, default=None, help="Window during adaptive (default adaptive).")

    # Subtractors (DEFAULT checks 1000)
    ap.add_argument("--subs-ceiling", type=int, default=200000,
                    help="Prime sieve ceiling for subtractors (default 200k).")
    ap.add_argument("--subs-max-checks", type=int, default=1000,
                    help="Max subtractor primes to try (default 1000).")

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
    ap.add_argument("--examples", type=int, default=0,
                    help="Print this many decompositions from the highest digit tested (default 0 = none).")

    # Stats outputs
    ap.add_argument("--ci", action="store_true", help="Print Wilson 95% CI for hit-rate.")
    ap.add_argument("--bayes-eps", type=float, default=0.1,
                    help="Null miss-rate epsilon (p0=1-eps). Default 0.1 → p0=0.9.")
    ap.add_argument("--bayes-summary", action="store_true",
                    help="Print cumulative log10 Bayes factor vs fixed p0.")

    # Show Δ only if asked
    ap.add_argument("--show-delta", action="store_true",
                    help="Show Δ metrics in CLI/CSV/JSONL (hidden by default).")

    args = ap.parse_args()
    if args.why:
        print(rationale()); return

    _EXTRA_MR_ROUNDS = max(0, int(args.extra_mr_rounds))
    rng = random.Random(args.seed)
    fractions = _parse_fractions(args.fractions)

    def _large_digits(D: int) -> bool:
        return D > 18

    # ---------- SWEEP MODE ----------
    in_sweep = (args.digits_list is not None) or (args.sweep is not None)
    if in_sweep:
        digits_space: List[int] = []
        if args.digits_list: digits_space.extend(args.digits_list)
        if args.sweep: digits_space.extend(args.sweep)
        digits_space = sorted(set(digits_space))
        dlo, dhi = digits_space[0], digits_space[-1]

        if not args.quiet:
            print("\n# Goldbach sweep (order: ≤18d center→adapt→subs | >18d subs→adapt→center)")
            print(f"# center_trials={args.trials_center} | adapt_trials={args.trials_adapt if args.adaptive else 0} "
                  f"| subs_ceiling={args.subs_ceiling} | subs_max_checks={args.subs_max_checks}")
            print(f"# samples per digit={args.samples} | seed={args.seed} | extraMR={_EXTRA_MR_ROUNDS}\n")

        p0 = max(0.0, min(1.0, 1.0 - args.bayes_eps))
        sum_log10_BF = 0.0
        total_h = 0; total_n = 0
        summary_rows: List[Dict[str, Any]] = []
        jsonl_rows: List[Dict[str, Any]] = []

        for D in digits_space:
            hits = 0
            tries_on_hits: List[int] = []
            deltas_half_on_hits: List[int] = []
            deltas_phase_on_hits: List[int] = []
            phase_counts = {"center":0, "adapt":0, "subs":0, "miss":0}

            for _ in range(args.samples):
                n = random_even_with_digits(D, rng)
                tries_used = 0
                found = False
                deltas = None
                pair = None
                phase = "miss"
                f_used = None

                if _large_digits(D):
                    # Order: SUBS → ADAPT → CENTER
                    foundS, deltasS, tS, pairS = _search_by_subtractors(
                        n, rng=rng, ceiling=args.subs_ceiling, max_checks=args.subs_max_checks
                    )
                    tries_used += tS
                    if foundS:
                        found, deltas, pair, phase = True, deltasS, pairS, "subs"
                    if (not found) and args.adaptive:
                        foundA, deltasA, tA, pairA, phaseA, fA = first_hit_adaptive(
                            n,
                            rng=rng,
                            total_trials=args.trials_adapt,
                            window=args.window_adapt,
                            fractions=fractions,
                            mini_batch=args.mini_batch,
                            subs_ceiling=args.subs_ceiling,
                            subs_max_checks=args.subs_max_checks,
                            allow_subs_fallback=False  # already ran subs
                        )
                        tries_used += tA
                        if foundA:
                            found, deltas, pair, phase, f_used = True, deltasA, pairA, "adapt", fA
                    if not found:
                        foundC, deltasC, tC, pairC, phaseC = first_hit_center(
                            n, rng=rng, trials_center=args.trials_center, window_center=args.window_center
                        )
                        tries_used += tC
                        if foundC:
                            found, deltas, pair, phase = True, deltasC, pairC, "center"
                        else:
                            phase = "miss"
                else:
                    # Order: CENTER → ADAPT → SUBS (original)
                    foundC, deltasC, tC, pairC, phaseC = first_hit_center(
                        n, rng=rng, trials_center=args.trials_center, window_center=args.window_center
                    )
                    tries_used += tC
                    if foundC:
                        found, deltas, pair, phase = True, deltasC, pairC, "center"
                    if (not found) and args.adaptive:
                        foundA, deltasA, tA, pairA, phaseA, fA = first_hit_adaptive(
                            n,
                            rng=rng,
                            total_trials=args.trials_adapt,
                            window=args.window_adapt,
                            fractions=fractions,
                            mini_batch=args.mini_batch,
                            subs_ceiling=args.subs_ceiling,
                            subs_max_checks=args.subs_max_checks,
                            allow_subs_fallback=True
                        )
                        tries_used += tA
                        if foundA:
                            found, deltas, pair, phase, f_used = True, deltasA, pairA, "adapt", fA
                    if (not found) and (not args.adaptive):
                        foundS, deltasS, tS, pairS = _search_by_subtractors(
                            n, rng=rng, ceiling=args.subs_ceiling, max_checks=args.subs_max_checks
                        )
                        tries_used += tS
                        if foundS:
                            found, deltas, pair, phase = True, deltasS, pairS, "subs"
                        else:
                            phase = "miss"

                phase_counts[phase] += 1

                if found:
                    hits += 1
                    tries_on_hits.append(tries_used)
                    if deltas is not None:
                        deltas_half_on_hits.append(deltas[0])
                        deltas_phase_on_hits.append(deltas[1])

                if args.jsonl:
                    rec = {
                        "digits": D, "n": str(n), "found": found,
                        "tries_used": tries_used, "phase": phase, "f_used": f_used,
                        "center_trials": args.trials_center, "adapt_trials": (args.trials_adapt if args.adaptive else 0),
                        "subs_ceiling": args.subs_ceiling, "subs_max_checks": args.subs_max_checks,
                        "seed": args.seed, "extra_mr_rounds": _EXTRA_MR_ROUNDS
                    }
                    if args.show_delta:
                        rec["delta_half"]  = deltas[0] if deltas else None
                        rec["delta_phase"] = deltas[1] if deltas else None
                    jsonl_rows.append(rec)

            total = args.samples; total_h += hits; total_n += total
            hit_rate = hits/total if total else 0.0
            avg_tries = (sum(tries_on_hits)/len(tries_on_hits)) if tries_on_hits else None
            median_delta_half   = _median(deltas_half_on_hits)
            median_delta_phase  = _median(deltas_phase_on_hits)
            lo95, hi95 = _wilson_ci(hits, total) if args.ci else (None, None)

            if args.bayes_summary and total > 0:
                p_hat = min(1 - 1e-12, max(1e-12, hit_rate))
                ll1 = _log10_binom_likelihood(hits, total, p_hat)
                ll0 = _log10_binom_likelihood(hits, total, p0)
                sum_log10_BF += (ll1 - ll0)

            bar = _ascii_bar(hit_rate)
            phases_txt = f"  [center:{phase_counts['center']} adapt:{phase_counts['adapt']} subs:{phase_counts['subs']} miss:{phase_counts['miss']}]"
            ci_txt = f"  CI95=[{lo95:.3f},{hi95:.3f}]" if args.ci else ""
            if args.show_delta:
                delta_txt = f"  medianΔ_half={median_delta_half if median_delta_half is not None else '—':>8}  medianΔ_phase={median_delta_phase if median_delta_phase is not None else '—':>8}  "
            else:
                delta_txt = ""
            print(f"{D:>3}d  hit_rate={hit_rate:6.3f}{ci_txt}  avg_trials_to_hit={('%.1f'%avg_tries) if avg_tries is not None else '—':>6}  "
                  f"{delta_txt}{bar}{phases_txt}")

            row = {
                "digits": D, "samples": total, "hits": hits, "misses": total - hits,
                "hit_rate": round(hit_rate, 6),
                "avg_trials_to_hit": round(avg_tries, 3) if avg_tries is not None else None,
                "phase_center": phase_counts["center"],
                "phase_adapt": phase_counts["adapt"],
                "phase_subs": phase_counts["subs"],
                "phase_miss": phase_counts["miss"],
                "center_trials": args.trials_center, "adapt_trials": (args.trials_adapt if args.adaptive else 0),
                "subs_ceiling": args.subs_ceiling, "subs_max_checks": args.subs_max_checks,
                "window_center": args.window_center, "window_adapt": args.window_adapt,
                "seed": args.seed, "extra_mr_rounds": _EXTRA_MR_ROUNDS
            }
            if args.ci:
                row["hit_rate_ci95_low"] = round(lo95, 6)
                row["hit_rate_ci95_high"] = round(hi95, 6)
            if args.show_delta:
                row["median_delta_from_half"]  = median_delta_half
                row["median_delta_from_phase"] = median_delta_phase
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

        if args.bayes_summary:
            print(f"\n# Bayes summary vs fixed p0={1-args.bayes_eps:.3f}")
            print(f"log10 Bayes factor (cumulative): {sum_log10_BF:.3f}")
            print("Interpretation:", "favoring Goldbach-friendly model" if sum_log10_BF>0 else ("favoring fixed-p0 null" if sum_log10_BF<0 else "indifferent"))

        print("\n# Meta-summary")
        overall_rate = total_h/total_n if total_n else 0.0
        print(f"overall hit_rate={overall_rate:.3f} across {total_n} trials | seed={args.seed} | extraMR={_EXTRA_MR_ROUNDS}")

        # Examples from highest digit
        if args.examples > 0:
            print("\n# Example decompositions at the HIGHEST digit size tested")
            D = dhi
            n = random_even_with_digits(D, rng)
            printed = 0
            # try subs first for high digits
            for (p, q) in subtractor_pairs_iter(
                n, rng=rng, ceiling=args.subs_ceiling, max_checks=max(args.subs_max_checks, 1000)
            ):
                print(f"{D:>3}d  n={n}  (subs)    {p} + {q} = {n}")
                printed += 1
                if printed >= args.examples: break
            if printed < args.examples:
                for (p, q) in centered_pairs_iter(
                    n, rng=rng, per_n_trials=max(1000, args.trials_center), window=args.window_center
                ):
                    print(f"{D:>3}d  n={n}  (center)  {p} + {q} = {n}")
                    printed += 1
                    if printed >= args.examples: break
            if printed == 0:
                print(f"(no examples found within budgets at {D} digits for n={n})")
        return

    # ---------- SINGLE-RUN ----------
    if args.digits is None:
        ap.error("Provide --digits for single-run OR use sweep flags. Use --why for the rationale.")

    D = args.digits
    for i in range(args.count):
        n = random_even_with_digits(D, rng)

        tries_used = 0
        found = False
        deltas = None
        pair = None
        phase = "miss"
        f_used = None

        if D > 18:
            # Order: SUBS → ADAPT → CENTER
            foundS, deltasS, tS, pairS = _search_by_subtractors(
                n, rng=rng, ceiling=args.subs_ceiling, max_checks=args.subs_max_checks
            )
            tries_used += tS
            if foundS:
                found, deltas, pair, phase = True, deltasS, pairS, "subs"
            if (not found) and args.adaptive:
                foundA, deltasA, tA, pairA, phaseA, fA = first_hit_adaptive(
                    n,
                    rng=rng,
                    total_trials=args.trials_adapt,
                    window=args.window_adapt,
                    fractions=_parse_fractions(args.fractions),
                    mini_batch=args.mini_batch,
                    subs_ceiling=args.subs_ceiling,
                    subs_max_checks=args.subs_max_checks,
                    allow_subs_fallback=False
                )
                tries_used += tA
                if foundA:
                    found, deltas, pair, phase, f_used = True, deltasA, pairA, "adapt", fA
            if not found:
                foundC, deltasC, tC, pairC, phaseC = first_hit_center(
                    n, rng=rng, trials_center=args.trials_center, window_center=args.window_center
                )
                tries_used += tC
                if foundC:
                    found, deltas, pair, phase = True, deltasC, pairC, "center"
                else:
                    phase = "miss"
        else:
            # Order: CENTER → ADAPT → SUBS
            foundC, deltasC, tC, pairC, phaseC = first_hit_center(
                n, rng=rng, trials_center=args.trials_center, window_center=args.window_center
            )
            tries_used += tC
            if foundC:
                found, deltas, pair, phase = True, deltasC, pairC, "center"
            if (not found) and args.adaptive:
                foundA, deltasA, tA, pairA, phaseA, fA = first_hit_adaptive(
                    n,
                    rng=rng,
                    total_trials=args.trials_adapt,
                    window=args.window_adapt,
                    fractions=_parse_fractions(args.fractions),
                    mini_batch=args.mini_batch,
                    subs_ceiling=args.subs_ceiling,
                    subs_max_checks=args.subs_max_checks,
                    allow_subs_fallback=True
                )
                tries_used += tA
                if foundA:
                    found, deltas, pair, phase, f_used = True, deltasA, pairA, "adapt", fA
            if (not found) and (not args.adaptive):
                foundS, deltasS, tS, pairS = _search_by_subtractors(
                    n, rng=rng, ceiling=args.subs_ceiling, max_checks=args.subs_max_checks
                )
                tries_used += tS
                if foundS:
                    found, deltas, pair, phase = True, deltasS, pairS, "subs"
                else:
                    phase = "miss"

        if args.pairs_only:
            if found and pair:
                p, q = pair
                print(f"{p} + {q} = {n}")
            else:
                print(f"# miss for n={n}")
            continue

        if found and pair:
            p, q = pair
            if args.show_delta:
                dh = deltas[0] if deltas else '—'
                dp = deltas[1] if deltas else '—'
                print(f"[{i+1:>3}/{args.count}] phase={phase:<6} tries={tries_used:<6} Δ_half={dh:>8} Δ_phase={dp:>8} "
                      f"f={('%.6f'%f_used) if f_used not in (None, -1.0) else ('—' if f_used is None else 'subs'):>8}  ->  {p} + {q} = {n}")
            else:
                print(f"[{i+1:>3}/{args.count}] phase={phase:<6} tries={tries_used:<6} "
                      f"f={('%.6f'%f_used) if f_used not in (None, -1.0) else ('—' if f_used is None else 'subs'):>8}  ->  {p} + {q} = {n}")
        else:
            if args.show_delta:
                print(f"[{i+1:>3}/{args.count}] phase=miss   tries={tries_used:<6} Δ_half=       — Δ_phase=       — f=       —  ->  # miss for n={n}")
            else:
                print(f"[{i+1:>3}/{args.count}] phase=miss   tries={tries_used:<6} f=       —  ->  # miss for n={n}")

if __name__ == "__main__":
    main()
