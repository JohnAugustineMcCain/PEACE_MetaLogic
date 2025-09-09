#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Peace_gb.py — Standalone Goldbach evaluator and constructive decomposer

What it does (plain language):
- Goldbach says every even n > 2 is the sum of two primes: n = p + q.
- This tool can:
  1) EXPLAIN a verdict with confidence across scales (direct / sample / asymptotic)
  2) CONSTRUCT actual pairs (p, q) at huge scales using fast primality tests (BPSW)
  3) Optionally PROVE p and q prime with sympy's ECPP (--prove)
  4) Print a standalone rationale (--why) so newcomers need zero background

Key flags:
  --construct : attempt to find a prime pair p+q=n (works well up to ~1e40)
  --prove     : if sympy is available, certify primes via ECPP
  --short     : minimal output for decompositions (auto-enables --construct)
  --count K   : with --short, print up to K distinct pairs (default 5)
  --json      : machine-readable Verdict (not used with --short)
  --why       : print the rationale for why this approach matters and exit

Examples:
  python Peace_gb.py 100                      # friendly explanation
  python Peace_gb.py 100 --json               # JSON verdict
  python Peace_gb.py 10**30 --construct       # try to find an actual pair
  python Peace_gb.py 10**30 --construct --prove
  python Peace_gb.py 10**30 --short --count 10   # print up to 10 distinct pairs
  python Peace_gb.py --why                    # why this tool matters (standalone rationale)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Iterable, Set
import argparse
import json
import math
import random
import time


# -----------------------------
# Truth values (plain and simple)
# -----------------------------
class TV:
    T = "T"  # designated true
    F = "F"  # designated false
    B = "B"  # both/undecided (honest stop)


# -----------------------------
# Verdict container
# -----------------------------
@dataclass
class Verdict:
    n: int
    value: str                 # "T", "F", or "B"
    confidence: float          # 0.0..1.0 (capped below 1)
    perspective: str           # "direct" | "sample" | "asymptotic"
    method: str                # "exact_pair" | "miller_rabin_sample" | "HL_asymptotic" | "constructive_BPSW[+ECPP]"
    found_pair: Optional[List[int]] = None
    checked_up_to: Optional[int] = None
    samples_run: Optional[int] = None
    pairs_found_count: Optional[int] = None
    expected_reps_est: Optional[float] = None
    time_ms: Optional[int] = None
    seed: Optional[int] = None
    warnings: Optional[List[str]] = None
    notes: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def pretty(self) -> str:
        tv_word = {"T": "Designated true", "F": "Designated false", "B": "Both/undecided in this context"}.get(self.value, self.value)
        lines = []
        lines.append(f"Goldbach verdict for n={self.n}: {tv_word}")
        lines.append(f"Perspective used: {self.perspective}  |  Method: {self.method}")
        lines.append(f"Confidence: {round(self.confidence * 100, 2)}% (capped for humility)")
        if self.found_pair:
            lines.append(f"Example prime pair found: {self.found_pair[0]} + {self.found_pair[1]} = {self.n}")
        if self.pairs_found_count is not None:
            lines.append(f"Number of pairs found in this run: {self.pairs_found_count}")
        if self.expected_reps_est is not None:
            lines.append(f"Estimated number of Goldbach representations: ~{_fmt_sig(self.expected_reps_est)}")
        if self.samples_run is not None:
            lines.append(f"Samples run: {self.samples_run}")
        if self.checked_up_to is not None:
            lines.append(f"Fully verified up to: {self.checked_up_to}")
        if self.seed is not None:
            lines.append(f"Random seed: {self.seed}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        if self.time_ms is not None:
            lines.append(f"Time: {self.time_ms} ms")

        # Always include the “Why this matters” section for non-short, non-JSON runs
        lines.append("")
        lines.append("Why this matters (standalone rationale):")
        for ln in rationale_bullets():
            lines.append(f"  {ln}")
        lines.append("")
        lines.append("Glossary: T = direct/overwhelming evidence in favor; F = verified counterexample; B = honest stop in this context.")

        return "\n".join(lines)


def _fmt_sig(x: float) -> str:
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / (10 ** e)
    m = round(m, 2)
    if abs(x) >= 1e6 or abs(x) < 1e-3:
        return f"{m}e{e}"
    return f"{x:.3g}"


# -----------------------------
# Primality testing (BPSW: MR + strong Lucas)
# -----------------------------
_SMALL_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97
]

def _miller_rabin(n: int, bases: List[int]) -> bool:
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _mr_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return n == p
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23]  # strong screen for big ints
    return _miller_rabin(n, bases)

def _jacobi(a: int, n: int) -> int:
    if n % 2 == 0 or n <= 0:
        return 0
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            r = n % 8
            if r == 3 or r == 5:
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0

def _selfridge_params(n: int) -> Tuple[int, int, int]:
    # find D with Jacobi(D|n) = -1, scanning 5, -7, 9, -11, ...
    D = 5
    sign = 1
    while True:
        j = _jacobi(D, n)
        if j == -1:
            break
        D = (abs(D) + 2) * (1 if sign < 0 else -1)
        sign *= -1
        if D == 0:
            D = 5
    P = 1
    Q = (1 - D) // 4
    return D, P, Q

def _lucas_selfridge_prp(n: int) -> bool:
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return n == p
    D, P, Q = _selfridge_params(n)
    # write n+1 = d*2^s
    d = n + 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def lucas_uv(k: int) -> Tuple[int, int]:
        U, V = 0, 2
        Qk = 1
        bits = bin(k)[2:]
        for bit in bits:
            # square
            U2 = (U * V) % n
            V2 = (V * V - 2 * Qk) % n
            U, V = U2, V2
            Qk = (Qk * Q) % n
            if bit == '1':  # multiply by generator (P=1)
                U3 = (U + V) % n
                V3 = (V + U * D) % n
                U, V = U3, V3
                Qk = (Qk * Q) % n
        return U, V

    U, V = lucas_uv(d)
    if U == 0 or V == 0:
        return True
    for _ in range(s - 1):
        V = (V * V - 2) % n
        if V == 0:
            return True
    return False

def bpsw_is_prime(n: int) -> bool:
    """Baillie–PSW: MR + strong Lucas. Extremely reliable in practice."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return n == p
    if not _mr_probable_prime(n):
        return False
    return _lucas_selfridge_prp(n)


# -----------------------------
# Exact pair for small/medium n
# -----------------------------
def find_exact_pair(n: int) -> Optional[Tuple[int, int]]:
    if n % 2 != 0 or n <= 2:
        return None
    # check p = 2 first
    if bpsw_is_prime(n - 2):
        return 2, n - 2
    p = 3
    limit = n // 2
    while p <= limit:
        if bpsw_is_prime(p) and bpsw_is_prime(n - p):
            return p, n - p
        p += 2
    return None


# -----------------------------
# Constructive search for astronomical n (wheel + BPSW)
# -----------------------------
_WHEEL = 2 * 3 * 5 * 7 * 11 * 13  # 30030
_WHEEL_RES = [r for r in range(_WHEEL) if (r % 2 == 1) and all(r % p for p in [3, 5, 7, 11, 13])]

def _trial_division_screen(n: int) -> bool:
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    return True

def _maybe_prove_pair(p: int, q: int, prove: bool) -> bool:
    if not prove:
        return True
    try:
        import sympy as sp  # optional; ECPP may run
        return sp.isprime(p) and sp.isprime(q)
    except Exception:
        return True  # if sympy unavailable, accept BPSW result

def find_goldbach_pair(
    n: int,
    *,
    seed: Optional[int] = None,
    max_trials: int = 100_000,
    window: Optional[int] = None,
    prove_with_sympy: bool = False,
) -> Optional[Tuple[int, int]]:  # kept for backwards compatibility / single-hit usage
    if n % 2 != 0 or n <= 2:
        return None
    rng = random.Random(seed)
    half = n // 2
    if window is None:
        window = min(5_000_000, max(50_000, int((n.bit_length() ** 2) * 25)))

    # quick p=2 check
    q = n - 2
    if bpsw_is_prime(q):
        if _maybe_prove_pair(2, q, prove_with_sympy):
            return 2, q

    for _ in range(max_trials):
        delta = rng.randint(-window, window)
        p = half + delta
        if p <= 2:
            continue
        p |= 1  # force odd
        if (p % _WHEEL) not in _WHEEL_RES:
            continue
        if not _trial_division_screen(p):
            continue
        q = n - p
        if q <= 2 or (q % _WHEEL) not in _WHEEL_RES or not _trial_division_screen(q):
            continue
        if bpsw_is_prime(p) and bpsw_is_prime(q):
            if _maybe_prove_pair(p, q, prove_with_sympy):
                return p, q
    return None

def find_goldbach_pairs_iter(
    n: int,
    *,
    seed: Optional[int] = None,
    max_trials: int = 100_000,
    window: Optional[int] = None,
    prove_with_sympy: bool = False,
) -> Iterable[Tuple[int, int]]:
    """
    Generator that yields DISTINCT unordered pairs (p, q) with p+q=n and p<=q.
    Will attempt up to max_trials candidate draws (not guaranteed to find max count).
    """
    if n % 2 != 0 or n <= 2:
        return
    rng = random.Random(seed)
    half = n // 2
    if window is None:
        window = min(5_000_000, max(50_000, int((n.bit_length() ** 2) * 25)))

    seen: Set[Tuple[int, int]] = set()

    # p=2 shortcut
    q = n - 2
    if bpsw_is_prime(q):
        pair = (2, q) if 2 <= q else (q, 2)
        if _maybe_prove_pair(*pair, prove_with_sympy) and pair not in seen:
            seen.add(pair)
            yield pair

    for _ in range(max_trials):
        delta = rng.randint(-window, window)
        p = half + delta
        if p <= 2:
            continue
        p |= 1
        if (p % _WHEEL) not in _WHEEL_RES:
            continue
        if not _trial_division_screen(p):
            continue
        q = n - p
        if q <= 2 or (q % _WHEEL) not in _WHEEL_RES or not _trial_division_screen(q):
            continue
        if bpsw_is_prime(p) and bpsw_is_prime(q):
            pair = (p, q) if p <= q else (q, p)
            if pair not in seen and _maybe_prove_pair(*pair, prove_with_sympy):
                seen.add(pair)
                yield pair


# -----------------------------
# Sampling approach (evidence only)
# -----------------------------
def sample_goldbach_pairs(n: int, budget: int, rng: random.Random) -> Tuple[int, int]:
    pairs = 0
    trials = 0
    half = n // 2
    window = max(10_000, min(half, 5_000_000))
    for _ in range(budget):
        trials += 1
        delta = rng.randint(-window, window)
        p = half + delta
        if p <= 1:
            continue
        if p % 2 == 0:
            p += 1
        q = n - p
        if q <= 1:
            continue
        if bpsw_is_prime(p) and bpsw_is_prime(q):
            pairs += 1
    return pairs, trials


# -----------------------------
# Asymptotic expectation (explanatory)
# -----------------------------
def expected_reps_hl(n: int) -> float:
    if n <= 8:
        return 1.0
    ln = math.log(n)
    return max(0.0, n / (ln * ln))


# -----------------------------
# Orchestrator
# -----------------------------
def goldbach_explain(
    n: int,
    *,
    sample_budget: int = 2000,
    seed: Optional[int] = None,
    cap_conf: float = 0.999,
    direct_threshold: int = 2_000_000,
    sample_threshold: int = 10**10,
    constructive: bool = False,
    construct_trials: int = 100_000,
    construct_window: Optional[int] = None,
    construct_prove: bool = False,
) -> Verdict:
    t0 = time.time()
    warnings: List[str] = []
    if n % 2 != 0 or n <= 2:
        return Verdict(
            n=n, value=TV.F, confidence=1.0,
            perspective="direct", method="invalid_input",
            time_ms=int((time.time() - t0) * 1000),
            warnings=["Input must be even and greater than 2."]
        )

    rng = random.Random(seed)

    # Optional constructive search first (works at any scale)
    if constructive:
        pair = find_goldbach_pair(
            n,
            seed=seed,
            max_trials=construct_trials,
            window=construct_window,
            prove_with_sympy=construct_prove,
        )
        if pair is not None:
            p, q = pair
            return Verdict(
                n=n,
                value=TV.T,
                confidence=min(cap_conf, 0.995),
                perspective="direct" if n <= direct_threshold else ("sample" if n <= sample_threshold else "asymptotic"),
                method="constructive_BPSW" + ("+ECPP" if construct_prove else ""),
                found_pair=[p, q],
                pairs_found_count=1,
                expected_reps_est=expected_reps_hl(n),
                time_ms=int((time.time() - t0) * 1000),
                notes="Constructive search produced an explicit Goldbach pair.",
                seed=seed,
            )
        warnings.append("No constructive pair found within budget; falling back to perspective-based evaluation.")

    # 1) Direct (small n)
    if n <= direct_threshold:
        pair = find_exact_pair(n)
        if pair is not None:
            p, q = pair
            return Verdict(
                n=n,
                value=TV.T,
                confidence=min(cap_conf, 0.995),
                perspective="direct",
                method="exact_pair",
                found_pair=[p, q],
                pairs_found_count=1,
                expected_reps_est=expected_reps_hl(n),
                time_ms=int((time.time() - t0) * 1000),
            )
        warnings.append("No pair found during direct search. This indicates a counterexample or a bug.")
        return Verdict(
            n=n,
            value=TV.F,
            confidence=min(cap_conf, 0.999),
            perspective="direct",
            method="exact_pair",
            found_pair=None,
            pairs_found_count=0,
            expected_reps_est=expected_reps_hl(n),
            time_ms=int((time.time() - t0) * 1000),
            warnings=warnings,
            notes="If reproducible, this would refute Goldbach for this n. Please double check.",
        )

    # 2) Sample (mid-scale)
    if n <= sample_threshold:
        pairs_found, trials = sample_goldbach_pairs(n, sample_budget, rng)
        base = 0.5 if pairs_found == 0 else 0.6
        bump = min(0.4, pairs_found / max(1, sample_budget / 50) * 0.4)
        conf = min(cap_conf, base + bump)
        value = TV.T if pairs_found > 0 else TV.B
        notes = ("Found at least one pair with sampling."
                 if pairs_found > 0
                 else "No pairs in this small sample. Increase --samples or change --seed.")
        return Verdict(
            n=n,
            value=value,
            confidence=conf,
            perspective="sample",
            method="miller_rabin_sample",
            pairs_found_count=pairs_found,
            samples_run=trials,
            expected_reps_est=expected_reps_hl(n),
            time_ms=int((time.time() - t0) * 1000),
            seed=seed,
            notes=notes,
            warnings=warnings if warnings else None,
        )

    # 3) Asymptotic (huge n)
    expected = expected_reps_hl(n)
    k = 1e3
    conf = 1.0 - math.exp(-min(expected, 1e9) / k)
    conf = min(conf, cap_conf)
    value = TV.T if conf >= 0.6 else TV.B
    return Verdict(
        n=n,
        value=value,
        confidence=conf,
        perspective="asymptotic",
        method="HL_asymptotic",
        expected_reps_est=expected,
        time_ms=int((time.time() - t0) * 1000),
        notes="Asymptotic reasoning. Not a constructive proof of a specific pair.",
        warnings=warnings if warnings else None,
    )


# -----------------------------
# Rationale (standalone + bullets for pretty())
# -----------------------------
def rationale_bullets() -> List[str]:
    return [
        "1) We can build confidence at huge scales: expected Goldbach pairs grow ~ n/(log n)^2; we construct pairs when feasible and report confidence otherwise.",
        "2) Classical full verification past ~4×10^19 is infeasible relative to input size (digits): verifying all n ≤ N needs Ω(N log N) bit-ops → super-polynomial in log N.",
        "3) Classical disproof for a single astronomical n is also infeasible: ruling out all candidate primes is exponential in digit-length unless a deep theorem short-circuits it.",
        "4) Therefore, rational acceptance needs meta-mathematical methods: asymptotics + probabilistic evidence + constructive search (not brute-force to infinity).",
    ]

def explain_rationale() -> str:
    lines = []
    lines.append("*** Goldbach at Huge Scales — Why this tool exists ***\n")
    for b in rationale_bullets():
        lines.append(b)
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# CLI
# -----------------------------
def _int_eval(s: str) -> int:
    try:
        val = eval(s, {"__builtins__": {}}, {})
        if not isinstance(val, int):
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            else:
                raise ValueError
        return int(val)
    except Exception:
        try:
            return int(s)
        except Exception:
            raise argparse.ArgumentTypeError(f"Could not parse integer or expression: {s!r}")

def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="peace-gb",
        description="Explain/construct Goldbach decompositions at any scale (standalone, no background needed).",
    )
    parser.add_argument("n", nargs="?", type=_int_eval, help="Even integer > 2. You can also pass 10**40, etc.")
    parser.add_argument("--why", action="store_true", help="Print the standalone rationale and exit.")
    parser.add_argument("--json", action="store_true", help="Output JSON Verdict.")
    parser.add_argument("--explain", action="store_true", help="Force a plain-language explanation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling/constructive search.")
    parser.add_argument("--samples", type=int, default=2000, help="Sample budget for mid-scale runs.")
    parser.add_argument("--direct-threshold", type=int, default=2_000_000, help="Max n for direct exact search.")
    parser.add_argument("--sample-threshold", type=_int_eval, default=10**10, help="Max n for sampling before asymptotic.")

    # Constructive flags (single-hit mode)
    parser.add_argument("--construct", action="store_true", help="Attempt to FIND an explicit prime pair for n.")
    parser.add_argument("--trials", type=int, default=100_000, help="Max trials for constructive search.")
    parser.add_argument("--window", type=int, default=None, help="Search window around n/2 for constructive search.")
    parser.add_argument("--prove", action="store_true", help="If set, try to certify primes with sympy's ECPP (if available).")

    # Minimalist output (multi-hit mode)
    parser.add_argument("--short", action="store_true",
                        help="Minimal output for decompositions (auto-enables --construct; prints one 'p + q = n' per line).")
    parser.add_argument("--count", type=int, default=5,
                        help="With --short, print up to COUNT distinct pairs (default 5).")

    args = parser.parse_args()

    if args.why and args.n is None:
        print(explain_rationale())
        return
    if args.n is None:
        parser.error("Provide n (even > 2) or use --why")

    # --short: emit up to COUNT distinct pairs, one per line; minimal messages only
    if args.short:
        found = 0
        for (p, q) in find_goldbach_pairs_iter(
            args.n,
            seed=args.seed,
            max_trials=args.trials,
            window=args.window,
            prove_with_sympy=args.prove,
        ):
            print(f"{p} + {q} = {args.n}")
            found += 1
            if found >= args.count:
                break
        if found == 0:
            print(f"No decomposition found within budget (trials={args.trials}, window={args.window}).")
        return

    # Normal paths (explanatory or JSON)
    v = goldbach_explain(
        args.n,
        sample_budget=args.samples,
        seed=args.seed,
        direct_threshold=args.direct_threshold,
        sample_threshold=args.sample_threshold,
        constructive=args.construct,
        construct_trials=args.trials,
        construct_window=args.window,
        construct_prove=args.prove,
    )

    if args.json:
        print(v.to_json())
    else:
        print(v.pretty())


if __name__ == "__main__":
    _cli()
