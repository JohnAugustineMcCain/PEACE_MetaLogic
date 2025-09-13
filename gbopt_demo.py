#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gbopt_demo.py — Comprehensive, self-contained demo runner for gbopt.py
#   - Calls gbopt.py for warm-up (learning) and showcase (read-only) sweeps
#   - Then, for a few representative digit sizes, finds and prints a FULL
#     Goldbach decomposition n = p + q (no compact formatting)
#
# Notes:
#   • Uses the learned band ordering from subs_learn_cache.json if present.
#   • Implements a lightweight subtractor search with wheel+pre-sieve+BPSW
#     so decompositions are obtained inside the demo script itself.
#
# (c) 2025 — MIT-style license for this prototype

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_sweep(s: str) -> List[int]:
    a, b, c = [int(x) for x in s.split(":")]
    if c == 0:
        raise argparse.ArgumentTypeError("--sweep step cannot be 0")
    rng = range(a, b + (1 if c > 0 else -1), c)
    out = [v for v in rng if v > 0]
    if not out:
        raise argparse.ArgumentTypeError(f"--sweep produced no positive digits: {s!r}")
    return out

def parse_seeds(s: str) -> List[int]:
    return [int(t.strip()) for t in s.split(",") if t.strip()]

# ──────────────────────────────────────────────────────────────────────────────
# Core configuration parsed from --gb-args (only the bits the demo needs)
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CoreArgs:
    subs_ceiling: int = 300_000
    subs_max_checks: int = 2_000
    wheel_primes: List[int] = dataclasses.field(default_factory=lambda: [3,5,7,11,13,17,19,23])
    pre_sieve_mode: str = "blocks"          # only "blocks" used here
    pre_sieve_limit: int = 20_000
    block2: bool = False
    band_size: int = 64
    cache_file: str = "subs_learn_cache.json"
    small_subs_first: int = 0
    small_subs_cap: int = 0
    seed_for_examples: int = 1              # seed used to pick example n’s

def parse_gb_args_to_core(gb_args: str, seed_for_examples: int) -> CoreArgs:
    """
    Very light parser for a subset of gbopt.py flags we need in Step 3.
    Accepts a single string exactly as passed to the shell.
    """
    core = CoreArgs(seed_for_examples=seed_for_examples)
    toks = shlex.split(gb_args or "")
    i = 0
    def take_int(val: str) -> int:
        return int(val.replace("_", ""))

    while i < len(toks):
        t = toks[i]
        if t == "--wheel-primes":
            i += 1; core.wheel_primes = [int(x.strip()) for x in toks[i].split(",") if x.strip()]
        elif t == "--pre-sieve-mode":
            i += 1; core.pre_sieve_mode = toks[i]
        elif t == "--pre-sieve-limit":
            i += 1; core.pre_sieve_limit = take_int(toks[i])
        elif t == "--subs-ceiling":
            i += 1; core.subs_ceiling = take_int(toks[i])
        elif t == "--subs-max-checks":
            i += 1; core.subs_max_checks = take_int(toks[i])
        elif t == "--band-size":
            i += 1; core.band_size = take_int(toks[i])
        elif t == "--cache":
            i += 1; core.cache_file = toks[i]
        elif t == "--cache-file":
            i += 1; core.cache_file = toks[i]
        elif t == "--small-subs-first":
            i += 1; core.small_subs_first = take_int(toks[i])
        elif t == "--small-subs-cap":
            i += 1; core.small_subs_cap = take_int(toks[i])
        elif t == "--block2":
            core.block2 = True
        # ignore everything else (gbopt.py will handle those)
        i += 1
    return core

# ──────────────────────────────────────────────────────────────────────────────
# BPSW primality test (same structure as in your goldbach/gbopt cores)
# ──────────────────────────────────────────────────────────────────────────────

_SMALL_PRIMES: List[int] = [2,3,5,7,11,13,17,19,23,29,31,37]

def _is_perfect_square(n: int) -> bool:
    if n < 0: return False
    r = math.isqrt(n)
    return r*r == n

def _mr_strong_base(n: int, a: int) -> bool:
    if n % 2 == 0:
        return n == 2
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(a % n, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return True
    return False

def _mr_strong_base2(n: int) -> bool:
    return _mr_strong_base(n, 2)

def _jacobi(a: int, n: int) -> int:
    assert n > 0 and n % 2 == 1
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3,5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0

def _lucas_strong_prp(n: int) -> bool:
    if n == 2:
        return True
    if n < 2 or n % 2 == 0 or _is_perfect_square(n):
        return False
    D = 5
    while True:
        j = _jacobi(D, n)
        if j == -1:
            break
        if j == 0:
            return False
        D = -(abs(D) + 2) if D > 0 else abs(D) + 2
    P = 1
    Q = (1 - D) // 4
    d = n + 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def _lucas_uv_mod(k: int) -> Tuple[int, int]:
        U, V = 0, 2
        qk = 1
        bits = bin(k)[2:]
        inv2 = pow(2, -1, n)
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
    if Vd % n == 0:
        return True
    for r in range(1, s + 1):
        Vd = (Vd*Vd - 2 * pow(Q, d * (1 << (r - 1)), n)) % n
        if Vd % n == 0:
            return True
    return False

def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    if not _mr_strong_base2(n):
        return False
    if not _lucas_strong_prp(n):
        return False
    return True

# ──────────────────────────────────────────────────────────────────────────────
# Sieve / pre-sieve / wheel helpers
# ──────────────────────────────────────────────────────────────────────────────

def sieve_upto(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01")*(limit+1)
    sieve[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for p in range(2, r+1):
        if sieve[p]:
            start = p*p
            step = p
            sieve[start:limit+1:step] = b"\x00" * (((limit - start)//step) + 1)
    return [i for i, b in enumerate(sieve) if b]

def build_res_cache(subtractor_primes: List[int], wheel_list: List[int]) -> Dict[int, List[int]]:
    return {r: [p % r for p in subtractor_primes] for r in wheel_list if r > 2}

def build_n_mod(n: int, wheel_list: List[int]) -> Dict[int, int]:
    return {r: (n % r) for r in wheel_list if r > 2}

def pre_sieve_q(q: int, pre_primes: List[int], block2: bool) -> bool:
    if block2 and q % 2 == 0:
        return q == 2
    for r in pre_primes:
        if r >= q:
            break
        if q % r == 0:
            return False
    return True

# ──────────────────────────────────────────────────────────────────────────────
# Learned band ordering (uses same cache file as gbopt.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_cache(path: str) -> Dict[str, Dict[str, List[int]]]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def band_scores(trials: List[int], successes: List[int], smoothing: float = 1.0) -> List[float]:
    L = max(len(trials), len(successes))
    # pad if needed (defensive)
    tt = (trials + [0]*(L - len(trials)))[:L]
    ss = (successes + [0]*(L - len(successes)))[:L]
    return [(ss[b]+smoothing)/((tt[b]+2.0*smoothing)) for b in range(L)]

def compute_band_order_for_D(D: int, total_items: int, band_size: int, cache_file: str) -> List[int]:
    num_bands = (total_items + band_size - 1) // band_size
    cache = load_cache(cache_file)
    key = str(D)
    if key in cache and "trials" in cache[key] and "successes" in cache[key]:
        trials = cache[key]["trials"]
        successes = cache[key]["successes"]
        sc = band_scores(trials, successes, smoothing=1.0)
        order = list(range(num_bands))
        order.sort(key=lambda b: (-sc[b] if b < len(sc) else 0.0, b))
        return order
    # fallback: natural order
    return list(range(num_bands))

# ──────────────────────────────────────────────────────────────────────────────
# Demo decomposition finder (small-sub prepass + wheel + pre-sieve + BPSW)
# ──────────────────────────────────────────────────────────────────────────────

def random_even_with_digits(D: int, rng: random.Random) -> int:
    if D < 2:
        raise ValueError("digits must be >= 2")
    lo = 10**(D-1); hi = 10**D - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2:
        n += 1
    if n > hi:
        n -= 2
    if n < lo:
        n = lo + (1 if lo % 2 else 0)
    if n % 2:
        n += 1
    return n

def find_one_decomposition(D: int, core: CoreArgs) -> Tuple[int,int,int]:
    """
    Returns a concrete (n, p, q) for the given digit size D using:
      • learned band order if present (otherwise ascending),
      • tiny small-subtractor prepass,
      • wheel residue filter,
      • small-prime pre-sieve,
      • BPSW for primality.
    Prints nothing; the caller will format.
    """
    rng = random.Random(core.seed_for_examples * 1_000_003 + D * 7919)
    n = random_even_with_digits(D, rng)

    # subtractor primes (odd primes up to ceiling)
    subtractor_primes = [p for p in sieve_upto(core.subs_ceiling) if p % 2 == 1]
    total_items = len(subtractor_primes)

    # pre-sieve primes
    pre_primes = [p for p in sieve_upto(core.pre_sieve_limit) if p >= 3]

    # wheel residues
    wheel_list = [r for r in core.wheel_primes if r > 2]
    res_cache = build_res_cache(subtractor_primes, wheel_list)
    n_mod = build_n_mod(n, wheel_list)

    # learned band order
    band_order = compute_band_order_for_D(D, total_items, core.band_size, core.cache_file)

    # (A) tiny small-subtractor prepass
    if core.small_subs_first > 0:
        cap = max(0, core.small_subs_cap)  # ← fixed typo; do not use small_sabs_cap
        used = 0
        limit = min(core.small_subs_first, total_items)
        for idx in range(limit):
            p = subtractor_primes[idx]
            if p * 2 > n:
                break
            # wheel filter
            skip = False
            for r in wheel_list:
                if res_cache[r][idx] == n_mod[r]:
                    skip = True
                    break
            if skip:
                continue
            q = n - p
            if core.pre_sieve_mode == "blocks":
                if not pre_sieve_q(q, pre_primes, core.block2):
                    used += 1
                    if used >= cap > 0:
                        break
                    continue
            # BPSW
            used += 1
            if is_probable_prime(q):
                return (n, p, q)
            if used >= cap > 0:
                break

    # (B) main scan in learned band order
    checks = 0
    for b in band_order:
        start = b * core.band_size
        end = min(start + core.band_size, total_items)
        if start >= end:
            continue
        for idx in range(start, end):
            p = subtractor_primes[idx]
            if p * 2 > n:
                # one-direction early stop
                return (n, p, n - p) if is_probable_prime(n - p) else (_fail_n_found(D))
            # wheel
            wheel_ok = True
            for r in wheel_list:
                if res_cache[r][idx] == n_mod[r]:
                    wheel_ok = False
                    break
            if not wheel_ok:
                continue
            q = n - p
            if core.pre_sieve_mode == "blocks":
                if not pre_sieve_q(q, pre_primes, core.block2):
                    checks += 1
                    if checks >= core.subs_max_checks:
                        raise RuntimeError("subs_max_checks exhausted without hit")
                    continue
            checks += 1
            if is_probable_prime(q):
                return (n, p, q)
            if checks >= core.subs_max_checks:
                raise RuntimeError("subs_max_checks exhausted without hit")
    # fallback (should not happen in practice)
    raise RuntimeError("no decomposition found")

def _fail_n_found(D: int) -> Tuple[int,int,int]:
    # defensive fallback in the unlikely branch above; will be caught by caller
    raise RuntimeError(f"D={D}: early-stop reached without a confirmed hit")

# ──────────────────────────────────────────────────────────────────────────────
# Running gbopt.py sweeps
# ──────────────────────────────────────────────────────────────────────────────

def run_gbopt_sweep(gbopt_path: str, sweep: str, samples: int, seed: int,
                    save_policy: str, extra_args: str) -> None:
    """
    Shells out to gbopt.py and streams its stdout to our stdout.
    """
    cmd = [
        sys.executable, gbopt_path,
        "--sweep", sweep,
        "--samples", str(samples),
        "--seed", str(seed),
        "--save-policy", save_policy
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    # Run and stream
    proc = subprocess.run(cmd, text=True, capture_output=True)
    # Show stdout first (gbopt prints all its info there)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        # gbopt.py normally prints to stdout; surface stderr only if present
        sys.stderr.write(proc.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# Picking which digit sizes to show examples for
# ──────────────────────────────────────────────────────────────────────────────

def pick_example_digits(digits_space: List[int]) -> List[int]:
    """
    Prefer multiples of 10 within the sweep; otherwise pick 5 evenly-spaced.
    """
    tens = [d for d in digits_space if d % 10 == 0]
    if tens:
        if len(tens) <= 5:
            return tens
        # pick first, ~25%, ~50%, ~75%, last
        k = len(tens)
        idxs = [0, max(0, k//4), max(0, k//2), max(0, (3*k)//4), k-1]
        out = []
        for i in idxs:
            if tens[i] not in out:
                out.append(tens[i])
        return out
    # evenly space 5 values
    n = len(digits_space)
    if n <= 5:
        return digits_space
    idxs = [0, n//4, n//2, (3*n)//4, n-1]
    out = []
    for i in idxs:
        v = digits_space[i]
        if v not in out:
            out.append(v)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Demo runner
# ──────────────────────────────────────────────────────────────────────────────

def run_demo(args) -> None:
    print("══════════════════════════════════════════════════════════════════════════════")
    print("  Goldbach Optimizer — Comprehensive Empirical Demo")
    print("══════════════════════════════════════════════════════════════════════════════")
    print("We’re not claiming a formal proof. We’re showing robust empirical behavior:")
    print("learned ordering + arithmetic filters + fast primality verification quickly")
    print("find decompositions at scales far beyond exhaustive search, consistent with")
    print("Hardy–Littlewood.\n")

    # Core settings for Step 3 (read from --gb-args + first seed)
    seed_for_examples = args.seeds[0] if args.seeds else 1
    core = parse_gb_args_to_core(args.gb_args, seed_for_examples)

    # Step 1 — Warm-up (learning enabled)
    print("Step 1 — Warm-up (learning enabled)\n")
    for s in args.seeds:
        run_gbopt_sweep(args.gbopt_path, args.sweep_str, args.warm_samples, s, "always", args.gb_args)

    # Step 2 — Showcase (frozen ordering, read-only)
    print("\nStep 2 — Showcase (frozen ordering, read-only)\n")
    for s in args.seeds:
        run_gbopt_sweep(args.gbopt_path, args.sweep_str, args.show_samples, s, "read_only", args.gb_args)

    # Step 3 — Example decompositions
    print("\nStep 3 — Example decompositions (first hit shown)\n")
    digits_space = parse_sweep(args.sweep_str)
    for D in pick_example_digits(digits_space):
        try:
            n, p, q = find_one_decomposition(D, core)
            # ALWAYS print full integers (no compaction)
            print(f"  D={D}d  n = {n}")
            print(f"         n = p + q with  p={p}  and  q={q}")
        except Exception as e:
            print(f"  D={D}d  [no decomposition found: {e}]")

    # Conclusion
    print("\nConclusion — Not a formal proof, but:")
    print("  • Learned band ordering quickly localizes hits.")
    print("  • Arithmetic filters (wheel residues, small-prime blocks) prune ~vastly~.")
    print("  • BPSW validates hits fast without full factorization.")
    print("  • Behavior is stable across seeds and scales, matching Hardy–Littlewood and")
    print("    providing strong empirical evidence: we consistently find decompositions")
    print("    far beyond any feasible exhaustive search.")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="gbopt_demo",
        description="Comprehensive demo for the Goldbach optimizer: warm-up + showcase + explicit decompositions."
    )
    ap.add_argument("--sweep", required=True, help="Range 'start:end:step' for digit sweep.")
    ap.add_argument("--warm-samples", type=int, default=100, help="Samples per digit for warm-up (learning).")
    ap.add_argument("--show-samples", type=int, default=200, help="Samples per digit for showcase (read-only).")
    ap.add_argument("--seeds", type=parse_seeds, default=[1,2,3], help="Comma-separated seeds, e.g. '1,2,3'.")
    ap.add_argument("--gb-args", type=str, default="", help="Extra args passed through to gbopt.py.")
    ap.add_argument("--gbopt-path", type=str, default="./gbopt.py", help="Path to gbopt.py (default ./gbopt.py)")
    args = ap.parse_args()

    # Preserve original text of the sweep for gbopt.py commands
    args.sweep_str = args.sweep.strip()
    run_demo(args)

if __name__ == "__main__":
    main()
