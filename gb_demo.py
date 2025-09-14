#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gb_demo.py — tiny runner for gb.py + short explainer
#
# Usage examples:
#   python3 gb_demo.py --gb-path ./gb.py --digits 480:500 --count 10 --seeds 1,2 \
#       --gb-args "--percent 0.5:1.2 --quiet"
#
#   python3 gb_demo.py --gb-path ./gb.py --digits 150:160 --count 25 \
#       --gb-args "--percent 1:2"
#
# This script:
#   1) shells out to gb.py with your chosen args (optionally for several seeds),
#   2) then prints a concise explanation of why gb.py’s design is useful.

from __future__ import annotations
import argparse
import shlex
import subprocess
import sys

def parse_seeds(s: str) -> list[int]:
    return [int(t.strip()) for t in s.split(",") if t.strip()]

def run_gb(gb_path: str, digits: str, count: int, seed: int|None, extra_args: str) -> int:
    cmd = [sys.executable, gb_path, "--digits", digits, "--count", str(count)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if extra_args:
        cmd += shlex.split(extra_args)
    print("\n────────────────────────────────────────────────────────────────────────────")
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    print("────────────────────────────────────────────────────────────────────────────\n")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        # surface errors from gb.py if any
        sys.stderr.write(proc.stderr)
    return proc.returncode

EXPLANATION = r"""
Why this code matters (the short version)

• Uniform random n per digit band:
  Each band starts from a truly uniform even d-digit n, so your sampling isn’t
  biased toward edges or special shapes. That keeps the stats meaningful.

• Random walk scaled by % of n:
  Steps of ~[A,B]% (with integer-safe stochastic rounding) let you scan the band
  smoothly without ever converting huge integers to floats.

• Guaranteed-hit trials:
  For every n the search continues until it finds a Goldbach pair p+q=n, so you
  never record a “give up” and your per-digit averages stay interpretable.

• Three “racers”, turn-taking by BPSW:
  (1) sequential small subtractors, (2) around the current avg-K hotspot, and
  (3) mid search near n/2. They take turns per expensive check, with a slight
  bias toward whichever is currently winning—so the compute budget flows to the
  most promising tactic at that digit scale.

• Wheel/bitset pre-sieve (baked in, no flags):
  Before any real test, candidates that would make q divisible by tiny primes
  (e.g., 3,5,7,11,13) are discarded in O(1) by residue masks. This usually
  halves or thirds the heavy work without ever discarding a true prime q.

• Lean primality pipeline:
  Small trial division → strong base-2/3 PRP → BPSW (via gmpy2). This validates
  primes extremely fast, scales to hundreds of digits, and doesn’t require
  factorization.

• Self-tuning:
  The code keeps a rolling window and adjusts trial-division depth to keep the
  fraction of “needs BPSW” in a sweet spot—minimizing overall time/trial.

The upshot: you get fast, reproducible, per-digit empirical data that accords
with the prime number theorem / Hardy–Littlewood intuition (≈O(log n) viable
candidates), while the baked-in arithmetic filters and turn-taking keep costs
predictable even at 300–800+ digits.
"""

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="gb_demo",
        description="Minimal demo: run gb.py with your args, then print a short explainer."
    )
    ap.add_argument("--gb-path", default="./gb.py", help="Path to gb.py (default ./gb.py)")
    ap.add_argument("--digits", required=True, help="Digit range for gb.py, e.g. '150:200'")
    ap.add_argument("--count", type=int, default=100, help="Samples per digit for gb.py (default 100)")
    ap.add_argument("--seeds", type=parse_seeds, default=[1], help="Comma-separated seeds (default: 1)")
    ap.add_argument("--gb-args", default="", help="Extra args passed straight through to gb.py")
    args = ap.parse_args()

    ret = 0
    for s in args.seeds:
        rc = run_gb(args.gb_path, args.digits, args.count, s, args.gb_args)
        ret = ret or rc

    print("\n\n============================== EXPLAINER ==============================")
    print(EXPLANATION.strip())
    print("======================================================================\n")

    sys.exit(ret)

if __name__ == "__main__":
    main()
