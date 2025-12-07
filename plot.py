#!/usr/bin/env python3
# goldbach_unlucky_hunter_final.py — Multiple decompositions + real unlucky numbers
from __future__ import annotations
import argparse, random, sys, time, math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

try:
    import gmpy2
    GMPY2 = True
except Exception:
    GMPY2 = False

# Your exact honest primality
MR_BASES = (2,3,5,7,11,13,17,19,23,29)
def is_probable_prime(n: int) -> bool:
    if n < 2: return False
    for p in MR_BASES:
        if n == p: return True
        if n % p == 0: return False
    if GMPY2: return gmpy2.is_prime(n) != 0
    d, s = n-1, 0
    while d % 2 == 0: d//=2; s+=1
    for a in MR_BASES:
        if a >= n: break
        x = pow(a, d, n)
        if x in (1, n-1): continue
        for _ in range(s-1):
            x = (x*x) % n
            if x == n-1: break
        else: return False
    return True

def next_prime(n: int) -> int:
    n = n + (1 if n%2==0 else 2)
    if GMPY2: return int(gmpy2.next_prime(n-2 if n%2==0 else n))
    while True:
        if is_probable_prime(n): return n
        n += 2

# Fast random prime generation — works at any digit length
def random_prime(digits: int, rng: random.Random) -> int:
    lo = 10**(digits - 1)
    hi = 10**digits - 1
    while True:
        cand = rng.randrange(lo, hi)
        if cand % 2 == 0: cand += 1
        if is_probable_prime(cand):
            return cand

def unlucky_candidate(digits: int, rng: random.Random) -> int:
    p = random_prime(digits, rng)
    n = 2 * p
    while len(str(n)) > digits + 1:
        p = random_prime(digits, rng)
        n = 2 * p
    return n

def rand_even_digits(d: int, rng: random.Random) -> int:
    lo, hi = 10**(d-1), 10**d - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2: n += 1
    if n > hi: n -= 2
    return n

@dataclass
class GoldbachStats:
    digits: int
    attempts_to_first: float
    total_partitions: float
    worst_attempts: int
    unlucky_fraction: float

def analyze_number(n: int, max_attempts: int = 100_000) -> tuple[int, int]:
    attempts = 0
    partitions = 0
    first_found = False
    attempts_to_first = 0
    q = 3
    while q <= n // 2 and attempts < max_attempts:
        p = n - q
        attempts += 1
        if is_probable_prime(p):
            partitions += 1
            if not first_found:
                attempts_to_first = attempts
                first_found = True
        q = next_prime(q)
    return attempts_to_first or attempts, partitions

def worker(job):
    digits, count, seed, unlucky_ratio = job
    rng = random.Random(seed)
    unlucky_count = int(count * unlucky_ratio)
    normal_count = count - unlucky_count

    total_first = total_parts = 0.0
    worst = 0

    for _ in range(normal_count):
        n = rand_even_digits(digits, rng)
        first, parts = analyze_number(n)
        total_first += first
        total_parts += parts
        worst = max(worst, first)

    for _ in range(unlucky_count):
        n = unlucky_candidate(digits, rng)
        first, parts = analyze_number(n)
        total_first += first
        total_parts += parts
        worst = max(worst, first)

    return GoldbachStats(
        digits=digits,
        attempts_to_first=total_first / count,
        total_partitions=total_parts / count,
        worst_attempts=worst,
        unlucky_fraction=unlucky_ratio
    )

def main():
    p = argparse.ArgumentParser(description="Goldbach: Multiple decompositions + unlucky focus")
    p.add_argument("--sweep", default="100:3000:200")
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--unlucky", type=float, default=0.4, help="fraction of 2x large prime numbers")
    p.add_argument("--workers", type=int)
    args = p.parse_args()

    digits = [int(x) for x in range(*map(int, args.sweep.split(":")))]
    seed = random.randrange(1 << 60)
    jobs = [(d, args.count, seed ^ d, args.unlucky) for d in digits]
    workers = min(len(jobs), args.workers or os.cpu_count() or 1)

    results = []
    print("digits │ avg first │ avg parts │ worst │ unlucky%")
    with ProcessPoolExecutor(max_workers=workers) as exe:
        for stat in exe.map(worker, jobs):
            print(f"[d={stat.digits:4d}] {stat.attempts_to_first:8.1f}  {stat.total_partitions:7.1f}  {stat.worst_attempts:6d}  {stat.unlucky_fraction*100:5.1f}%")
            results.append(stat)

    # Live plot
    ds = [r.digits for r in results]
    firsts = [r.attempts_to_first for r in results]
    parts = [r.total_partitions for r in results]
    worsts = [r.worst_attempts for r in results]

    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(ds, firsts, 'o-', label="Avg attempts to first", color="red")
    plt.plot(ds, worsts, 's--', label="Worst case (unlucky)", color="darkred")
    plt.yscale('log')
    plt.legend()
    plt.title("Goldbach Unlucky Hunter — Attempts to First Partition")
    plt.subplot(2,1,2)
    plt.plot(ds, parts, 'o-', label="Average partitions found", color="green")
    plt.yscale('log')
    plt.legend()
    plt.title("Partition Richness")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
