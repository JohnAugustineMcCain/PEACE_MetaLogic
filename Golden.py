#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, random, sys, time
from dataclasses import dataclass
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import gmpy2
    GMPY2 = True
except Exception:
    GMPY2 = False

LEARNING_DB: Dict[int, float] = {}

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

def goldbach_decompose(n: int, digits: int) -> int:
    attempts = 0
    start_q = 3
    if digits in LEARNING_DB:
        start_q = max(3, int(LEARNING_DB[digits] * 0.75))
    q = next_prime(start_q - 1 if start_q > 3 else 2)
    while q <= n // 2:
        p = n - q
        attempts += 1
        if is_probable_prime(p):
            new_avg = (LEARNING_DB.get(digits, attempts) * 0.9) + (attempts * 0.1)
            LEARNING_DB[digits] = new_avg
            return attempts
        q = next_prime(q)
    raise RuntimeError("No partition")

def rand_even_with_digits(d: int, rng: random.Random) -> int:
    lo, hi = 10**(d-1), 10**d - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2: n += 1
    if n > hi: n -= 2
    return n

@dataclass(frozen=True)
class Job:
    digits: int
    count: int
    seed: int

def worker(job: Job) -> tuple[int, float, float]:
    rng = random.Random(job.seed)
    total_ms = total_attempts = 0.0
    for _ in range(job.count):
        n = rand_even_with_digits(job.digits, rng)
        t0 = time.perf_counter()
        attempts = goldbach_decompose(n, job.digits)
        total_ms += (time.perf_counter() - t0) * 1000
        total_attempts += attempts
    return job.digits, total_ms / job.count, total_attempts / job.count

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    start, end, step = map(int, args.sweep.split(":"))
    digits = list(range(start, end + 1, step))
    seed = args.seed if args.seed is not None else random.randrange(1 << 60)
    jobs = [Job(d, args.count, seed ^ d) for d in digits]
    workers = min(len(jobs), args.workers or os.cpu_count() or 1)

    print("digits │ ms/trial │ avg attempts")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as exe:
        for d, ms, attempts in exe.map(worker, jobs):
            print(f"[d={d:4d}] {ms:7.2f}  {attempts:8.2f}")
    print(f"Done in {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
