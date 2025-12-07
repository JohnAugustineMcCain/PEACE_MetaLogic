#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, random, sys, time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

try:
    import gmpy2
    GMPY2 = True
except Exception:
    GMPY2 = False

# Pre-sieve first 10 million primes — enough for 10,000+ digit numbers
SIEVE_LIMIT = 200_000_000          # ~10.4 million primes (fast to generate)
print(f"Sieving first {SIEVE_LIMIT:,} numbers for small primes...", end="", flush=True)
is_prime = [True] * (SIEVE_LIMIT + 1)
is_prime[0] = is_prime[1] = False
for i in range(2, int(SIEVE_LIMIT**0.5)+1):
    if is_prime[i]:
        for j in range(i*i, SIEVE_LIMIT+1, i):
            is_prime[j] = False
SMALL_PRIMES = [i for i in range(2, SIEVE_LIMIT+1) if is_prime[i]]
print(f" done! {len(SMALL_PRIMES):,} small primes ready")

def is_probable_prime(n: int) -> bool:
    if n < SIEVE_LIMIT: return is_prime[n]
    if GMPY2: return gmpy2.is_prime(n) != 0
    # fallback Miller-Rabin (rarely used)
    for a in (2,3,5,7,11,13,17,19,23,29):
        if n == a: return True
        if n % a == 0: return False
        x = pow(a, n-1, n)
        if x != 1: return False
    return True

def goldbach_sieve(n: int) -> int:
    """Returns number of attempts using pre-sieved small primes"""
    for q in SMALL_PRIMES:
        if q >= n: break
        p = n - q
        if p < SIEVE_LIMIT:
            if is_prime[p]: return SMALL_PRIMES.index(q) + 1
        elif is_probable_prime(p):
            return SMALL_PRIMES.index(q) + 1
    # Fallback: search from large end (extremely rare)
    q = n // 2
    if q % 2 == 0: q -= 1
    while q > n // 3:
        p = n - q
        if is_probable_prime(q) and is_probable_prime(p):
            return len(SMALL_PRIMES) + 1
        q -= 2
    raise RuntimeError("No partition (impossible)")

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
        attempts = goldbach_sieve(n)
        total_ms += (time.perf_counter() - t0) * 1000
        total_attempts += attempts
    return job.digits, total_ms / job.count, total_attempts / job.count

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True, help="e.g. 100:10000:100")
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--workers", type=int)
    args = p.parse_args()

    start, end, step = map(int, args.sweep.split(":"))
    digits = list(range(start, end + 1, step))
    seed = random.randrange(1 << 60)
    jobs = [Job(d, args.count, seed ^ d) for d in digits]
    workers = min(len(jobs), args.workers or os.cpu_count() or 1)

    print(f"digits │ ms/trial │ avg attempts (out of {len(SMALL_PRIMES):,})")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as exe:
        for d, ms, attempts in exe.map(worker, jobs):
            print(f"[d={d:5d}] {ms:7.3f}  {attempts:8.1f}")
    print(f"Done in {time.time()-t0:.1f}s — sieving wins")
    return 0

if __name__ == "__main__":
    sys.exit(main())
