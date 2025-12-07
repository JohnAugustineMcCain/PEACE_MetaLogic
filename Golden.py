#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, random, sys, time, threading, queue
from dataclasses import dataclass
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor

try:
    import gmpy2
    GMPY2 = True
except Exception:
    GMPY2 = False

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

LEARNING_DB: Dict[int, float] = {}

def goldbach_decompose(n: int, digits: int) -> tuple[int, int, int]:
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
            return min(p,q), max(p,q), attempts
        q = next_prime(q)
    raise RuntimeError("No partition")

def rand_even_with_digits(d: int, rng: random.Random) -> int:
    lo, hi = 10**(d-1), 10**d - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2: n += 1
    if n > hi: n -= 2
    return n

@dataclass
class DigitJob: digits: int; count: int; seed: int

def run_digit_worker(job: DigitJob, echo_each: bool, progress_every: int, q) -> None:
    rng = random.Random(job.seed)
    total_ms = total_attempts = 0.0
    last_p = last_q = last_n = 0
    for i in range(1, job.count + 1):
        n = rand_even_with_digits(job.digits, rng)
        t0 = time.perf_counter()
        p, q, attempts = goldbach_decompose(n, job.digits)
        dt = (time.perf_counter() - t0) * 1000
        total_ms += dt
        total_attempts += attempts
        last_p, last_q, last_n = p, q, n
        if echo_each:
            q.put(("trial", job.digits, attempts, dt))
        elif progress_every and i % progress_every == 0:
            q.put(("tick", job.digits, i, job.count))
    avg_ms = total_ms / job.count
    avg_attempts = total_attempts / job.count
    q.put(("done", job.digits, avg_ms, avg_attempts, last_p, last_q, last_n))

def parse_sweep(s: str) -> List[int]:
    a, b, step = map(int, s.split(":"))
    return list(range(a, b + 1, step)) if step > 0 else list(range(a, b - 1, step))

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--echo-each", action="store_true")
    p.add_argument("--progress-every", type=int, default=100)
    args = p.parse_args()

    digits = parse_sweep(args.sweep)
    seed = args.seed if args.seed is not None else random.randrange(1 << 60)
    jobs = [DigitJob(d, args.count, seed ^ (d << 13)) for d in digits]
    workers = min(len(digits), args.workers or os.cpu_count() or 1)
    echo_each = args.echo_each and not args.quiet
    prog = 0 if echo_each or args.quiet else args.progress_every

    q = queue.Queue()
    stop = threading.Event()

    def printer():
        if not args.quiet:
            print("digits │ ms/trial │ avg attempts")
        while not stop.is_set():
            try:
                msg = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if msg[0] == "tick" and not args.quiet:
                print(f"[d={msg[1]:4d}] {msg[2]}/{msg[3]}", flush=True)
            elif msg[0] == "done" and not args.quiet:
                print(f"[d={msg[1]:4d}] {msg[2]:7.2f}  {msg[3]:8.2f}", flush=True)

    threading.Thread(target=printer, daemon=True).start()

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(run_digit_worker, job, echo_each, prog, q) for job in jobs]
        for f in futures:
            f.result()
    stop.set()

    print(f"\nDone. Sweep: {digits[0]}–{digits[-1]} digits, {args.count} samples each, {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
