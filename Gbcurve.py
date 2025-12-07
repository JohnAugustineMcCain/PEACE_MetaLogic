#!/usr/bin/env python3
# gold.py — streaming, cross-digit parallel Goldbach sweeper + density plotter
from __future__ import annotations
import argparse, math, os, random, sys, time, threading, queue
from dataclasses import dataclass
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import gmpy2
    GMPY2 = True
except Exception:
    GMPY2 = False

# ---------- primality ----------
MR_BASES = (2,3,5,7,11,13,17,19,23,29)
def is_probable_prime(n: int) -> bool:
    if n < 2: return False
    for p in MR_BASES:
        if n == p: return True
        if n % p == 0: return False
    if GMPY2:
        return gmpy2.is_prime(n) != 0
    d, s = n-1, 0
    while d % 2 == 0: d//=2; s+=1
    for a in MR_BASES:
        if a % n == 0: continue
        x = pow(a, d, n)
        if x in (1, n-1): continue
        for _ in range(s-1):
            x = (x*x) % n
            if x == n-1: break
        else: return False
    return True

def next_prime(n: int) -> int:
    if n <= 2: return 2
    n = n+1 if n%2==0 else n+2
    if GMPY2: return int(gmpy2.next_prime(n))
    while True:
        n += 2
        if is_probable_prime(n): return n

# ---------- decompose ----------
FASTLANE = (3,5,7,11,13,17)

@dataclass
class Decomp:
    p: int
    q: int
    attempts: int
    via_fastlane: bool

def goldbach_decompose(n: int) -> Decomp:
    if n < 4 or n % 2 != 0:
        raise ValueError("n must be even and >= 4")
    if n == 4: return Decomp(2, 2, 1, False)
    attempts = 0
    for q in FASTLANE:
        if q > n // 2: break
        p = n - q
        attempts += 1
        if is_probable_prime(p):
            return Decomp(min(p,q), max(p,q), attempts, True)
    q = next_prime(max(FASTLANE))
    while q <= n // 2:
        p = n - q
        attempts += 1
        if is_probable_prime(p):
            return Decomp(min(p,q), max(p,q), attempts, False)
        q = next_prime(q)
    raise RuntimeError(f"No Goldbach partition found for {n}")

# ---------- jobs ----------
@dataclass
class DigitJob:
    digits: int
    count: int
    seed: int

@dataclass
class DigitResult:
    digits: int
    samples: int
    avg_ms_per_trial: float
    avg_attempts: float
    max_attempts: int
    last_decomp: Tuple[int,int,int]

def rand_even_with_digits(d: int, rng: random.Random) -> int:
    lo, hi = 10**(d-1), 10**d - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2: n += 1
    if n > hi: n -= 2
    return n

def run_digit_worker(job: DigitJob, echo_each: bool, progress_every: int, q) -> DigitResult:
    rng = random.Random(job.seed)
    total_ms = 0.0
    total_attempts = 0
    max_attempts = 0
    last_p = last_q = last_n = 0
    for i in range(1, job.count + 1):
        n = rand_even_with_digits(job.digits, rng)
        t0 = time.perf_counter()
        dec = goldbach_decompose(n)
        dt = (time.perf_counter() - t0) * 1000.0
        total_ms += dt
        total_attempts += dec.attempts
        max_attempts = max(max_attempts, dec.attempts)
        last_p, last_q, last_n = dec.p, dec.q, n
        if echo_each:
            q.put(("trial", job.digits, n, dec.p, dec.q, dec.attempts, dt))
        elif progress_every > 0 and i % progress_every == 0:
            q.put(("tick", job.digits, i, job.count))
    avg_ms_trial = total_ms / job.count
    avg_attempts = total_attempts / job.count
    q.put(("done", job.digits, avg_ms_trial, avg_attempts, max_attempts, last_p, last_q, last_n))
    return DigitResult(job.digits, job.count, avg_ms_trial, avg_attempts, max_attempts, (last_p, last_q, last_n))

# ---------- helpers ----------
def parse_sweep(spec: str) -> List[int]:
    a, b, s = map(int, spec.split(":"))
    if s == 0: raise SystemExit("--sweep step must be non-zero.")
    return list(range(a, b + 1, s)) if s > 0 else list(range(a, b - 1, s))

def approx_physical_workers() -> int:
    try:
        import psutil
        return max(1, psutil.cpu_count(logical=False) or os.cpu_count() or 1)
    except Exception:
        return max(1, os.cpu_count() or 1)

# ---------- plotting ----------
def plot_density(results: List[DigitResult], save_csv: bool = False):
    import matplotlib.pyplot as plt
    results.sort(key=lambda r: r.digits)
    digits = [r.digits for r in results]
    avg = [r.avg_attempts for r in results]
    maxes = [r.max_attempts for r in results]

    plt.figure(figsize=(12, 7))
    plt.plot(digits, avg, 'o-', color='#1f77b4', linewidth=3, label='Average attempts')
    plt.plot(digits, maxes, 's--', color='#ff7f0e', alpha=0.7, label='Maximum attempts (worst case)')
    plt.title("Goldbach Decomposition Density\n(lower = more partitions = stronger conjecture)", fontsize=16)
    plt.xlabel("Number of digits", fontsize=14)
    plt.ylabel("Average attempts to find one partition", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("goldbach_density.png", dpi=200)
    plt.show()

    if save_csv:
        import csv
        with open("goldbach_density.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["digits", "samples", "avg_attempts", "max_attempts"])
            for r in results:
                w.writerow([r.digits, r.samples, r.avg_attempts, r.max_attempts])
        print("Data saved to goldbach_density.csv")

# ---------- main ----------
def main() -> int:
    ap = argparse.ArgumentParser(description="Goldbach sweeper + density plotter")
    ap.add_argument("--sweep", required=True, help="e.g. 50:200:10")
    ap.add_argument("--count", type=int, default=1000, help="samples per digit length")
    ap.add_argument("--workers", type=int, help="default = physical cores")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--echo-each", action="store_true")
    ap.add_argument("--progress-every", type=int, default=0)
    ap.add_argument("--plot", action="store_true", help="Show density curve when finished")
    ap.add_argument("--save-csv", action="store_true", help="Export raw data")
    args = ap.parse_args()

    digits = parse_sweep(args.sweep)
    base_seed = args.seed if args.seed is not None else random.randrange(2**63-1)
    jobs = [DigitJob(d, args.count, base_seed ^ (d*0x9E3779B185EBCA87)) for d in digits]
    workers = min(len(digits), args.workers or approx_physical_workers())
    echo_each = args.echo_each and not args.quiet
    progress_every = 0 if echo_each or args.quiet else max(0, args.progress_every)

    from multiprocessing import Manager
    mgr = Manager()
    q = mgr.Queue()

    stop_flag = {"stop": False}
    def consume():
        hdr = "T Per digit (ms) / Avg attempts / Max attempts / p + q = n"
        print(hdr, flush=True)
        while not stop_flag["stop"]:
            try:
                msg = q.get(timeout=0.1)
            except queue.Empty:
                continue
            typ = msg[0]
            if typ == "trial":
                _, d, n, p, qv, attempts, dt = msg
                print(f"[d={d}] {p} + {qv} = {n} (attempts: {attempts})", flush=True)
            elif typ == "tick":
                _, d, i, total = msg
                print(f"[d={d}] … {i}/{total}", flush=True)
            elif typ == "done":
                _, d, ms_trial, avg_attempts, max_attempts, p, qv, n = msg
                print(f"[d={d}] {ms_trial:6.2f} ms/trial │ {avg_attempts:5.3f} avg │ {max_attempts:2d} max │ {p} + {qv} = {n}", flush=True)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()

    t0 = time.perf_counter()
    results = []
    try:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(run_digit_worker, jb, echo_each, progress_every, q) for jb in jobs]
                for fut in as_completed(futs):
                    results.append(fut.result())
        else:
            for jb in jobs:
                results.append(run_digit_worker(jb, echo_each, progress_every, q))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        stop_flag["stop"] = True
        consumer.join(timeout=1.0)
        return 130

    wall_ms = (time.perf_counter() - t0) * 1000.0
    stop_flag["stop"] = True
    consumer.join(timeout=1.0)

    print(f"\n[run completed] digits={digits[0]}..{digits[-1]}  samples={args.count}  workers={workers}  wall={wall_ms/1000:.1f}s")
    
    if args.plot:
        plot_density(results, save_csv=args.save_csv)

    return 0

if __name__ == "__main__":
    sys.exit(main())
