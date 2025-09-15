#!/usr/bin/env python3
# gold.py — streaming, cross-digit parallel Goldbach sweeper
from __future__ import annotations
import argparse, math, os, random, sys, time, threading, queue
from dataclasses import dataclass
from typing import Tuple, Optional, List
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
    # Miller-Rabin
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
    n = n+1 if n%2==0 else n
    if GMPY2: return int(gmpy2.next_prime(n))
    while True:
        if is_probable_prime(n): return n
        n += 2

# ---------- decompose ----------
FASTLANE = (3,5,7,11,13,17)

@dataclass
class Decomp: p:int; q:int; attempts:int; via_fastlane:bool

def goldbach_decompose(n:int)->Decomp:
    attempts=0
    for q in FASTLANE:
        if q>=n: break
        p = n-q; attempts+=1
        if is_probable_prime(p): return Decomp(p,q,attempts,True)
    q = next_prime(max(FASTLANE))
    while q<n:
        p = n-q; attempts+=1
        if is_probable_prime(p): return Decomp(p,q,attempts,False)
        q = next_prime(q+2)
    raise RuntimeError("No decomposition found (unexpected for even n>2)")

# ---------- jobs ----------
@dataclass
class DigitJob: digits:int; count:int; seed:int

@dataclass
class DigitResult:
    digits:int; samples:int
    avg_ms_per_trial:float; avg_ms_per_attempt:float; avg_attempts:float
    last_decomp:Tuple[int,int,int]

def rand_even_with_digits(d:int, rng:random.Random)->int:
    lo, hi = 10**(d-1), 10**d-1
    n = rng.randrange(lo, hi+1)
    if n%2: n+=1
    if n>hi: n-=2
    return n

# worker payload wrapper so we can pass config + a reporting queue
def run_digit_worker(job:DigitJob, echo_each:bool, progress_every:int, q)->DigitResult:
    rng = random.Random(job.seed)
    total_ms=0.0; total_attempts=0
    last_p=last_q=last_n=0
    for i in range(1, job.count+1):
        n = rand_even_with_digits(job.digits, rng)
        t0=time.perf_counter()
        dec = goldbach_decompose(n)
        dt=(time.perf_counter()-t0)*1000.0
        total_ms += dt; total_attempts += dec.attempts
        last_p,last_q,last_n = dec.p, dec.q, n
        if echo_each:
            # stream every trial
            q.put(("trial", job.digits, n, dec.p, dec.q, dec.attempts, dt))
        elif progress_every>0 and (i%progress_every)==0:
            q.put(("tick", job.digits, i, job.count))
    avg_ms_trial = total_ms/job.count
    avg_ms_attempt = total_ms/max(1,total_attempts)
    avg_attempts = total_attempts/job.count
    # tell parent this digit finished (so it can print immediately)
    q.put(("done", job.digits, avg_ms_trial, avg_ms_attempt, avg_attempts, last_p, last_q, last_n))
    return DigitResult(job.digits, job.count, avg_ms_trial, avg_ms_attempt, avg_attempts, (last_p,last_q,last_n))

# ---------- helpers ----------
def parse_sweep(spec:str)->List[int]:
    a,b,s = map(int, spec.split(":"))
    if s==0: raise SystemExit("--sweep step must be non-zero.")
    out=[]; x=a
    if s>0:
        while x<=b: out.append(x); x+=s
    else:
        while x>=b: out.append(x); x+=s
    return out

def approx_physical_workers()->int:
    try:
        import psutil
        return max(1, psutil.cpu_count(logical=False) or os.cpu_count() or 1)
    except Exception:
        return max(1, os.cpu_count() or 1)

# ---------- main ----------
def main()->int:
    ap = argparse.ArgumentParser(description="Goldbach sweeper (streaming)")
    ap.add_argument("--sweep", required=True, help="start:end:step (e.g., 30:40:1)")
    ap.add_argument("--count", type=int, default=1000, help="samples per digit")
    ap.add_argument("--workers", type=int, help="processes across digits (default≈physical cores)")
    ap.add_argument("--seed", type=int, default=None, help="base RNG seed")
    ap.add_argument("--quiet", action="store_true", help="suppress per-trial echo; final decomp only")
    ap.add_argument("--echo-each", action="store_true", help="print every trial’s p + q = n (slow)")
    ap.add_argument("--progress-every", type=int, default=0, help="if >0 and not --echo-each, print ticks every N trials")
    args = ap.parse_args()

    digits = parse_sweep(args.sweep)
    base_seed = args.seed if args.seed is not None else random.randrange(2**63-1)
    jobs = [DigitJob(d, args.count, base_seed ^ (d*0x9E3779B185EBCA87)) for d in digits]

    workers = min(len(digits), args.workers or approx_physical_workers())

    # echo rules
    echo_each = bool(args.echo_each) and not args.quiet
    progress_every = 0 if echo_each or args.quiet else max(0, args.progress_every)

    from multiprocessing import Manager
    mgr = Manager()
    q = mgr.Queue()

    # simple consumer thread to print streaming messages
    stop_flag = {"stop": False}
    def consume():
        # header up-front
        hdr = "T Per digit (ms) / Avg T per n (ms) / Avg. Tests / p + q = n"
        print(hdr, flush=True)
        while not stop_flag["stop"]:
            try:
                msg = q.get(timeout=0.1)
            except queue.Empty:
                continue
            typ = msg[0]
            if typ == "trial":
                _, d, n, p, qv, attempts, dt = msg
                # per-trial echo (fast but still costly, user asked for it)
                print(f"[d={d}] {p} + {qv} = {n}", flush=True)
            elif typ == "tick":
                _, d, i, total = msg
                print(f"[d={d}] … {i}/{total}", flush=True)
            elif typ == "done":
                _, d, ms_trial, ms_attempt, avg_attempts, p, qv, n = msg
                # per-digit summary as soon as that digit completes
                print(f"[d={d}] {ms_trial:.3f} / {ms_attempt:.3f} / {avg_attempts:.2f} / {p} + {qv} = {n}", flush=True)
    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()

    t0=time.perf_counter()
    results=[]
    try:
        if workers>1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(run_digit_worker, jb, echo_each, progress_every, q) for jb in jobs]
                for fut in as_completed(futs):
                    results.append(fut.result())
        else:
            # single-process (still streams via queue)
            for jb in jobs:
                results.append(run_digit_worker(jb, echo_each, progress_every, q))
    except KeyboardInterrupt:
        print("\nInterrupted. Cancelling…", file=sys.stderr, flush=True)
        stop_flag["stop"]=True; consumer.join(timeout=1.0)
        return 130

    wall_ms=(time.perf_counter()-t0)*1000.0
    # for --quiet, print minimal per-digit lines (without decomp) and last decomp
    if args.quiet:
        results.sort(key=lambda r: r.digits)
        print("T Per digit (ms) / Avg T per n (ms) / Avg. Tests", flush=True)
        for r in results:
            print(f"[d={r.digits}] {r.avg_ms_per_trial:.3f} / {r.avg_ms_per_attempt:.3f} / {r.avg_attempts:.2f}", flush=True)
        last = results[-1].last_decomp
        print(f"Last decomp: {last[0]} + {last[1]} = {last[2]}", flush=True)

    stop_flag["stop"]=True
    consumer.join(timeout=1.0)

    print(f"[run] workers={workers} digits={digits[0]}..{digits[-1]} count={args.count} seed={base_seed} wall_ms={wall_ms:.1f}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
