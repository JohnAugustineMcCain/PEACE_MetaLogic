#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# subs_gb_opt.py — Optimized subtractor-only Goldbach finder
#
# - Deterministic ascending subtractors (p increasing), with options:
#   • Wheel residue prefilter (skip p that force q = n-p divisible by small primes)
#   • Pre-sieve q with small primes before BPSW
#   • Optional Thompson-sampling bandit over contiguous subtractor bands
#
# Outputs per-digit hit rate and avg_checks_to_hit; supports sweep, CI, CSV.
#
# (c) 2025 — MIT-style permissive for this prototype

from __future__ import annotations
import argparse, math, random, csv
from typing import List, Tuple, Optional, Dict

# ----------------------------- Small helpers -----------------------------

def _wilson_ci(h: int, n: int, z: float = 1.96) -> Tuple[float,float]:
    if n == 0: return (0.0, 1.0)
    p = h/n; denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (max(0.0, center-half), min(1.0, center+half))

def _ascii_bar(frac: float, width: int = 40) -> str:
    n = max(0, min(width, int(round(frac * width))))
    return "█"*n + "·"*(width-n)

def random_even_with_digits(D: int, rng: random.Random) -> int:
    if D < 2: raise ValueError("digits must be >= 2")
    lo = 10**(D-1); hi = 10**D - 1
    n = rng.randrange(lo, hi + 1)
    if n % 2: n += 1
    if n > hi: n -= 2
    if n < lo: n = lo + (1 if lo % 2 else 0)
    if n % 2: n += 1
    return n

def sieve_upto(limit: int) -> List[int]:
    if limit < 2: return []
    sieve = bytearray(b"\x01")*(limit+1)
    sieve[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for p in range(2, r+1):
        if sieve[p]:
            start = p*p; step = p
            sieve[start:limit+1:step] = b"\x00" * (((limit - start)//step) + 1)
    return [i for i, b in enumerate(sieve) if b]

def _median(vals: List[int]) -> Optional[int]:
    if not vals: return None
    s = sorted(vals); m = len(s)//2
    return s[m] if len(s)%2 else (s[m-1]+s[m])//2

# -------------------------- BPSW primality test --------------------------

_SMALL_PRIMES: List[int] = [2,3,5,7,11,13,17,19,23,29,31,37]

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

def is_probable_prime(n: int) -> bool:
    if n < 2: return False
    for p in _SMALL_PRIMES:
        if n == p: return True
        if n % p == 0: return False
    if not _mr_strong_base2(n): return False
    if not _lucas_strong_prp(n): return False
    return True

# ---------------------- Residue filter & pre-sieve -----------------------

def build_wheel_residues(n: int, wheel_primes: List[int]) -> Dict[int, int]:
    """
    Precompute n mod r for each small prime r in the 'wheel'.
    For a candidate subtractor p, if p % r == n % r, then q=n-p ≡ 0 (mod r), so reject p.
    """
    return {r: (n % r) for r in wheel_primes if r != 2}

def passes_wheel(p: int, n_mod: Dict[int, int]) -> bool:
    for r, nr in n_mod.items():
        if p % r == nr:
            return False
    return True

def q_pre_sieve(q: int, pre_primes: List[int]) -> bool:
    """
    Quick small-prime filter on q before BPSW.
    Accept if q is itself one of the pre_primes, or coprime to all of them.
    """
    for r in pre_primes:
        if r >= q: break
        if q % r == 0:
            return False
    return True

# -------------------------- Bandit over bands ----------------------------

class BanditCursor:
    """
    Thompson sampling over contiguous bands of indices in an ascending prime list.
    Within each band we advance sequentially, preserving deterministic order inside the band.
    """
    def __init__(self, total_items: int, band_size: int, rng: random.Random):
        self.band_size = max(1, band_size)
        self.num_bands = (total_items + self.band_size - 1) // self.band_size
        self.alpha = [1.0]*self.num_bands
        self.beta  = [1.0]*self.num_bands
        self.next_idx = [b*self.band_size for b in range(self.num_bands)]
        self.end_idx  = [min((b+1)*self.band_size, total_items) for b in range(self.num_bands)]
        self.rng = rng

    def has_items(self) -> bool:
        return any(self.next_idx[b] < self.end_idx[b] for b in range(self.num_bands))

    def _draw_band(self) -> Optional[int]:
        # draw only among bands with remaining items
        candidates = [b for b in range(self.num_bands) if self.next_idx[b] < self.end_idx[b]]
        if not candidates: return None
        thetas = []
        for b in candidates:
            thetas.append((b, self.rng.betavariate(self.alpha[b], self.beta[b])))
        thetas.sort(key=lambda x: x[1], reverse=True)
        return thetas[0][0]

    def next_item_index(self) -> Optional[int]:
        b = self._draw_band()
        if b is None: return None
        i = self.next_idx[b]
        if i >= self.end_idx[b]:
            return None
        self.next_idx[b] += 1
        # record which band we used by returning (i, b)
        return i

    def reward(self, idx: int, hit: bool):
        b = idx // self.band_size
        if hit:
            self.alpha[b] += 1.0
        else:
            self.beta[b]  += 1.0

# ------------------------- Core search procedure ------------------------

def search_subtractor_only(
    n: int,
    primes: List[int],
    *,
    max_checks: int,
    wheel_primes: List[int],
    pre_primes: List[int],
    use_bandit: bool,
    band_size: int,
    rng: random.Random
) -> Tuple[bool, int]:
    """
    Deterministic ascending subtractors, optionally guided by a bandit over bands.
    Returns (found, checks_used). 'checks' counts subtractor primes that pass the wheel filter.
    """
    if n % 2 or n < 4:
        return False, 0

    n_mod = build_wheel_residues(n, wheel_primes)
    checks = 0

    if use_bandit:
        cur = BanditCursor(len(primes), band_size, rng)
        while checks < max_checks and cur.has_items():
            idx = cur.next_item_index()
            if idx is None: break
            p = primes[idx]
            # wheel prefilter (cheap)
            if not passes_wheel(p, n_mod):
                # wheel rejects do not count as "checks"
                continue
            q = n - p
            if q <= 2:
                cur.reward(idx, False); continue
            # small-prime pre-sieve on q
            if not q_pre_sieve(q, pre_primes):
                checks += 1
                cur.reward(idx, False)
                continue
            # final BPSW on q
            if is_probable_prime(q):
                checks += 1
                cur.reward(idx, True)
                return True, checks
            checks += 1
            cur.reward(idx, False)
        return False, checks

    # Pure sequential (deterministic) over ascending primes
    for p in primes:
        if checks >= max_checks:
            break
        if not passes_wheel(p, n_mod):
            continue
        q = n - p
        if q <= 2:
            continue
        if not q_pre_sieve(q, pre_primes):
            checks += 1
            continue
        if is_probable_prime(q):
            checks += 1
            return True, checks
        checks += 1
    return False, checks

# ------------------------------- CLI bits --------------------------------

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

def rationale() -> str:
    return (
        "\n*** Subtractor-only Goldbach (optimized) ***\n\n"
        "Given even n, we search primes p (ascending) so that q=n-p is prime.\n"
        "Speedups:\n"
        "  • Wheel residue prefilter: skip p that force q divisible by small primes.\n"
        "  • Pre-sieve q by many small primes before running BPSW.\n"
        "  • Thompson-sampling over contiguous bands of subtractors (optional) to learn which band hits fastest for this n.\n"
        "All optimizations preserve correctness: any reported hit is validated by BPSW.\n"
    )

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="subs-gb-opt",
        description="Optimized subtractor-only Goldbach search with wheel/presieve/bandit."
    )
    ap.add_argument("--digits", type=int, help="Single digit size (mutually exclusive with --sweep).")
    ap.add_argument("--count", type=int, default=10, help="How many n for single-run (default 10).")
    ap.add_argument("--sweep", type=_parse_sweep, default=None, help="Range 'start:end:step' for digit sweep.")
    ap.add_argument("--samples", type=int, default=100, help="How many n per digit in sweep (default 100).")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed (default 2025).")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-n prints; show only summary rows.")
    ap.add_argument("--csv", type=str, default=None, help="Write sweep CSV summary.")
    ap.add_argument("--ci", action="store_true", help="Show Wilson 95% CI for hit rate.")
    ap.add_argument("--why", action="store_true", help="Print a short rationale and exit.")

    # Subtractor config
    ap.add_argument("--subs-ceiling", type=int, default=300000,
                    help="Upper bound for subtractor primes (default 300000).")
    ap.add_argument("--subs-max-checks", type=int, default=2000,
                    help="Max subtractor primes to examine per n (after wheel filtering) (default 2000).")

    # Optimizations
    ap.add_argument("--wheel-primes", type=str, default="3,5,7,11,13",
                    help="Comma-separated small primes for residue wheel (default '3,5,7,11,13').")
    ap.add_argument("--pre-sieve-limit", type=int, default=20000,
                    help="Pre-sieve q with primes up to this value before BPSW (default 20000).")
    ap.add_argument("--bandit", action="store_true",
                    help="Enable Thompson-sampling over bands of ascending subtractors.")
    ap.add_argument("--band-size", type=int, default=64,
                    help="Contiguous band size for the bandit (default 64).")

    args = ap.parse_args()
    if args.why:
        print(rationale()); return

    rng = random.Random(args.seed)

    # Prepare subtractor primes (ascending, deterministic)
    subtractor_primes = [p for p in sieve_upto(args.subs_ceiling) if p % 2 == 1 and p != 1]
    # Prepare pre-sieve primes for q (exclude 2; keep odd primes)
    pre_primes = [p for p in sieve_upto(args.pre_sieve_limit) if p >= 3]

    # Parse wheel list
    wheel_primes = []
    for tok in args.wheel_primes.split(","):
        v = int(tok.strip())
        if v <= 2: continue
        wheel_primes.append(v)

    def run_for_digits(D: int, trials: int) -> Tuple[int,int,List[int]]:
        hits = 0
        checks_on_hits: List[int] = []
        for _ in range(trials):
            n = random_even_with_digits(D, rng)
            found, checks = search_subtractor_only(
                n,
                subtractor_primes,
                max_checks=args.subs_max_checks,
                wheel_primes=wheel_primes,
                pre_primes=pre_primes,
                use_bandit=args.bandit,
                band_size=max(1, args.band_size),
                rng=rng
            )
            if found:
                hits += 1
                checks_on_hits.append(checks)
        return hits, trials, checks_on_hits

    # Sweep mode
    if args.sweep is not None:
        digits_space = args.sweep
        rows: List[Dict[str, object]] = []
        if not args.quiet:
            print("\n# Optimized subtractor-only sweep")
            print(f"# subs_ceiling={args.subs_ceiling} | subs_max_checks={args.subs_max_checks} "
                  f"| bandit={'on' if args.bandit else 'off'} (band_size={args.band_size})")
            print(f"# wheel={wheel_primes} | pre_sieve_limit={args.pre_sieve_limit}")
            print(f"# samples per digit={args.samples} | seed={args.seed}\n")

        for D in digits_space:
            hits, total, checks_on_hits = run_for_digits(D, args.samples)
            rate = hits/total if total else 0.0
            avg_checks = (sum(checks_on_hits)/len(checks_on_hits)) if checks_on_hits else None
            lo95, hi95 = _wilson_ci(hits, total) if args.ci else (None, None)
            bar = _ascii_bar(rate)

            ci_txt = f"  CI95=[{lo95:.3f},{hi95:.3f}]" if args.ci else ""
            print(f"{D:>3}d  hit_rate={rate:6.3f}{ci_txt}  avg_checks_to_hit={('%.1f'%avg_checks) if avg_checks is not None else '—':>6}  {bar}")

            row = {
                "digits": D,
                "samples": total,
                "hits": hits,
                "misses": total - hits,
                "hit_rate": round(rate, 6),
                "avg_checks_to_hit": round(avg_checks, 3) if avg_checks is not None else None,
                "subs_ceiling": args.subs_ceiling,
                "subs_max_checks": args.subs_max_checks,
                "bandit": args.bandit,
                "band_size": args.band_size,
                "wheel_primes": ",".join(map(str, wheel_primes)),
                "pre_sieve_limit": args.pre_sieve_limit,
                "seed": args.seed,
            }
            if args.ci:
                row["hit_rate_ci95_low"] = round(lo95, 6)
                row["hit_rate_ci95_high"] = round(hi95, 6)
            rows.append(row)

        if args.csv and rows:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader(); writer.writerows(rows)
            print(f"[saved] CSV -> {args.csv}")
        return

    # Single-run mode
    if args.digits is None:
        ap.error("Provide --digits for single-run OR --sweep for a range. Use --why for the rationale.")

    D = args.digits
    print(f"# Subtractor-only (optimized) | D={D} | count={args.count} | seed={args.seed}")
    print(f"# subs_ceiling={args.subs_ceiling} | subs_max_checks={args.subs_max_checks} "
          f"| bandit={'on' if args.bandit else 'off'} (band_size={args.band_size}) | wheel={wheel_primes} | pre_sieve_limit={args.pre_sieve_limit}")
    hits, total, checks_on_hits = run_for_digits(D, args.count)
    rate = hits/total if total else 0.0
    avg_checks = (sum(checks_on_hits)/len(checks_on_hits)) if checks_on_hits else None
    print(f"hit_rate={rate:.3f}  avg_checks_to_hit={avg_checks:.1f}" if avg_checks is not None else f"hit_rate={rate:.3f}  avg_checks_to_hit=—")


if __name__ == "__main__":
    main()
