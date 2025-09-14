#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Goldbach sampler with gmpy2 (guaranteed-hit trials, built-in wheel/bitset pre-sieve):
- First n per digit band is a UNIFORM RANDOM even d-digit number
- Random increasing even n via percent step (with stochastic rounding)
- Unbounded subtractor stream: 3,5,7,... generated on demand
- Three competing racers per n:
    seq   : sequential over subtractors (3,5,7,...) (infinite)
    around: centered around running average attempts K (infinite)
    mid   : balanced search around n/2 (checks q then p)
- Per-BPSW turn-taking (leader gets 2 checks per turn, others 1)
- Self-tuning: auto trial-division depth + strong PRP(2,3) before BPSW
- Wheel/bitset pre-sieve:
    * p-side: forbid residues p ≡ n (mod r) for r ∈ {3,5,7,11,13}, combined via W=15015
    * q-side: only allow residues coprime to W (skip q divisible by wheel primes)
- Per-digit summaries by default; --quiet hides the representative p+q=n
- Representative decomposition picked by weighted-random (avoids “ugly” round/edge cases)

Usage examples:
  python gb.py --digits 480:500 --percent 0.5:1.2 --seed 2 --count 5
  python gb.py --digits 150:180 --count 200 --percent 1:2 --seed 42 --quiet
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass
from collections import deque

try:
    import gmpy2
except ImportError:
    print("ERROR: gmpy2 is required. Install with: pip install gmpy2", file=sys.stderr)
    sys.exit(1)


# ---------------------------- Helpers ---------------------------------- #

def parse_range_pair(s: str, name: str, as_int: bool = True):
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"--{name} must be in the form A:B")
    a, b = parts[0].strip(), parts[1].strip()
    try:
        if as_int:
            A, B = int(a), int(b)
        else:
            A, B = float(a), float(b)
    except Exception:
        raise ValueError(f"--{name} must contain {'integers' if as_int else 'numbers'}")
    if A <= 0 or B <= 0:
        raise ValueError(f"--{name} must be positive values")
    if A > B:
        raise ValueError(f"--{name} lower bound must be ≤ upper bound")
    return A, B


def parse_percent_pair_intscaled(s: str):
    """Parse 'A:B' as exact scaled integers (no floats). Interpreted as r/(100*10^k)."""
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("--percent must be in the form A:B")
    a, b = parts[0].strip(), parts[1].strip()

    def to_scaled(x: str):
        if not x or any(c for c in x if c not in "0123456789."):
            raise ValueError("--percent values must be non-negative plain decimals")
        if x.count(".") > 1:
            raise ValueError("--percent values must be valid decimals")
        if "." in x:
            whole, frac = x.split(".")
            whole = whole or "0"
            dp = len(frac)
            digits = (whole + frac) if (whole + frac) else "0"
            return int(digits), dp
        else:
            return int(x), 0

    ai, adp = to_scaled(a)
    bi, bdp = to_scaled(b)
    k = max(adp, bdp)

    A_scaled = ai * (10 ** (k - adp))
    B_scaled = bi * (10 ** (k - bdp))

    if A_scaled <= 0 or B_scaled <= 0:
        raise ValueError("--percent values must be > 0")
    if A_scaled > B_scaled:
        raise ValueError("--percent lower bound must be ≤ upper bound")
    return A_scaled, B_scaled, k


def first_even_with_digits(d: int) -> int:
    if d == 1:
        return 4
    lo = 10 ** (d - 1)
    if lo % 2:
        lo += 1
    return lo


def max_even_with_digits(d: int) -> int:
    hi = (10 ** d) - 1
    if hi % 2:
        hi -= 1
    return max(4, hi)


def gmpy2_version_str():
    v = getattr(gmpy2, "__version__", None)
    if v:
        return v
    vfun = getattr(gmpy2, "version", None)
    try:
        return vfun() if callable(vfun) else "unknown"
    except Exception:
        return "unknown"


# ---------------------------- Data ------------------------------------- #

@dataclass
class Config:
    digits_lo: int
    digits_hi: int
    count_per_digits: int
    percent_num_lo: int
    percent_num_hi: int
    percent_scale: int
    percent_display: str
    seed: int | None
    quiet: bool


@dataclass
class Metrics:
    ms_elapsed: float
    subs_tried: int       # attempts after pre-sieve (across racers)
    bpsw_checks: int      # counted but not printed (kept for autotune)
    success: bool
    n: int | None
    p: int | None
    q: int | None
    method_winner: str | None


# -------------------------- Core Sampler -------------------------------- #

class GoldbachSampler:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        # Unbounded subtractor stream: store odd primes 3,5,7,... and grow on demand
        self.sub_primes: list[int] = []
        self._last_prime = gmpy2.mpz(2)  # so next_prime gives 3

        # Self-tuning small trial division primes
        self.trial_prs: list[int] = []     # includes 2
        self.trial_bound = 64              # auto-tuned
        self.trial_bound_min = 32
        self.trial_bound_max = 1024
        self._ensure_trial_primes(self.trial_bound)

        # Rolling window for autotune
        self.window = deque(maxlen=64)  # (subs_tried, bpsw_checks, ms)

        # -------- Wheel / bitset pre-sieve --------
        # Use primes {3,5,7,11,13}; W = 15015
        self.wheel_primes = (3, 5, 7, 11, 13)
        W = 1
        for r in self.wheel_primes:
            W *= r
        self.W = W

        # q-side: residues allowed iff coprime to W (i.e., not 0 mod any wheel prime)
        # Build once; tiny cost.
        allowed = bytearray(1 for _ in range(self.W))
        for r in self.wheel_primes:
            for t in range(0, self.W, r):
                allowed[t] = 0  # multiples of r are forbidden
        self.allowed_q_mask = allowed  # 1 = allowed, 0 = forbidden

    # ---- subtractor generator (unbounded) ----
    def _ensure_sub_primes(self, need_idx: int):
        """Ensure we have at least `need_idx` odd primes in self.sub_primes."""
        while len(self.sub_primes) < need_idx:
            self._last_prime = gmpy2.next_prime(self._last_prime)
            p = int(self._last_prime)
            if p % 2 == 1:
                self.sub_primes.append(p)

    def _p_at(self, idx: int) -> int:
        """1-based index."""
        self._ensure_sub_primes(idx)
        return self.sub_primes[idx - 1]

    # ---- trial division primes ----
    def _ensure_trial_primes(self, need: int):
        if self.trial_prs:
            p = gmpy2.mpz(self.trial_prs[-1])
        else:
            self.trial_prs.append(2)
            p = gmpy2.mpz(2)
        while len(self.trial_prs) < need:
            p = gmpy2.next_prime(p)
            self.trial_prs.append(int(p))

    # ---- random even starter ----
    def _random_even_with_digits(self, d: int) -> int:
        lo = first_even_with_digits(d)
        hi = max_even_with_digits(d)
        count = ((hi - lo) // 2) + 1
        offset = self.rng.randrange(count)  # uniform in 0..count-1
        return lo + 2 * offset

    # ---- step sizing (big-int safe + stochastic rounding) ----
    def _rand_even_step(self, n: int) -> int:
        r = self.rng.randint(self.cfg.percent_num_lo, self.cfg.percent_num_hi)
        denom = 100 * (10 ** self.cfg.percent_scale)
        num = n * r
        q, rem = divmod(num, denom)   # exact integer division
        if rem and self.rng.random() < (rem / denom):
            q += 1
        if q < 1:
            q = 1
        step = 2 * q
        return step if step >= 2 else 2

    # ---- PRP (strong MR) ----
    @staticmethod
    def _mr_strong_prp(n_mpz: gmpy2.mpz, base: int) -> bool:
        a = base % n_mpz
        if a == 0:
            return True
        d = n_mpz - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        x = gmpy2.powmod(a, d, n_mpz)
        if x == 1 or x == n_mpz - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n_mpz
            if x == n_mpz - 1:
                return True
        return False

    # ---- primality pipeline: TD -> PRP(2,3) -> BPSW ----
    def _is_probable_prime(self, q: int, bpsw_counter: list[int]) -> tuple[bool, bool]:
        if q < 2:
            return False, False
        if (q & 1) == 0:
            return (q == 2), False if q != 2 else False

        q_mpz = gmpy2.mpz(q)

        for sp in self.trial_prs[:self.trial_bound]:
            if q == sp:
                return True, False
            if q % sp == 0:
                return False, False

        if not self._mr_strong_prp(q_mpz, 2):
            return False, False
        if not self._mr_strong_prp(q_mpz, 3):
            return False, False

        bpsw_counter[0] += 1
        return bool(gmpy2.is_prime(q_mpz)), True

    # ---- index streams ----
    @staticmethod
    def _centered_indices_infinite(k0: int):
        """Infinite generator: k0, k0-1, k0+1, k0-2, k0+2, ... (positive indices only)."""
        if k0 < 1:
            k0 = 1
        yield k0
        off = 1
        while True:
            l = k0 - off
            r = k0 + off
            if l >= 1:
                yield l
            yield r
            off += 1

    @staticmethod
    def _mid_q_candidates(n: int):
        """Infinite generator of odd q around n//2: ..., -3, -1, +1, +3, ..."""
        mid = n // 2
        if (mid & 1) == 0:
            left = mid - 1
            right = mid + 1
            yield left; yield right
            step = 3
        else:
            yield mid
            step = 2
        while True:
            yield mid - step
            yield mid + step
            step += 2

    # ---- build p-side forbidden residue bitset for this n ----
    def _build_p_forbidden(self, n: int) -> bytearray:
        """bytearray of length W; entry=1 if residue class is forbidden for p."""
        mask = bytearray(0 for _ in range(self.W))
        for r in self.wheel_primes:
            a = int(n % r)  # BigInt mod small int is fast
            for t in range(a, self.W, r):
                mask[t] = 1
        return mask

    # ---- triple-racer duel, per-BPSW turns; run until hit (with pre-sieve) ----
    def _test_one_n_triple_duel(self, n: int, k0_guess: int, wins: dict[str, int]) -> Metrics:
        start = time.perf_counter()
        bpsw_counter = [0]
        tried = 0  # counts only *post-pre-sieve* candidates

        # Precompute p-side forbidden residues for this n
        p_forbidden = self._build_p_forbidden(n)
        W = self.W
        allowed_q_mask = self.allowed_q_mask

        tried_p_global = set()     # indices used by seq/around (avoid duplicate p)
        tried_q_mid = set()        # q residues tried (for bookkeeping; not strictly needed)
        seq_next = 1
        around_iter = self._centered_indices_infinite(k0_guess)
        mid_iter = self._mid_q_candidates(n)
        mid_pending_q = None

        def budget_for(method: str) -> int:
            mx = max(wins.values()) if wins else 0
            return 2 if wins.get(method, 0) == mx and mx > 0 else 1

        order_pref = ['seq', 'around', 'mid']

        def pick_next_method(current: str | None) -> str:
            others = [m for m in order_pref if m != current]
            others.sort(key=lambda m: (-wins.get(m, 0), order_pref.index(m)))
            return others[0]

        current = max(order_pref, key=lambda m: (wins.get(m, 0), -order_pref.index(m)))
        budget = budget_for(current)

        while True:  # keep going until we find a decomposition
            method = current
            did_bpsw_now = False

            if method == 'seq':
                # advance to next unused p index, skip by p-side wheel mask BEFORE counting
                while True:
                    while seq_next in tried_p_global:
                        seq_next += 1
                    idx = seq_next
                    seq_next += 1
                    tried_p_global.add(idx)
                    p = self._p_at(idx)
                    if p_forbidden[p % W]:
                        # q = n - p would be divisible by a wheel prime; skip silently
                        continue
                    break  # p passed pre-sieve

                tried += 1  # only count nontrivial candidate
                q = n - p
                if q >= 2 and ((q & 1) == 1 or q == 2):
                    is_prime, did_bpsw = self._is_probable_prime(q, bpsw_counter)
                    did_bpsw_now = did_bpsw
                    if is_prime:
                        ms = (time.perf_counter() - start) * 1000.0
                        return Metrics(ms, tried, bpsw_counter[0], True, n, p, q, 'seq')

            elif method == 'around':
                # pull next centered index; skip duplicates and wheel-forbidden BEFORE counting
                while True:
                    idx = next(around_iter)
                    if idx in tried_p_global or idx < 1:
                        continue
                    tried_p_global.add(idx)
                    p = self._p_at(idx)
                    if p_forbidden[p % W]:
                        continue
                    break  # p passed pre-sieve

                tried += 1
                q = n - p
                if q >= 2 and ((q & 1) == 1 or q == 2):
                    is_prime, did_bpsw = self._is_probable_prime(q, bpsw_counter)
                    did_bpsw_now = did_bpsw
                    if is_prime:
                        ms = (time.perf_counter() - start) * 1000.0
                        return Metrics(ms, tried, bpsw_counter[0], True, n, p, q, 'around')

            else:  # mid (q-first)
                progressed = False
                for _ in range(2):  # allow q-then-p microstep if no BPSW needed
                    if mid_pending_q is not None:
                        q = mid_pending_q
                        p = n - q
                        # p-side wheel mask isn't needed here; we already know q is prime-ish
                        if p >= 2 and ((p & 1) == 1 or p == 2):
                            tried += 1  # we will actually test p now
                            is_prime, did_bpsw = self._is_probable_prime(p, bpsw_counter)
                            did_bpsw_now = did_bpsw
                            mid_pending_q = None
                            if is_prime:
                                ms = (time.perf_counter() - start) * 1000.0
                                return Metrics(ms, tried, bpsw_counter[0], True, n, p, q, 'mid')
                        else:
                            mid_pending_q = None
                        progressed = True
                        break

                    q_candidate = next(mid_iter)

                    # Quick structural rejections (cheap) BEFORE counting:
                    if q_candidate <= 1 or q_candidate >= n:
                        continue
                    if (q_candidate & 1) == 0 and q_candidate != 2:
                        continue
                    # Wheel mask on q-side: skip residues divisible by wheel primes
                    if not allowed_q_mask[q_candidate % W]:
                        continue
                    # Optional: avoid exact repeats for bookkeeping
                    if q_candidate in tried_q_mid:
                        continue
                    tried_q_mid.add(q_candidate)

                    # Passed pre-sieve; count it
                    tried += 1
                    is_q_prime, did_bpsw = self._is_probable_prime(q_candidate, bpsw_counter)
                    did_bpsw_now = did_bpsw
                    if is_q_prime:
                        mid_pending_q = q_candidate
                    progressed = True
                    break

                if not progressed:
                    pass  # keep looping

            # budget accounting: switch after each BPSW consumption
            if did_bpsw_now:
                budget -= 1
                if budget <= 0:
                    current = pick_next_method(current)
                    budget = budget_for(current)

        # unreachable

    # ---- random walk over n in a digit band ----
    def run_for_digits(self, d: int, count_limit: int):
        lo = first_even_with_digits(d)
        hi = max_even_with_digits(d)
        n = self._random_even_with_digits(d)  # uniform random even d-digit start
        produced = 0
        while produced < count_limit and n <= hi:
            yield n
            produced += 1
            step = self._rand_even_step(n)
            n_next = n + step
            if n_next > hi or len(str(n_next)) > d:
                break
            n = n_next

    # ---- autotune TD depth ----
    def autotune_after_trial(self, subs_tried: int, bpsw_checks: int, ms: float):
        self.window.append((subs_tried, bpsw_checks, ms))
        if len(self.window) < 16:
            return
        subs_sum = sum(s for s, _, _ in self.window)
        bpsw_sum = sum(b for _, b, _ in self.window)
        bpsw_frac = (bpsw_sum / subs_sum) if subs_sum else 0.0
        # target ~0.15–0.30
        if bpsw_frac > 0.35 and self.trial_bound < self.trial_bound_max:
            self.trial_bound = min(self.trial_bound + 16, self.trial_bound_max)
            self._ensure_trial_primes(self.trial_bound)
        elif bpsw_frac < 0.12 and self.trial_bound > self.trial_bound_min:
            self.trial_bound = max(self.trial_bound - 8, self.trial_bound_min)


# ------------------------------- CLI / Main ------------------------------ #

def print_config(cfg: Config):
    print("=== Config ===")
    print(f"digits: {cfg.digits_lo}:{cfg.digits_hi}")
    print(f"count (per digit length): {cfg.count_per_digits}")
    print(f"percent step range: {cfg.percent_display} (%)")
    print(f"seed: {cfg.seed if cfg.seed is not None else '(none)'}")
    print(f"python: {sys.version.split()[0]}")
    print(f"gmpy2: {gmpy2_version_str()}")
    print("================\n")


def weight_for_rep(p: int, q: int, n: int, lo: int, hi: int) -> float:
    # down-weight trailing zeros, band edges, and tiny p (aesthetic only)
    tz = 0
    m = n
    while m % 10 == 0 and tz < 64:
        m //= 10
        tz += 1
    w_tz = 1.0 / (1.0 + tz)

    span = hi - lo
    margin = span // 50 if span > 0 else 0  # ~2%
    near_edge = ((n - lo) < margin) or ((hi - n) < margin)
    w_edge = 0.5 if near_edge else 1.0

    w_p = 0.6 + min(0.4, (p or 0) / 50000.0)
    return w_tz * w_edge * w_p


def choose_representative(successes, rng: random.Random, d: int):
    if not successes:
        return None
    lo = first_even_with_digits(d)
    hi = max_even_with_digits(d)
    weights = [weight_for_rep(p, q, n, lo, hi) for (p, q, n) in successes]
    total = sum(weights)
    if total <= 0:
        return rng.choice(successes)
    r = rng.random() * total
    acc = 0.0
    for w, item in zip(weights, successes):
        acc += w
        if r <= acc:
            return item
    return successes[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Goldbach sampler (guaranteed-hit trials; wheel pre-sieve; random start per band; triple racer; self-tuning filters)."
    )
    parser.add_argument("--digits", required=True,
                        help="Start:End digit lengths (e.g., 1:20). Each digit length is sampled independently.")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of samples per digit length (default: 100).")
    parser.add_argument("--percent", default="1:2",
                        help="Random step size as a percent range of current n (e.g., 1:2 = 1%%..2%%, default).")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility (optional).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress showing p+q=n in summary output.")

    args = parser.parse_args(argv)

    digits_lo, digits_hi = parse_range_pair(args.digits, "digits", as_int=True)
    if digits_lo < 1:
        raise SystemExit("ERROR: --digits lower bound must be ≥ 1")

    A_scaled, B_scaled, k = parse_percent_pair_intscaled(args.percent)

    cfg = Config(
        digits_lo=digits_lo,
        digits_hi=digits_hi,
        count_per_digits=max(1, int(args.count)),
        percent_num_lo=A_scaled,
        percent_num_hi=B_scaled,
        percent_scale=k,
        percent_display=args.percent,
        seed=args.seed,
        quiet=args.quiet,
    )

    sampler = GoldbachSampler(cfg)
    print_config(cfg)

    for d in range(cfg.digits_lo, cfg.digits_hi + 1):
        total_ms = 0.0
        total_subs = 0
        total_trials = 0

        wins = {'seq': 0, 'around': 0, 'mid': 0}
        avg_k_so_far = 64.0  # reasonable starting guess; updates from data
        successes = []
        sampler.window.clear()

        for n in sampler.run_for_digits(d, cfg.count_per_digits):
            met = sampler._test_one_n_triple_duel(n, int(round(avg_k_so_far)), wins)

            total_trials += 1
            total_ms += met.ms_elapsed
            total_subs += met.subs_tried

            if met.method_winner in wins:
                wins[met.method_winner] += 1

            successes.append((met.p, met.q, met.n))

            # update running average attempts K (already post-pre-sieve)
            avg_k_so_far = (avg_k_so_far * (total_trials - 1) + met.subs_tried) / total_trials

            # keep autotune happy (bpsw count stored but not printed)
            sampler.autotune_after_trial(met.subs_tried, met.bpsw_checks, met.ms_elapsed)

        if total_trials == 0:
            print(f"Digits: {d} / Avg. Time per trial (ms): 0.000 / Avg. K until hit: 0.000")
            continue

        avg_ms = total_ms / total_trials
        avg_k = total_subs / total_trials

        line = (f"Digits: {d} / Avg. Time per trial (ms): {avg_ms:.3f} / "
                f"Avg. K until hit: {avg_k:.3f}")
        if not cfg.quiet and successes:
            rep = choose_representative(successes, sampler.rng, d)
            if rep is not None:
                p, q, n = rep
                line += f" / {p} + {q} = {n}"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
