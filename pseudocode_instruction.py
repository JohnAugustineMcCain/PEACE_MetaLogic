TYPE TruthValue = { T, F, B }   # B includes undecidable/unknown/inconsistent

RECORD Verdict:
    tv: TruthValue
    confidence: float in [0,1]
    reason: Map
    witness: Optional<Map>

RECORD TrialRow:
    t: int
    probe_name: str
    y: {0,1}              # outcome (1 = probe passed; 0 = probe failed)
    p_mix: float          # forecasted success prob
    pmf_at_y: float       # likelihood of observed outcome under mixture
    pit: float            # randomized PIT for calibration
    ci: float             # confidence index after update
    earned: float         # cumulative earned confidence
    witness: Optional<Map>
    notes: str

# ---------- Domain Adapter (problem-specific plugin) ----------

INTERFACE DomainAdapter:
    name(): str

    # Human-facing "ask-first" clarifications that fix the ambient context
    ask_first_questions(): List[str]

    # Finite, bounded probes implied by the conjecture.
    # Each returns: outcome y ∈ {0,1}, optional constructive witness, and features for forecasting
    next_probe_schedule(trial_index t): (probe_name: str, probe_params: Map)
    run_probe(probe_name, probe_params): (y:0|1, witness:Optional<Map>, feat:Map)

    # Detects a constructive refuter (e.g., nontrivial Collatz cycle, even n with no Goldbach decomp
    # after truly exhaustive search up to a declared bound, explicit envelope violation with certificate, etc.)
    is_counterexample(witness: Map): bool

    # Optional: domain partitions for stability checks (e.g., residue classes, scale bands)
    bucket_key(feat: Map): Hashable

    # Heuristic family predicting P(y=1 | feat) before running the probe
    heuristics(): List<Heuristic>   # each Heuristic: predict_p(feat)->[0,1]

# ---------- Calibration engine (problem-independent) ----------

RECORD Heuristic:
    name: str
    predict_p: (feat:Map) -> float   # Bernoulli success probability

CLASS Bernoulli:
    pmf(y, p) = p if y=1 else (1-p)
    cdf(y, p) = 0 if y<0; (1-p) if y=0; 1 if y≥1

FUNCTION randomized_pit(y:int, p:float, rng) -> float:
    Fy  = Bernoulli.cdf(y, p)
    Fy_ = Bernoulli.cdf(y-1, p) if y>0 else 0
    return Fy_ + rng.uniform(0,1) * (Fy - Fy_)

CLASS CalibrationEngine:
    INIT(heuristics H[1..m], seed):
        w[i] = 1/m for all i
        ll_each[i] = 0
        ll_mix = 0
        pits = []
        bucket_scores = Map<BucketKey, float>()   # for stability across regimes
        n = 0
    FORECAST(feat):
        p_list[i] = clamp(H[i].predict_p(feat), eps, 1-eps)
        p_mix = sum_i w[i]*p_list[i]
        return (p_list, p_mix, weights=w)
    UPDATE(feat, y, bucket_key):
        (p_list, p_mix, w) = FORECAST(feat)
        pmf_i[i] = Bernoulli.pmf(y, p_list[i]);  ll_each[i] += log(max(pmf_i[i], tiny))
        pmf_mix  = Bernoulli.pmf(y, p_mix);      ll_mix     += log(max(pmf_mix, tiny))
        # multiplicative weights update
        w_new[i] = w[i]*pmf_i[i];  W = sum_i w_new[i];  w = (W>0) ? w_new/W : uniform
        pit = randomized_pit(y, p_mix, rng); pits.append(pit)
        bucket_scores[bucket_key] += log(max(pmf_mix, tiny))
        return (p_mix, pmf_mix, pit, weights=w)
    CONFIDENCE_INDEX():
        s_cal  = ks_uniformity_score(pits)          # PIT uniformity ∈ [0,1]
        s_pred = advantage_over_baseline(ll_each)   # avg logscore edge, squashed to [0,1]
        s_stab = stability(bucket_scores)           # variance penalty across buckets
        return geometric_mean([s_cal, s_pred, s_stab])

FUNCTION ks_uniformity_score(pits)->[0,1]: 
    # KS-style D statistic → exp(-c*D^2)

FUNCTION advantage_over_baseline(ll_each)->[0,1]:
    # (best - baseline)/n → 1 - exp(-k·max(0,·))

FUNCTION stability(bucket_scores)->[0,1]:
    # 1/(1 + α·Var(bucket_means))

# ---------- Earned confidence gate (problem-independent) ----------

CLASS EarnedConfidence:
    INIT(): value = 0.0
    BUMP(eps): value = min(1.0, value + eps)

STRUCT TeacherConfig:
    trials: int
    pmf_floor: float      # minimum likelihood-at-outcome to count as calibrated
    pit_band: (lo, hi)    # PIT must land in central band
    ci_floor: float       # confidence index threshold
    eps: float            # earned bump size

# ---------- Main Algorithm (problem-independent) ----------

FUNCTION META_TEACH(adapter: DomainAdapter, cfg: TeacherConfig, seed:int) -> (Verdict, List<TrialRow>, Summary):
    rng ← Random(seed)
    H   ← adapter.heuristics()
    CE  ← CalibrationEngine(H, seed)
    EC  ← EarnedConfidence()

    ask_first ← adapter.ask_first_questions()   # recorded in summary

    rows ← []
    witnesses ← []
    success_count ← 0

    FOR t in 0 .. cfg.trials-1:
        (probe_name, params) ← adapter.next_probe_schedule(t)
        (y, witness, feat)   ← adapter.run_probe(probe_name, params)

        IF witness != null AND adapter.is_counterexample(witness):
            # Verification asymmetry: a single constructive refuter flips to FALSE.
            v = Verdict(tv=F, confidence=1.0, reason={"refuter": witness, "probe":probe_name}, witness=witness)
            rows.append( TrialRow(t, probe_name, y, 0, 0, 0, 0, EC.value, witness, "counterexample") )
            RETURN (v, rows, summary_from(ask_first, CE, EC, witnesses, adapter))

        bucket = adapter.bucket_key(feat)
        (p_mix, pmf_at_y, pit, _) = CE.UPDATE(feat, y, bucket)
        ci = CE.CONFIDENCE_INDEX()

        pmf_ok = (pmf_at_y >= cfg.pmf_floor)
        pit_ok = (cfg.pit_band.lo <= pit <= cfg.pit_band.hi)
        ci_ok  = (ci >= cfg.ci_floor)
        IF pmf_ok AND pit_ok AND ci_ok:
            EC.BUMP(cfg.eps)

        IF y==1 AND witness != null AND len(witnesses) < 5:
            witnesses.append(witness)
            success_count += 1

        rows.append( TrialRow(t, probe_name, y, p_mix, pmf_at_y, pit, ci, EC.value, witness,
                              notes=f"pmf_ok={pmf_ok},pit_ok={pit_ok},ci_ok={ci_ok}") )

    # No refuter found within budget.
    # Bounded probes: we can assert TRUE for the bounded claims with witnesses; global statement remains BOTH.
    tv_bounded = (len(witnesses)>0) ? T : B
    v = Verdict(
        tv = tv_bounded, 
        confidence = CE.CONFIDENCE_INDEX(),
        reason = {
            "policy": "TRUE/FALSE only for bounded probes with constructive artifacts; global stays BOTH.",
            "earned_confidence": EC.value,
            "ask_first": ask_first
        },
        witness = (len(witnesses)>0) ? witnesses[0] : null
    )
    RETURN (v, rows, summary_from(ask_first, CE, EC, witnesses, adapter))

FUNCTION summary_from(ask_first, CE, EC, witnesses, adapter) -> Map:
    RETURN {
        "problem": adapter.name(),
        "confidence_index": CE.CONFIDENCE_INDEX(),
        "earned_confidence": EC.value,
        "examples": witnesses,
        "notes": "See per-trial logs for forecasts, PIT, and stability buckets."
    }
