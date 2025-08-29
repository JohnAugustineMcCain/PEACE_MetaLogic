# peace_c.py
# PEACE-C: A computational framework for meta-tracking truth & liminal cognition
# -----------------------------------------------------------------------------
# This module operationalizes key ideas from PEACE ("Paraconsistent Epistemic And Contextual Evaluation")
# into a runnable engine that can integrate with an LLM for linguistic/contextual discovery.
#
# Core ideas implemented:
# - Trivalent truth values {T, F, B} with B (Both) as the neutral, non-explosive default
# - Context completeness Cc(φ) as a stopping criterion for discovery
# - Perspectives as evaluative lenses (I, κ) selected by admissibility & load-bearing filters
# - Liminal expansion: during conflict/novelty, reallocate compute to self-referential context search
# - Oracle: pattern-based or LLM-based provisional verdicts with confidence; explicit fallback to discovery
#
# You can swap the LLM client by implementing LLMAdapter.ask(). See MockLLM and SimplePromptLLM below.
#
# (c) 2025 John Augustine McCain — MIT License for this prototype

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Protocol
import math
import random
import textwrap

# ----------------------------
# Truth values & basic helpers
# ----------------------------

class TruthValue(Enum):
    T = auto()   # True
    F = auto()   # False
    B = auto()   # Both / Underspecified (default, non-explosive)

def tv_to_str(tv: TruthValue) -> str:
    return {TruthValue.T: "T", TruthValue.F: "F", TruthValue.B: "B"}[tv]

# Designated values for consequence: T and B (paraconsistent safety)
DESIGNATED = {TruthValue.T, TruthValue.B}

# ----------------------------
# Evidence store
# ----------------------------

@dataclass
class Evidence:
    positive: float = 0.0  # evidence for
    negative: float = 0.0  # evidence against

    def base_truth(self, tp: float = 0.6, tn: float = 0.6) -> TruthValue:
        """Threshold evidence into {T,F,B}. If both are strong, return B (conflict)."""
        pos, neg = self.positive, self.negative
        if pos >= tp and neg < tn:
            return TruthValue.T
        if neg >= tn and pos < tp:
            return TruthValue.F
        if pos >= tp and neg >= tn:
            return TruthValue.B
        return TruthValue.B

@dataclass
class EvidenceStore:
    store: Dict[str, Evidence] = field(default_factory=dict)

    def add(self, atom: str, dpos: float = 0.0, dneg: float = 0.0) -> None:
        ev = self.store.get(atom, Evidence())
        ev.positive = min(1.0, ev.positive + dpos)
        ev.negative = min(1.0, ev.negative + dneg)
        self.store[atom] = ev

    def truth(self, atom: str) -> TruthValue:
        return self.store.get(atom, Evidence()).base_truth()

# ----------------------------
# Context & perspectives
# ----------------------------

@dataclass
class Context:
    facts: Dict[str, Any] = field(default_factory=dict)      # raw facts (strings -> values)
    notes: List[str] = field(default_factory=list)           # textual notes / rationales
    needed: List[str] = field(default_factory=list)          # outstanding questions
    order: List[str] = field(default_factory=list)           # rough priority / recency

    def add_fact(self, key: str, value: Any, note: Optional[str] = None) -> None:
        self.facts[key] = value
        if note:
            self.notes.append(note)
        self.order.append(key)

    def add_question(self, q: str) -> None:
        if q not in self.needed:
            self.needed.append(q)

    def answer_question(self, q: str) -> None:
        if q in self.needed:
            self.needed.remove(q)

@dataclass
class Perspective:
    """A perspective is a lens (assumptions I) with a verdict function κ."""
    name: str
    assumptions: Dict[str, Any] = field(default_factory=dict)
    kappa: Callable[[str, Context], TruthValue] = lambda phi, ctx: TruthValue.B
    # Admissibility & load-bearing checks (pluggable)
    admissible: Callable[[str, Context], bool] = lambda phi, ctx: True
    load_bearing: Callable[[str, Context], bool] = lambda phi, ctx: True
    priority: int = 0  # higher = earlier

# ----------------------------
# LLM adapter protocol
# ----------------------------

class LLMAdapter(Protocol):
    def ask(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        ...

class MockLLM:
    """A tiny heuristic stand-in for a real LLM. Replace with a production adapter."""
    def ask(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # Primitive, prompt-sensitive behaviors to simulate useful outputs.
        p = prompt.lower()
        if "propose 5 clarifying questions" in p:
            return "\\n".join([
                "What is the precise claim and its time frame?",
                "Which definitions are we using for key terms?",
                "What sources are admissible as evidence here?",
                "What would count as decisive contrary evidence?",
                "What is the decision threshold or required confidence?",
            ])
        if "assign a trivalent verdict" in p:
            # Heuristic: if the context mentions 'conflict' or 'novel' pick B; else coinflip T/F
            if "conflict" in p or "novel" in p or "ambiguous" in p:
                return "B 0.55 Rationale: conflicting signals, hold at Both and seek context."
            return random.choice(["T 0.62 Rationale: pattern prior supports truth.",
                                  "F 0.64 Rationale: pattern prior opposes truth."])
        if "generate candidate perspectives" in p:
            return "\\n".join([
                "EmpiricalPattern: assume regularities found in corpus hold here.",
                "CausalMechanism: assume a causal story must support the claim.",
                "PragmaticContext: assume speaker intent and stakes matter.",
            ])
        if "evidence snippets" in p:
            return "\\n".join([
                "Pro: Independent report aligns with the claim.",
                "Con: Source credibility is uncertain; possible bias.",
            ])
        return "B 0.50 Rationale: default neutrality."

# ----------------------------
# Context completeness estimator
# ----------------------------

@dataclass
class CcEstimator:
    target_questions: int = 5

    def compute(self, phi: str, ctx: Context) -> float:
        # Cc = 1 - outstanding_needed / target cap (>= 0)
        outstanding = min(len(ctx.needed), self.target_questions)
        return max(0.0, min(1.0, 1.0 - outstanding / float(self.target_questions)))

    def next_questions(self, phi: str, ctx: Context, llm: Optional[LLMAdapter] = None) -> List[str]:
        if llm is None:
            llm = MockLLM()
        prompt = f"""
        We are evaluating the claim: "{phi}".
        Propose 5 clarifying questions that, if answered, would most increase context completeness.
        """
        qs = [q.strip("- ").strip() for q in llm.ask(prompt).splitlines() if q.strip()]
        return qs[:5]

# ----------------------------
# Oracle for fast leaps (LLM-backed or heuristic)
# ----------------------------

@dataclass
class OracleVerdict:
    tv: TruthValue
    confidence: float
    rationale: str

@dataclass
class Oracle:
    llm: Optional[LLMAdapter] = None

    def predict(self, phi: str, ctx: Context) -> OracleVerdict:
        if self.llm is None:
            self.llm = MockLLM()
        prompt = f"""
        We are analyzing the claim: "{phi}".
        Context notes: {ctx.notes}
        Facts: {ctx.facts}
        Please assign a trivalent verdict T, F, or B with a confidence in [0,1].
        Format: "<TV> <conf> Rationale: ..."
        If signals conflict or novelty is high, prefer B with moderate confidence.
        """
        out = self.llm.ask(prompt)
        parts = out.strip().split()
        tv = TruthValue.B
        conf = 0.5
        rationale = out.strip()
        if parts:
            tag = parts[0].upper()
            if tag in ("T", "F", "B"):
                tv = {"T": TruthValue.T, "F": TruthValue.F, "B": TruthValue.B}[tag]
            for i, tok in enumerate(parts[1:], start=1):
                try:
                    conf = float(tok)
                    break
                except ValueError:
                    continue
        return OracleVerdict(tv=tv, confidence=max(0.0, min(1.0, conf)), rationale=rationale)

# ----------------------------
# Engine
# ----------------------------

@dataclass
class EngineConfig:
    cc_threshold: float = 0.8
    novelty_threshold: float = 0.6
    max_liminal_rounds: int = 4

@dataclass
class Engine:
    cc: CcEstimator = field(default_factory=lambda: CcEstimator(target_questions=5))
    # Optional: attach a retriever after constructing Engine
    retriever: Any = None
    oracle: Oracle = field(default_factory=Oracle)
    config: EngineConfig = field(default_factory=EngineConfig)
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    perspectives: List[Perspective] = field(default_factory=list)
    llm: LLMAdapter = field(default_factory=MockLLM)

    def measure_novelty(self, phi: str, ctx: Context) -> float:
        # Crude novelty proxy: few facts + many needed → high novelty
        nfacts = len(ctx.facts)
        need = len(ctx.needed)
        score = min(1.0, max(0.0, (need + 1) / (nfacts + need + 1)))
        return score

    def admissible_load_bearing(self, phi: str, ctx: Context) -> List[Perspective]:
        cands = [p for p in self.perspectives if p.admissible(phi, ctx) and p.load_bearing(phi, ctx)]
        return sorted(cands, key=lambda p: p.priority, reverse=True)

    def contextual_verdict(self, phi: str, ctx: Context) -> TruthValue:
        # Aggregate kappa over maximal admissible load-bearing perspectives
        ps = self.admissible_load_bearing(phi, ctx)
        if not ps:
            return TruthValue.B
        votes = [p.kappa(phi, ctx) for p in ps]
        # Simple merge: if all agree non-B, take it; else B
        if all(v == TruthValue.T for v in votes):
            return TruthValue.T
        if all(v == TruthValue.F for v in votes):
            return TruthValue.F
        return TruthValue.B

    def liminal_expand(self, phi: str, ctx: Context) -> Context:
        rounds = 0
        while self.cc.compute(phi, ctx) < self.config.cc_threshold and rounds < self.config.max_liminal_rounds:
            rounds += 1
            # Ask LLM for best questions
            qs = self.cc.next_questions(phi, ctx, self.llm)
            for q in qs:
                ctx.add_question(q)
            # "Answer" some questions via oracle/evidence synthesis (placeholder)
            # In real use, you'd route to tools, retrieval, or humans.
            to_answer = min(2, len(ctx.needed))
            for q in list(ctx.needed)[:to_answer]:
                # Try retrieval first, then ask LLM
                snippet = ""
                if getattr(self, "retriever", None) is not None:
                    hits = self.retriever.search(q, k=2)
                    if hits:
                        joined = " \n ".join([f"[{h.doc_id}] {h.text[:300]}" for h in hits])
                        snippet = f"Retrieved: {joined}"
                if not snippet:
                    ev_prompt = f'Provide evidence snippets for question: "{q}" given claim "{phi}".'
                    snippet = self.llm.ask(ev_prompt)
                ctx.add_fact(f"ans:{q[:40]}", snippet, note=f"Answered: {q}")
                ctx.answer_question(q)
        return ctx

    def tick(self, phi: str, ctx: Context) -> Tuple[TruthValue, float, str]:
        novelty = self.measure_novelty(phi, ctx)
        if novelty > self.config.novelty_threshold:
            ctx = self.liminal_expand(phi, ctx)

        # Quick oracle pass
        o = self.oracle.predict(phi, ctx)
        # If oracle says B with low conf, try contextual verdict
        if o.tv == TruthValue.B and o.confidence < 0.66:
            tv = self.contextual_verdict(phi, ctx)
            conf = max(o.confidence, self.cc.compute(phi, ctx))
            rationale = f"Contextual verdict {tv_to_str(tv)} at Cc={self.cc.compute(phi, ctx):.2f}; Oracle said: {o.rationale}"
            return tv, conf, rationale
        else:
            return o.tv, o.confidence, o.rationale

# ----------------------------
# Default perspectives
# ----------------------------

def empirical_pattern_perspective() -> Perspective:
    def kappa(phi: str, ctx: Context) -> TruthValue:
        # Very naive: if at least one "Pro" snippet in notes, tilt T; if "Con" dominates, tilt F.
        pros = sum("Pro:" in n for n in ctx.notes)
        cons = sum("Con:" in n for n in ctx.notes)
        if pros > cons and pros >= 1:
            return TruthValue.T
        if cons > pros and cons >= 1:
            return TruthValue.F
        return TruthValue.B

    return Perspective(
        name="EmpiricalPattern",
        assumptions={"regularities": True},
        kappa=kappa,
        admissible=lambda phi, ctx: True,
        load_bearing=lambda phi, ctx: True,
        priority=10
    )

def pragmatic_context_perspective() -> Perspective:
    def kappa(phi: str, ctx: Context) -> TruthValue:
        # If stakes or intent appear in facts/notes, prefer B unless strongly resolved
        text = " ".join(ctx.notes) + " " + " ".join(map(str, ctx.facts.values()))
        if any(w in text.lower() for w in ["stake", "intent", "sarcasm", "irony"]):
            return TruthValue.B
        return TruthValue.B
    return Perspective(
        name="PragmaticContext",
        assumptions={"speech_acts": True},
        kappa=kappa,
        admissible=lambda phi, ctx: True,
        load_bearing=lambda phi, ctx: True,
        priority=5
    )
