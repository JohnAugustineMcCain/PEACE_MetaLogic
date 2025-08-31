# PEACE MetaLogic
An attempt to synthesize the Cognitive Science insights of Paul Jorion with the independent research of John McCain

## Mission Statement

PEACE is not just another retrieval or document analyzer.  
It is an **epistemic engine**: a system that treats knowledge as a living process.  
Instead of stopping at “answers,” it models how knowing itself unfolds — through tension, paradox, clarification, and eventual collapse into truth, falsehood, or ambiguity.  

The goal is to explore how machines can **reason about their own reasoning**, tracking confidence, novelty, and context completeness as first-class citizens.  
This project is both a working prototype and a research experiment: part logic engine, part cognitive model, and part invitation to re-imagine what it means for AI to *know*.

## First toy prototype (untested)

### Features

1. **Pluggable LLM adapters with ready-to-fill skeletons**
   - `llm_adapters.py` includes:
     - `MockLLM` (offline, already works)
     - `OpenAIClientLLM` skeleton
     - `AnthropicClientLLM` skeleton  
   Swap in your real client by implementing `.ask()`.

2. **Lightweight retrieval hook for liminal expansion**
   - `retrieval.py` implements a tiny, dependency-free bag-of-words retriever.
   - The engine tries retrieval results to answer clarifying questions before asking the LLM.

3. **Demo that uses both**
   - `demo_with_retrieval.py` shows an end-to-end run with a mini “civic” corpus and the mock LLM.

---

### Files

- `peace_c.py`
- `llm_adapters.py`
- `retrieval.py`
- `demo_with_retrieval.py`

---

### Run the retrieval demo

```bash
python3 demo_with_retrieval.py

```

### Use a real LLM

In demo_with_retrieval.py, swap the import:

```
from llm_adapters import OpenAIClientLLM  # or AnthropicClientLLM
# ...
engine = Engine(
    cc=CcEstimator(target_questions=5),
    oracle=Oracle(llm=OpenAIClientLLM()),
    config=EngineConfig(cc_threshold=0.8, novelty_threshold=0.55, max_liminal_rounds=3),
    llm=OpenAIClientLLM()
)
```

Then implement .ask() in your adapter (API call + return str).

### Point-and-shoot retrieval

Add your documents to the mini-corpus in demo_with_retrieval.py:

```
corpus = [
  ("doc1", "Your text here...", {"date":"2025-08-28"}),
  # ...more docs
]
retriever = Retriever()
retriever.index(corpus)
engine.retriever = retriever
```

## What's the meaning of this?

The point is to make an LLM into a Meta-logical reasoning machine that can look at a document and generate summarized information about it while asking itself questions to remainin epistemically consistent, making "safe" decisions based upon what can he determined to be true rather than blindly accepting information to tie into probabilistic generation.

Whether or not this is actually feasable is yet to be tested. If it works, it could be an important step in creating systems that genuinely think and can reason philosophically. I dont have the resources or expertise to do this on my own.

Engine.liminal_expand(...) now:

- asks the LLM for clarifying questions,

- retrieves top-k snippets to answer some,

- falls back to the LLM for “evidence snippets” if retrieval is empty,

- raises Cc until the threshold is met and then collapses.

## Future directions

*These are not current features. This is about trying to predict the direction this work is heading*

### Near-term improvements
1. **Add JSONL tracing**  
   Each tick writes claim, questions, chosen answers, retrieval hits, LLM prompts, raw completions, verdict, Cc, novelty, and time. This gives you reproducibility and error analysis.

2. **Add a policy module**  
   Map (truth value, confidence, Cc) to actions such as continue searching, escalate, or decide. Make thresholds configurable in code.

3. **Add perspective plugins**  
   - *Causal mechanism*: require a minimal causal chain before `T`.  
   - *Source trust*: weight evidence by source reliability.  
   - *Temporal consistency*: penalize contradictions across time.  
   - *Pragmatics*: detect irony or hedges and hold `B` unless clarified.

4. **Integrate an embedding retriever**  
   Replace the bag-of-words retriever with a vector index. Keep the same interface so your engine code does not change.

5. **Calibrate confidence**  
   Create a small benchmark of claims with known outcomes. Run the engine and compute calibration curves for confidence vs accuracy. Adjust thresholds to minimize over-confidence.

---

### Medium steps that raise the ceiling
1. **Active questioning**  
   Train a small scorer that ranks clarifying questions by expected Cc gain using past logs. This makes the liminal loop smarter.

2. **Pattern oracle with justification**  
   Require the LLM to return a verdict, a probability, and three minimal supporting sentences with citations. Store each citation in the trace.

3. **Partial collapses**  
   Allow a claim to collapse to `T` or `F` inside one perspective while remaining `B` globally. Show users both the local and the global view.

4. **Multi-turn memory**  
   Persist context across related claims. Let Cc and evidence accumulate so later decisions get faster and crisper.

5. **Safety rails**  
   Add checks for hallucination, circular citation, and stale evidence. If detected, force `B` and trigger targeted questions.

---

### Ambitious extensions that fit the vision
1. **Meta-cognitive controller**  
   A small bandit or RL policy chooses among search actions: retrieve, ask, reframe the claim, switch perspective, or collapse. Reward is a mix of accuracy, cost, and latency.

2. **Human-in-the-loop**  
   When `B` persists, generate the smallest set of yes/no questions for a human. Update Cc and re-decide. This showcases PEACE as a real assistant.

3. **Cross-modal context**  
   Add channels for tables, figures, or audio transcripts. The same liminal loop runs, only with different monitors.

4. **Formal bridge**  
   Export a proof sketch when a collapse occurs: assumptions used, rules applied, and evidence edges. This gives you an interpretable artifact.

## sm_peace_oracle.py

`sm_peace_oracle.py` is a **PEACE-C–compatible self-modifying oracle**.  
It combines your SQLite-backed provenance, safety/math perspectives, and problem classes with the PEACE-C engine (liminal expansion, context completeness, and trivalent truth).

### What it does
- **Runs the PEACE-C engine** on natural-language claims (`T/F/B` with context completeness `Cc` and a liminal expansion loop).
- **Logs provenance to SQLite** (`peace_cache.db`): claims, prompts, retrieval hits, verdicts, confidences, timestamps.
- **Evaluates code modifications** via **safety** / **mathematical soundness** / **category error** perspectives.
- **Solves math problems** with your existing `MathematicalProblem` flow, plus meta-logical analysis when direct solutions aren’t feasible.

### Truth mapping
- `TruthValue.T` ↔ `TV.TRUE`  
- `TruthValue.F` ↔ `TV.FALSE`  
- `TruthValue.B` ↔ `TV.BOTH`  
- `TV.UNKNOWN` is treated as **hold** → `TruthValue.B` (keeps paraconsistent safety; avoids explosion).

### File relationships
- Depends on your PEACE-C core (`peace_c.py`) for:
  - `Engine`, `EngineConfig`, `CcEstimator`, `Oracle`, `Context`, `TruthValue`
  - perspectives: `empirical_pattern_perspective()`, `pragmatic_context_perspective()`
- Optional: `retrieval.py` (tiny bag-of-words retriever)
- Optional: `llm_adapters.py` (e.g., `MockLLM`, OpenAI/Anthropic skeletons)

### Quickstart
```bash
# run the demo
python3 sm_peace_oracle.py
