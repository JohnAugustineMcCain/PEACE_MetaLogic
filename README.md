# PEACE MetaLogic
An attempt to synthesize the Cognitive Science insights of Paul Jorion with the independent research of John McCain

## First toy prototype (untested)

## Features

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

## Files

- `peace_c.py`
- `llm_adapters.py`
- `retrieval.py`
- `demo_with_retrieval.py`

---

## Run the retrieval demo

```bash
python3 demo_with_retrieval.py

```

## Use a real LLM

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

Whether or not this is actually feasable is yet to be tested. If it works, it could be an important step in creating systems that genuinely think and can reason philosophically.

Engine.liminal_expand(...) now:

- asks the LLM for clarifying questions,

- retrieves top-k snippets to answer some,

- falls back to the LLM for “evidence snippets” if retrieval is empty,

- raises Cc until the threshold is met and then collapses.

To be used with a:

- a JSONL run-logger,

- a “policy” hook that decides actions from (TruthValue, confidence, context)
