# PEACE MetaLogic
An attempt to synthesize the Cognitive Science insights of Paul Jorion with the independent research of John McCain

### First prototype instructions (untested)

bash```
python3 /mnt/data/demo_with_retrieval.py
```

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


corpus = [
  ("doc1", "Your text here...", {"date":"2025-08-28"}),
  # ...more docs
]
retriever = Retriever()
retriever.index(corpus)
engine.retriever = retriever
```

### What's the meaning of this?

	•	Engine.liminal_expand(...) now:
	1.	asks the LLM for clarifying questions,
	2.	retrieves top-k snippets to answer some,
	3.	falls back to the LLM for “evidence snippets” if retrieval is empty,
	4.	raises Cc until the threshold is met and then collapses.


	•	a JSONL run-logger,
	•	a “policy” hook that decides actions from (TruthValue, confidence, context)
