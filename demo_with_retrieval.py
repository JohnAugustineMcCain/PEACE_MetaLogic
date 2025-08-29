
# demo_with_retrieval.py
from peace_c import (
    Engine, EngineConfig, CcEstimator, Oracle, Context,
    empirical_pattern_perspective, pragmatic_context_perspective, tv_to_str
)
from llm_adapters import MockLLM  # swap for OpenAIClientLLM or AnthropicClientLLM
from retrieval import Retriever

def main():
    phi = "The city plans to reduce downtown speed limits next month."

    # Build a tiny corpus as if scraped from news/civic sources
    corpus = [
        ("press_001", "City Hall press release: Proposal to reduce downtown speed limits to 20 mph will be presented to council next Tuesday.", {"date":"2025-08-20"}),
        ("news_017", "Local paper reports debate on traffic calming measures. Several council members favor lower limits but timeline unclear.", {"date":"2025-08-22"}),
        ("council_aa", "Agenda item: Transportation. Discussion of speed policy postponed to September session.", {"date":"2025-08-25"}),
        ("forum_x", "Neighborhood forum post: I heard they already voted. That seems wrong; cannot find minutes.", {"date":"2025-08-21"}),
    ]
    retriever = Retriever()
    retriever.index(corpus)

    ctx = Context()
    ctx.add_fact("source_hint", "Heard at a neighborhood meeting")
    ctx.add_fact("timeframe", "Next month (unspecified week)")

    engine = Engine(
        cc=CcEstimator(target_questions=5),
        oracle=Oracle(llm=MockLLM()),
        config=EngineConfig(cc_threshold=0.8, novelty_threshold=0.55, max_liminal_rounds=3),
        llm=MockLLM()
    )
    engine.retriever = retriever
    engine.perspectives = [empirical_pattern_perspective(), pragmatic_context_perspective()]

    tv, conf, rationale = engine.tick(phi, ctx)

    print("=== PEACE-C Demo (Retrieval + LLM) ===")
    print(f"Claim: {phi}")
    print(f"TruthValue: {tv_to_str(tv)}  Confidence: {conf:.2f}")
    print("Rationale:")
    print(rationale)
    print("\nFacts collected:")
    for k, v in ctx.facts.items():
        snippet = v if isinstance(v, str) else str(v)
        print(f" - {k}: {snippet[:140]}{'...' if len(snippet)>140 else ''}")
    print("\nOutstanding questions:")
    for q in ctx.needed:
        print(f" - {q}")

if __name__ == "__main__":
    main()
