
# llm_adapters.py
# Pluggable LLM adapter skeletons for PEACE-C.
# These are SAFE placeholders: they show you how to wire real clients without making any calls here.

from __future__ import annotations
from typing import Protocol

class LLMAdapter(Protocol):
    def ask(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str: ...

class MockLLM:
    """Matches the interface; same behavior as in peace_c.MockLLM for quick offline runs."""
    def ask(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
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
            if any(w in p for w in ["conflict", "novel", "ambiguous"]):
                return "B 0.55 Rationale: conflicting signals, hold at Both and seek context."
            return "T 0.62 Rationale: pattern prior supports truth."
        if "evidence snippets" in p:
            return "\\n".join([
                "Pro: City press release suggests upcoming policy change.",
                "Con: No council vote on record yet; timeline uncertain.",
            ])
        return "B 0.50 Rationale: default neutrality."

# ---------------- Real-client skeletons (fill in your keys + client code) ----------------

class OpenAIClientLLM:
    """Skeleton: replace `...` with your OpenAI client call. Returns plain text."""
    def __init__(self, model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY"):
        self.model = model
        self.api_key_env = api_key_env

    def ask(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # Pseudocode only:
        # import os, openai
        # openai.api_key = os.getenv(self.api_key_env)
        # resp = openai.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role":"user","content":prompt}],
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        # )
        # return resp.choices[0].message.content.strip()
        raise NotImplementedError("Wire your OpenAI client here.")

class AnthropicClientLLM:
    """Skeleton: replace `...` with your Anthropic client call. Returns plain text."""
    def __init__(self, model: str = "claude-3-haiku", api_key_env: str = "ANTHROPIC_API_KEY"):
        self.model = model
        self.api_key_env = api_key_env

    def ask(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # Pseudocode only:
        # import os
        # from anthropic import Anthropic
        # client = Anthropic(api_key=os.getenv(self.api_key_env))
        # resp = client.messages.create(
        #     model=self.model,
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        #     messages=[{"role":"user","content":prompt}],
        # )
        # return resp.content[0].text.strip()
        raise NotImplementedError("Wire your Anthropic client here.")
