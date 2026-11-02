"""One price list for every LLM call the project bills.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

AS_OF = "2026-09-02"
_OPENAI_SOURCE = ("openai.com/api/pricing; matches langchain_community 0.3.31 and the litellm "
                  "model_cost table on this date")
_GEMINI_SOURCE = "ai.google.dev/pricing (<=200k-token prompts); matches the litellm model_cost table"


@dataclass(frozen=True)
class ModelPrice:
    model: str
    provider: str
    input: float          # USD per 1M fresh prompt tokens
    cached_input: float   # USD per 1M prompt tokens served from the provider's cache
    output: float         # USD per 1M completion tokens, reasoning/thinking included
    source: str
    as_of: str = AS_OF
    note: str = ""


def _openai(model, inp, cached, out, note=""):
    return ModelPrice(model, "openai", inp, cached, out, _OPENAI_SOURCE, note=note)


def _gemini(model, inp, cached, out):
    return ModelPrice(model, "google", inp, cached, out, _GEMINI_SOURCE)


PRICES: dict[str, ModelPrice] = {p.model: p for p in (
    # --- OpenAI: the six paper models, the ablation/SIFTA models, and the legacy ones on disk.
    _openai("gpt-5", 1.25, 0.125, 10.0),
    _openai("gpt-5-mini", 0.25, 0.025, 2.0),
    _openai("gpt-5-nano", 0.05, 0.005, 0.40),
    _openai("gpt-5-chat-latest", 1.25, 0.125, 10.0),
    _openai("gpt-5.1", 1.25, 0.125, 10.0, note="litellm model_cost table; not in langchain_community"),
    # SIFTA / GEPA reflection LM. Absent from langchain, so the DSPy rows were
    # the only ones priced by litellm before this table existed.
    _openai("gpt-5.2", 1.75, 0.175, 14.0, note="litellm model_cost table; not in langchain_community"),
    _openai("gpt-4.1", 2.0, 0.5, 8.0),
    _openai("gpt-4.1-mini", 0.40, 0.10, 1.60),
    _openai("gpt-4.1-nano", 0.10, 0.025, 0.40),
    _openai("gpt-4o", 2.50, 1.25, 10.0),
    _openai("gpt-4o-2024-05-13", 5.0, 5.0, 15.0, note="dated snapshot priced differently from gpt-4o; no cached rate"),
    _openai("gpt-4o-mini", 0.15, 0.075, 0.60),
    _openai("o1", 15.0, 7.5, 60.0),
    _openai("o3", 2.0, 0.5, 8.0),
    _openai("o3-mini", 1.10, 0.55, 4.40),
    _openai("o4-mini", 1.10, 0.275, 4.40),
    _openai("gpt-4", 30.0, 30.0, 60.0, note="legacy; no cached rate"),
    # The codex CLI model. Billed at the account's negotiated rate, which is what
    # the invoices show; the public litellm table lists 4.0 / 0.4 / 20.0 instead.
    ModelPrice("gpt-5.6-sol", "openai", 5.0, 0.5, 30.0,
               source="account price list 2026-08-25 (litellm model_cost lists 4.0 / 0.4 / 20.0)",
               as_of="2026-08-25"),
    ModelPrice("gpt-6-sol", "openai", 2.0, 0.2, 10.0,
               source="developers.openai.com/api/docs/models/gpt-6-sol",
               as_of="2026-09-27"),
    # --- Google.
    _gemini("gemini-2.5-pro", 1.25, 0.125, 10.0),
    _gemini("gemini-2.5-flash", 0.30, 0.03, 2.50),
    ModelPrice("gemini-3.7-flash", "google", 0.75, 0.075, 3.75,
               source="ai.google.dev/gemini-api/docs/pricing, standard tier, price through 2026-12-31",
               as_of="2026-09-03"),
    ModelPrice("gemini-3.8-flash", "google", 0.75, 0.075, 3.75,
               source="ai.google.dev/gemini-api/docs/pricing, standard tier, price through 2026-12-31",
               as_of="2026-10-26"),
)}

_PROVIDER_PREFIX = re.compile(r"^(?:openai|azure|gemini|google|google_genai|vertex_ai|anthropic|models)/")
_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")
_EFFORT_SUFFIX = re.compile(r"-(?:minimal|low|medium|high|xhigh)$")


def canonical_model(name: str | None) -> str | None:
    """The price-list key a provider or project model name resolves to, or None."""
    if not name:
        return None
    key = _PROVIDER_PREFIX.sub("", name.strip().lower())
    for candidate in (key, _DATE_SUFFIX.sub("", key), _EFFORT_SUFFIX.sub("", _DATE_SUFFIX.sub("", key))):
        if candidate in PRICES:
            return candidate
    return None


def rates_for(name: str | None) -> ModelPrice | None:
    key = canonical_model(name)
    return PRICES[key] if key else None


def price_tokens(model: str | None, prompt_tokens: int, completion_tokens: int,
                 cached_prompt_tokens: int = 0) -> float | None:
    """USD for one call or one aggregate, or None when the model has no price.

    `prompt_tokens` includes the cached portion (the provider's convention);
    `cached_prompt_tokens` is billed at the cached rate and the remainder fresh.
    """
    rates = rates_for(model)
    if rates is None:
        return None
    prompt = int(prompt_tokens or 0)
    cached = min(int(cached_prompt_tokens or 0), prompt)
    fresh = prompt - cached
    return (fresh * rates.input + cached * rates.cached_input
            + int(completion_tokens or 0) * rates.output) / 1_000_000


def describe(name: str | None) -> dict | None:
    """The entry as a plain dict, for writing beside a cost in an artifact."""
    rates = rates_for(name)
    return asdict(rates) if rates else None


def known_models() -> list[str]:
    return sorted(PRICES)
