import os
import time

import litellm

from prismadv.llm_backend.llm_pricing_info import price_tokens
from prismadv.llm_backend.model_routes import litellm_completion_kwargs, litellm_route

# In-memory log of every agent LLM exchange (prompt + response), in call order.
# Drivers reset it before a run and persist it afterwards as a trajectory.
MESSAGE_LOG: list[dict] = []

# Token/cost/latency totals for the current run, in the shared cost_summary shape.
USAGE_TOTALS: dict = {}


def _blank_usage() -> dict:
    return {
        "cost": 0.0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "reasoning_tokens": 0,
        "requests": 0,
        "uncosted_requests": 0,
        "elapsed_s": 0.0,
    }


def reset_message_log():
    MESSAGE_LOG.clear()
    USAGE_TOTALS.clear()
    USAGE_TOTALS.update(_blank_usage())


def get_message_log():
    return list(MESSAGE_LOG)


def get_usage_summary() -> dict:
    """Totals for the current run, keyed like every other cost_summary stage."""
    usage = dict(USAGE_TOTALS) if USAGE_TOTALS else _blank_usage()
    usage["elapsed_s"] = round(usage["elapsed_s"], 3)
    return usage


def _record_usage(model, response, elapsed_s):
    if not USAGE_TOTALS:
        USAGE_TOTALS.update(_blank_usage())
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0

    cost = price_tokens(model, prompt_tokens, completion_tokens, cached)
    if cost is None:
        # Unknown model: still report tokens, just not a dollar figure.
        USAGE_TOTALS["uncosted_requests"] += 1
    else:
        USAGE_TOTALS["cost"] += cost
    USAGE_TOTALS["total_tokens"] += total_tokens
    USAGE_TOTALS["prompt_tokens"] += prompt_tokens
    USAGE_TOTALS["completion_tokens"] += completion_tokens
    USAGE_TOTALS["cached_prompt_tokens"] += cached
    USAGE_TOTALS["reasoning_tokens"] += reasoning
    USAGE_TOTALS["requests"] += 1
    USAGE_TOTALS["elapsed_s"] += elapsed_s


def call_llm(prompt):
    # Agentic baselines use gpt-5 by default; the driver sets POCKETFLOW_MODEL per run.
    # Routed through litellm so a Gemini name reaches Gemini with GEMINI_API_KEY.
    model = os.environ.get("POCKETFLOW_MODEL", "gpt-5")
    started = time.perf_counter()
    response = litellm.completion(
        model=litellm_route(model),
        messages=[{"role": "user", "content": prompt}],
        **litellm_completion_kwargs(model),
    )
    _record_usage(model, response, time.perf_counter() - started)
    content = response.choices[0].message.content
    MESSAGE_LOG.append({"model": model, "prompt": prompt, "response": content})
    return content
