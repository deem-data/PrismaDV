from __future__ import annotations

from typing import Any

from prismadv.llm_backend.llm_pricing_info import price_tokens

STAGE = "constraint_generation"


def _nested(usage: dict, outer: str, inner: str) -> int:
    details = usage.get(outer) or {}
    return int(details.get(inner) or 0) if isinstance(details, dict) else 0


def usage_from_trajectory(trajectory: dict[str, Any], elapsed_s: float | None = None) -> dict[str, Any]:
    """One cost_summary stage for the episode. `elapsed_s` is omitted when unknown."""
    info = trajectory.get("info") or {}
    stats = info.get("model_stats") or {}
    usage = {
        "cost": 0.0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "reasoning_tokens": 0,
        "requests": int(stats.get("api_calls") or 0),
    }
    for message in trajectory.get("messages") or []:
        response = ((message.get("extra") or {}).get("response") or {}) if isinstance(message, dict) else {}
        block = response.get("usage") or {}
        if not block:
            continue
        prompt = int(block.get("prompt_tokens") or 0)
        completion = int(block.get("completion_tokens") or 0)
        usage["prompt_tokens"] += prompt
        usage["completion_tokens"] += completion
        usage["total_tokens"] += int(block.get("total_tokens") or (prompt + completion))
        usage["cached_prompt_tokens"] += _nested(block, "prompt_tokens_details", "cached_tokens")
        usage["reasoning_tokens"] += _nested(block, "completion_tokens_details", "reasoning_tokens")
    # Priced from the project list like every other family, so the agent row
    # sits on the same price sheet; mini's own litellm figure only when the
    # model is not on the list (counted, so the row is never silently cheaper).
    model = model_from_trajectory(trajectory)
    priced = price_tokens(model, usage["prompt_tokens"], usage["completion_tokens"],
                          usage["cached_prompt_tokens"]) if usage["total_tokens"] else None
    if priced is not None:
        usage["cost"] = priced
        usage["uncosted_requests"] = 0
    else:
        usage["cost"] = float(stats.get("instance_cost") or 0.0)
        usage["uncosted_requests"] = usage["requests"] if not usage["cost"] else 0
        usage["cost_source"] = "mini-swe-agent instance_cost (litellm)"
    if elapsed_s is not None:
        usage["elapsed_s"] = round(elapsed_s, 3)
    return usage


def model_from_trajectory(trajectory: dict[str, Any]) -> str | None:
    """The litellm route the episode ran on, from mini's config or the first response."""
    config = ((trajectory.get("info") or {}).get("config") or {}).get("model") or {}
    if config.get("model_name"):
        return str(config["model_name"])
    for message in trajectory.get("messages") or []:
        response = ((message.get("extra") or {}).get("response") or {}) if isinstance(message, dict) else {}
        if response.get("model"):
            return str(response["model"])
    return None


def cost_summary_from_trajectory(trajectory: dict[str, Any], elapsed_s: float | None = None,
                                 failed_attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The episode's stage. A retried task reports every attempt it paid for:
    the successful trajectory plus the trajectories of attempts that ended
    without a deliverable, which mini still writes out."""
    usage = usage_from_trajectory(trajectory, elapsed_s)
    for failed in failed_attempts or ():
        extra = usage_from_trajectory(failed)
        for key in ("cost", "total_tokens", "prompt_tokens", "completion_tokens",
                    "cached_prompt_tokens", "reasoning_tokens", "requests", "uncosted_requests"):
            usage[key] = usage.get(key, 0) + extra.get(key, 0)
        if extra.get("cost_source"):
            usage["cost_source"] = extra["cost_source"]
    usage["attempts"] = 1 + len(failed_attempts or ())
    return {STAGE: usage}
