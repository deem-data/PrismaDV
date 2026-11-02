"""LLM usage accounting for the DSPy-based optimization runs."""

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import dspy

from prismadv.llm_backend.llm_pricing_info import price_tokens

CACHE_KEY_IGNORED = ["api_key", "api_base", "base_url"]
COMPLETION_FN_BY_MODEL_TYPE = {
    "chat": "dspy.clients.lm.litellm_completion",
    "text": "dspy.clients.lm.litellm_text_completion",
    "responses": "dspy.clients.lm.litellm_responses_completion",
}

CACHE_MODEL_ALIASES = {"gemini/gemini-3.8-flash": "gemini/gemini-3.7-flash"}


def alias_cache_keys(cache) -> None:
    """Make `cache` (a dspy Cache) key a request of an aliased model as the request of the model it replaces."""
    key = cache.cache_key

    def aliased_key(request: dict[str, Any], ignored_args_for_cache_key: list[str] | None = None) -> str:
        if request.get("model") in CACHE_MODEL_ALIASES:
            request = {**request, "model": CACHE_MODEL_ALIASES[request["model"]]}
        return key(request, ignored_args_for_cache_key)

    cache.cache_key = aliased_key


def cache_request(request: dict[str, Any], model_type: str = "chat") -> dict[str, Any]:
    """The dict DSPy actually hashes for `request`: the request plus the function tag."""
    return {**request, "_fn_identifier": COMPLETION_FN_BY_MODEL_TYPE[model_type]}


def cached_usage(request: dict[str, Any]) -> dict | None:
    """The usage block DSPy stored for `request`, or None when it is not cached.

    `Cache.get` deep-copies the stored response and sets `usage = {}` on the
    copy; the object in the store still carries the tokens the original call
    used. Read it back directly, memory first then disk, without going through
    `get` so nothing is blanked.
    """
    cache = getattr(dspy, "cache", None)
    if cache is None:
        return None
    try:
        key = cache.cache_key(request, CACHE_KEY_IGNORED)
    except Exception:  # noqa: BLE001
        return None
    stored = None
    if getattr(cache, "enable_memory_cache", False) and key in cache.memory_cache:
        stored = cache.memory_cache[key]
    elif getattr(cache, "enable_disk_cache", False) and key in cache.disk_cache:
        stored = cache.disk_cache[key]
    usage = getattr(stored, "usage", None) if stored is not None else None
    if not usage:
        return None
    return dict(usage)

TASK_LM = "task_lm"
REFLECTION_LM = "reflection_lm"

# Mirrors the guard in dspy.clients.lm: OpenAI's reasoning models are rejected
# unless temperature is exactly 1.0 and max_tokens is at least 16000. The
# pattern is DSPy's, not the API's -- the same models answer at other
# temperatures through LangChain -- so we match DSPy's rule to stay runnable
# under it rather than trying to argue with it.
_REASONING_MODEL = re.compile(r"^(?:o[1345]|gpt-5)(?:-(?:mini|nano))?")
_REASONING_TEMPERATURE = 1.0
_REASONING_MIN_MAX_TOKENS = 16000


def is_reasoning_model(model: str) -> bool:
    """True when DSPy will demand reasoning-model parameters for this model."""
    family = model.split("/")[-1].lower() if "/" in model else model.lower()
    return bool(_REASONING_MODEL.match(family))


def apply_reasoning_requirements(model: str, kwargs: dict) -> dict:
    """Coerce kwargs in place to what DSPy accepts; return what was changed.

    The project standard is temperature 0.6, which a reasoning model cannot
    honour. Rather than let every arm crash on construction, raise only the
    parameters that must move and report them, so the deviation stays visible
    in the run log instead of becoming an undocumented difference between arms.
    """
    if not is_reasoning_model(model):
        return {}

    overrides = {}
    temperature = kwargs.get("temperature", _REASONING_TEMPERATURE)
    if temperature != _REASONING_TEMPERATURE:
        overrides["temperature"] = (temperature, _REASONING_TEMPERATURE)
        kwargs["temperature"] = _REASONING_TEMPERATURE

    max_tokens = kwargs.get("max_tokens", _REASONING_MIN_MAX_TOKENS)
    if max_tokens < _REASONING_MIN_MAX_TOKENS:
        overrides["max_tokens"] = (max_tokens, _REASONING_MIN_MAX_TOKENS)
        kwargs["max_tokens"] = _REASONING_MIN_MAX_TOKENS

    if overrides:
        changes = ", ".join(f"{k} {old} -> {new}" for k, (old, new) in overrides.items())
        print(f"[usage] {model} is a reasoning model; DSPy requires {changes}")
    return overrides


def blank_usage() -> dict:
    return {
        "cost": 0.0,
        "cost_spent": 0.0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "reasoning_tokens": 0,
        "requests": 0,
        "cache_served_requests": 0,
        # Calls priced by neither the project price list nor litellm: a cache hit
        # whose stored usage could not be recovered, and models missing from
        # both. Reported so a low cost is never mistaken for a cheap run when it
        # is really an unpriced one.
        "uncosted_requests": 0,
        # Calls the project list could not price and litellm's own figure was
        # used for instead -- a model that needs an entry in llm_pricing_info.
        "fallback_priced_requests": 0,
        # Summed API latency of the calls this execution made. Under
        # SIFTA_NUM_THREADS > 1 this exceeds the round's wall-clock, which the
        # runner records separately.
        "api_seconds": 0.0,
        # Summed latency the cache-served requests took when they were first
        # made (workflow_sifta/runtime/cache_store.py keeps it with the cache): the API
        # time the cache saved. A hit without a stored latency adds nothing.
        "api_seconds_saved": 0.0,
    }


def _nested(usage: dict, outer: str, inner: str) -> int:
    details = usage.get(outer) or {}
    if isinstance(details, dict):
        return details.get(inner) or 0
    return getattr(details, inner, 0) or 0


class UsageAccumulator:
    """Thread-safe usage totals per LM role, for one optimization run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._by_role: dict[str, dict] = {}

    def reset(self) -> None:
        with self._lock:
            self._by_role.clear()

    def record(self, role: str, entry: dict[str, Any], model: str | None = None,
               replayed_usage: dict | None = None) -> None:
        """Account one history entry. `replayed_usage` is the stored usage of a cache hit."""
        usage = entry.get("usage") or {}
        served_from_cache = bool(getattr(entry.get("response"), "cache_hit", False))
        if served_from_cache and not usage and replayed_usage:
            usage = replayed_usage
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        cached = _nested(usage, "prompt_tokens_details", "cached_tokens")

        # Priced from the project list first, so the DSPy rows land on the same
        # price sheet as every other method. litellm's own figure is only a
        # fallback for a model the list does not know, and is counted as such.
        model = entry.get("model") or model
        cost = price_tokens(model, prompt_tokens, completion_tokens, cached) if usage else None
        fallback = cost is None and bool(entry.get("cost"))
        if fallback:
            cost = float(entry["cost"])

        with self._lock:
            totals = self._by_role.setdefault(role, blank_usage())
            totals["cost"] += cost or 0.0
            if not served_from_cache:
                totals["cost_spent"] += cost or 0.0
            totals["total_tokens"] += total_tokens
            totals["prompt_tokens"] += prompt_tokens
            totals["completion_tokens"] += completion_tokens
            totals["cached_prompt_tokens"] += cached
            totals["reasoning_tokens"] += _nested(usage, "completion_tokens_details", "reasoning_tokens")
            totals["requests"] += 1
            if served_from_cache:
                totals["cache_served_requests"] += 1
            if fallback:
                totals["fallback_priced_requests"] += 1
            elif cost is None:
                totals["uncosted_requests"] += 1

    def add_latency(self, role: str, seconds: float, saved: bool = False) -> None:
        with self._lock:
            self._by_role.setdefault(role, blank_usage())["api_seconds_saved" if saved else "api_seconds"] += seconds

    def summary(self) -> dict:
        """Per-role totals plus a `total` row, rounded for JSON."""
        with self._lock:
            by_role = {role: dict(totals) for role, totals in self._by_role.items()}

        total = blank_usage()
        for totals in by_role.values():
            for key in total:
                total[key] += totals[key]

        out = {**by_role, "total": total}
        for totals in out.values():
            totals["cost"] = round(totals["cost"], 6)
            totals["cost_spent"] = round(totals["cost_spent"], 6)
            totals["api_seconds"] = round(totals["api_seconds"], 3)
            totals["api_seconds_saved"] = round(totals["api_seconds_saved"], 3)
        return out


ACCUMULATOR = UsageAccumulator()


class TrackedLM(dspy.LM):
    """A dspy.LM that reports what it spent -- and what a cache hit would have cost."""

    def __init__(self, model: str, *args, usage_role: str = TASK_LM,
                 usage_accumulator: UsageAccumulator | None = None, **kwargs):
        self.reasoning_overrides = apply_reasoning_requirements(model, kwargs)
        super().__init__(model, *args, **kwargs)
        self.usage_role = usage_role
        self.usage_accumulator = usage_accumulator or ACCUMULATOR
        # `forward` and the `update_history` it leads to run on the same thread
        # inside one `__call__`; under num_threads > 1 each thread has its own.
        self._replay = threading.local()

    def _request_as_cached(self, prompt, messages, kwargs) -> dict:
        """The request dict DSPy hashes, rebuilt the way `LM.forward` builds it."""
        kwargs = dict(kwargs)
        kwargs.pop("cache", None)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)
        return cache_request(dict(model=self.model, messages=messages, **kwargs), self.model_type)

    def _note_replay(self, results, prompt, messages, kwargs) -> None:
        self._replay.usage = None
        if getattr(results, "cache_hit", False) and not getattr(results, "usage", None):
            self._replay.usage = cached_usage(self._request_as_cached(prompt, messages, kwargs))

    def _note_latency(self, results, prompt, messages, kwargs, seconds: float) -> None:
        """A fresh call: record its latency, and keep it with the cache under the call's key. A cache hit: record
        the latency the original call took as saved."""
        store = _latency_store()
        key = None
        if store is not None:
            try:
                key = dspy.cache.cache_key(self._request_as_cached(prompt, messages, kwargs), CACHE_KEY_IGNORED)
            except Exception:  # noqa: BLE001
                key = None
        if getattr(results, "cache_hit", False):
            original = store.get(key) if key is not None else None
            if original:
                self.usage_accumulator.add_latency(self.usage_role, original, saved=True)
        else:
            self.usage_accumulator.add_latency(self.usage_role, seconds)
            if key is not None:
                store.set(key, seconds)

    def forward(self, prompt=None, messages=None, **kwargs):
        started = time.perf_counter()
        results = super().forward(prompt=prompt, messages=messages, **kwargs)
        self._note_latency(results, prompt, messages, kwargs, time.perf_counter() - started)
        self._note_replay(results, prompt, messages, kwargs)
        return results

    async def aforward(self, prompt=None, messages=None, **kwargs):
        started = time.perf_counter()
        results = await super().aforward(prompt=prompt, messages=messages, **kwargs)
        self._note_latency(results, prompt, messages, kwargs, time.perf_counter() - started)
        self._note_replay(results, prompt, messages, kwargs)
        return results

    def update_history(self, entry):
        super().update_history(entry)
        replayed = getattr(self._replay, "usage", None)
        self._replay.usage = None
        self.usage_accumulator.record(self.usage_role, entry, model=self.model, replayed_usage=replayed)


_ROLLOUT_ID = None


def set_rollout_id(rollout_id: int | None) -> None:
    """Make every LM built afterwards carry this DSPy rollout_id, which is part of the cache key: replicate N of an
    experiment shares cached answers between its arms, and different replicates sample independently."""
    global _ROLLOUT_ID
    _ROLLOUT_ID = rollout_id


# A replicate seed k > 0 reruns a split with fresh LLM samples: it shifts the rollout ids (and SIFTA's sampler seed)
# by these offsets. k = 0 is the original run.
REPLICATE_SEED_OFFSET = 1000          # task LM rollout id and SIFTA seed: split + 1000 k
REFLECTION_SEED_OFFSET = 100000       # reflection LM rollout id: split * 100 + round + 1 + 100000 k


def replicate_rollout_id(split_seed: int, replicate_seed: int = 0) -> int:
    """The task LM rollout_id of split `split_seed` under replicate seed `replicate_seed` (the split for k = 0)."""
    return split_seed + REPLICATE_SEED_OFFSET * replicate_seed


def tracked_lm(model: str, *, usage_role: str = TASK_LM, **kwargs) -> TrackedLM:
    """Drop-in replacement for dspy.LM that accounts for its own usage."""
    if _ROLLOUT_ID is not None:
        kwargs.setdefault("rollout_id", _ROLLOUT_ID)
    return TrackedLM(model, usage_role=usage_role, **kwargs)


_LATENCY = None
_LATENCY_LOCK = threading.Lock()


def _latency_store():
    """The API seconds of each fresh call, kept next to DSPy's disk cache (None without a disk cache)."""
    global _LATENCY
    cache = getattr(dspy, "cache", None)
    if cache is None or not getattr(cache, "enable_disk_cache", False):
        return None
    with _LATENCY_LOCK:
        if _LATENCY is None:
            from diskcache import Cache

            _LATENCY = Cache(str(Path(os.environ["DSPY_CACHEDIR"]) / "latency"))
    return _LATENCY


def close_latency_store() -> None:
    global _LATENCY
    with _LATENCY_LOCK:
        if _LATENCY is not None:
            _LATENCY.close()
            _LATENCY = None


def usage_summary() -> dict:
    return ACCUMULATOR.summary()


def reset_usage() -> None:
    ACCUMULATOR.reset()
