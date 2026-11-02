import time
from contextlib import contextmanager
from typing import Awaitable
from typing import Callable, Any

from prismadv.llm_backend.callbacks.usage import get_usage_callback


@contextmanager
def _track_usage():
    # One handler for every provider, priced from prismadv.llm_backend.llm_pricing_info.
    with get_usage_callback() as cb:
        yield cb


def run_with_cost(key: str, fn: Callable[[], Any], cost_summary: dict) -> Any:
    # Recorded in `finally` so a call that raises still reports what it billed.
    with _track_usage() as cb:
        started = time.perf_counter()
        try:
            return fn()
        finally:
            cost_summary[key] = _get_cost_dict(cb, time.perf_counter() - started)


async def arun_with_cost(key: str, coro_fn: Callable[[], "Awaitable[Any]"], cost_summary: dict) -> Any:
    with _track_usage() as cb:
        started = time.perf_counter()
        result = await coro_fn()
        elapsed = time.perf_counter() - started
        cost_summary[key] = _get_cost_dict(cb, elapsed)
    return result


def _get_cost_dict(cb, elapsed_s):
    return {
        "cost": cb.total_cost,
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "cached_prompt_tokens": getattr(cb, "prompt_tokens_cached", 0),
        "reasoning_tokens": getattr(cb, "reasoning_tokens", 0),
        "requests": getattr(cb, "successful_requests", 0),
        # Calls whose model has no entry in the price list. Reported so a low
        # cost is never mistaken for a cheap stage when it is an unpriced one.
        "uncosted_requests": getattr(cb, "uncosted_requests", 0),
        "elapsed_s": round(elapsed_s, 3),
    }
