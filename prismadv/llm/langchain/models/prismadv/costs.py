from typing import Awaitable
from typing import Callable, Any

from langchain_community.callbacks import get_openai_callback


def run_with_cost(key: str, fn: Callable[[], Any], cost_summary: dict) -> Any:
    with get_openai_callback() as cb:
        result = fn()
        cost_summary[key] = _get_cost_dict(cb)
    return result


async def arun_with_cost(key: str, coro_fn: Callable[[], "Awaitable[Any]"], cost_summary: dict) -> Any:
    with get_openai_callback() as cb:
        result = await coro_fn()
        cost_summary[key] = _get_cost_dict(cb)
    return result


def _get_cost_dict(cb):
    return {
        "cost": cb.total_cost,
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
    }
