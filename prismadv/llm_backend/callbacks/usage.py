"""One LangChain callback that counts tokens and prices them for every provider."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import register_configure_hook

from prismadv.llm_backend.llm_pricing_info import price_tokens


class UsageCallbackHandler(BaseCallbackHandler):
    """Token and dollar totals for every LLM call made while it is active."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    prompt_tokens_cached: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    successful_requests: int = 0
    uncosted_requests: int = 0
    total_cost: float = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._model_by_run: dict[UUID, str] = {}

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\t\tPrompt Tokens Cached: {self.prompt_tokens_cached}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"\t\tReasoning Tokens: {self.reasoning_tokens}\n"
            f"Successful Requests: {self.successful_requests} ({self.uncosted_requests} unpriced)\n"
            f"Total Cost (USD): ${self.total_cost}"
        )

    @property
    def always_verbose(self) -> bool:
        return True

    # -- model name, captured at start because some providers omit it at the end

    def _remember_model(self, serialized: Any, run_id: UUID | None, kwargs: dict) -> None:
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or params.get("model_id")
        if not model and isinstance(serialized, dict):
            model = (serialized.get("kwargs") or {}).get("model") or (serialized.get("kwargs") or {}).get("model_name")
        if model and run_id is not None:
            with self._lock:
                self._model_by_run[run_id] = str(model)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *,
                     run_id: UUID | None = None, **kwargs: Any) -> None:
        self._remember_model(serialized, run_id, kwargs)

    def on_chat_model_start(self, serialized: dict[str, Any], messages: Any, *,
                            run_id: UUID | None = None, **kwargs: Any) -> None:
        self._remember_model(serialized, run_id, kwargs)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass

    # -- accounting

    def on_llm_end(self, response: LLMResult, *, run_id: UUID | None = None, **kwargs: Any) -> None:
        usage_metadata = None
        response_metadata: dict = {}
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None
        if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
            usage_metadata = generation.message.usage_metadata
            response_metadata = generation.message.response_metadata or {}

        prompt_tokens_cached = reasoning_tokens = 0
        if usage_metadata:
            prompt_tokens = int(usage_metadata.get("input_tokens") or 0)
            completion_tokens = int(usage_metadata.get("output_tokens") or 0)
            total_tokens = int(usage_metadata.get("total_tokens") or (prompt_tokens + completion_tokens))
            prompt_tokens_cached = int((usage_metadata.get("input_token_details") or {}).get("cache_read") or 0)
            reasoning_tokens = int((usage_metadata.get("output_token_details") or {}).get("reasoning") or 0)
        else:
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage") or {}
            if not token_usage:
                with self._lock:
                    self.successful_requests += 1
                    self._model_by_run.pop(run_id, None)
                return
            prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
            completion_tokens = int(token_usage.get("completion_tokens") or 0)
            total_tokens = int(token_usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        model_name = (response_metadata.get("model_name")
                      or (response.llm_output or {}).get("model_name"))
        with self._lock:
            started_as = self._model_by_run.pop(run_id, None)
        model_name = model_name or started_as

        cost = price_tokens(model_name, prompt_tokens, completion_tokens, prompt_tokens_cached)
        with self._lock:
            if cost is None:
                self.uncosted_requests += 1
            else:
                self.total_cost += cost
            self.total_tokens += total_tokens
            self.prompt_tokens += prompt_tokens
            self.prompt_tokens_cached += prompt_tokens_cached
            self.completion_tokens += completion_tokens
            self.reasoning_tokens += reasoning_tokens
            self.successful_requests += 1

    def __copy__(self) -> "UsageCallbackHandler":
        return self

    def __deepcopy__(self, memo: Any) -> "UsageCallbackHandler":
        return self


usage_callback_var: ContextVar[Optional[UsageCallbackHandler]] = ContextVar(
    "prismadv_usage_callback", default=None
)
register_configure_hook(usage_callback_var, True)


@contextmanager
def get_usage_callback() -> Generator[UsageCallbackHandler, None, None]:
    """Count and price every LLM call made inside the block.

    Example:
        >>> with get_usage_callback() as cb:
        ...     chain.invoke(...)
        >>> cb.total_cost, cb.successful_requests
    """
    handler = UsageCallbackHandler()
    token = usage_callback_var.set(handler)
    try:
        yield handler
    finally:
        usage_callback_var.reset(token)
