from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    Generator,
    Optional,
)

from prismadv.llm_backend.callbacks.gemini_info import GeminiCallbackHandler
from prismadv.llm_backend.callbacks.openai_info import OpenAICallbackHandler

openai_callback_var: ContextVar[Optional[OpenAICallbackHandler]] = ContextVar(
    "openai_callback", default=None
)

gemini_callback_var: ContextVar[Optional[GeminiCallbackHandler]] = ContextVar(
    "gemini_callback", default=None
)


@contextmanager
def get_openai_callback() -> Generator[OpenAICallbackHandler, None, None]:
    """Get the OpenAI callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        OpenAICallbackHandler: The OpenAI callback handler.

    Example:
        >>> with get_openai_callback() as cb:
        ...     # Use the OpenAI callback handler
    """
    cb = OpenAICallbackHandler()
    openai_callback_var.set(cb)
    yield cb
    openai_callback_var.set(None)


@contextmanager
def get_gemini_callback() -> Generator[GeminiCallbackHandler, None, None]:
    """Get the Gemini callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        GeminiCallbackHandler: The Gemini callback handler.

    Example:
        >>> with get_gemini_callback() as cb:
        ... # Use the Gemini callback handler
    """
    cb = GeminiCallbackHandler()
    gemini_callback_var.set(cb)
    yield cb
    gemini_callback_var.set(None)
