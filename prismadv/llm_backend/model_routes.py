"""Where a bare model name is sent, and with which key.

The project names models without a provider (`gpt-5`, `gemini-3.7-flash`) and
every artifact is keyed that way. Each backend wants something else: LangChain
picks a client class, litellm (mini-swe-agent, PocketFlow, DSPy) wants a
`provider/model` route, and Google's LangChain client reads `GOOGLE_API_KEY`
while litellm and the project's `.env` use `GEMINI_API_KEY`. This is the one
place that knows those conventions.
"""
from __future__ import annotations

import os
import re

# The project names a reasoning-effort variant by suffixing the model
# (`gpt-5-low`, `gpt-6-sol-medium`); the provider only knows the base model
# and takes the effort as a request parameter.
_EFFORT_SUFFIX = re.compile(r"-(minimal|low|medium|high|xhigh)$")


def split_effort(model_name: str) -> tuple[str, str | None]:
    """`(base model, reasoning effort)`; the effort is None when the name has no suffix."""
    match = _EFFORT_SUFFIX.search(model_name)
    if not match:
        return model_name, None
    return model_name[:match.start()], match.group(1)


def litellm_completion_kwargs(model_name: str) -> dict[str, str]:
    """Extra `litellm.completion` arguments the project name implies (the reasoning effort)."""
    _, effort = split_effort(model_name)
    return {"reasoning_effort": effort} if effort else {}


def provider_for(model_name: str) -> str:
    """`openai`, `gemini` or `anthropic` for a bare or already-routed model name."""
    if "/" in model_name:
        return model_name.split("/", 1)[0]
    if model_name.startswith("gemini-"):
        return "gemini"
    if model_name.startswith("claude-"):
        return "anthropic"
    return "openai"


def litellm_route(model_name: str) -> str:
    """The `provider/model` string litellm-based tools take; a routed name passes through.

    An effort suffix is dropped here: it is not part of the provider's model id
    (see `litellm_completion_kwargs` for where it goes instead).
    """
    if "/" in model_name:
        return model_name
    base, _ = split_effort(model_name)
    return f"{provider_for(base)}/{base}"


def google_api_key() -> str | None:
    """The Gemini key under either name Google's SDKs look for."""
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
