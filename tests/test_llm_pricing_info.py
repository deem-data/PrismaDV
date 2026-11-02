"""The single price list, and the consumers that must all price through it."""
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from prismadv.llm_backend import llm_pricing_info as pricing
from prismadv.llm_backend.callbacks.usage import UsageCallbackHandler, get_usage_callback
from prismadv.llm_backend.llm_pricing_info import PRICES, canonical_model, price_tokens, rates_for


def test_every_entry_states_its_source_and_date():
    for model, entry in PRICES.items():
        assert entry.model == model
        assert entry.source and entry.as_of
        assert entry.input >= 0 and entry.cached_input >= 0 and entry.output >= 0


def test_model_names_resolve_across_providers_and_suffixes():
    assert canonical_model("gpt-5") == "gpt-5"
    assert canonical_model("openai/gpt-5") == "gpt-5"
    assert canonical_model("gpt-5-2025-08-07") == "gpt-5"
    assert canonical_model("gpt-5-mini-2025-08-07") == "gpt-5-mini"
    # The project's reasoning-effort naming bills as the base model.
    assert canonical_model("gpt-5-low") == "gpt-5"
    assert canonical_model("gpt-5-high") == "gpt-5"
    assert canonical_model("gemini/gemini-2.5-pro") == "gemini-2.5-pro"
    assert canonical_model("models/gemini-2.5-flash") == "gemini-2.5-flash"
    assert canonical_model("gpt-5.6-sol") == "gpt-5.6-sol"
    assert canonical_model("gpt-5.2") == "gpt-5.2"
    # A dated snapshot with its own price stays its own entry.
    assert canonical_model("gpt-4o-2024-05-13") == "gpt-4o-2024-05-13"
    assert canonical_model("gpt-4o-2024-08-06") == "gpt-4o"
    assert canonical_model("not-a-model") is None
    assert canonical_model(None) is None


def test_cached_tokens_are_billed_once_at_the_cached_rate():
    # 1000 prompt tokens of which 200 cached, 100 output, on gpt-5.
    usd = price_tokens("gpt-5", 1000, 100, 200)
    assert usd == pytest.approx((800 * 1.25 + 200 * 0.125 + 100 * 10.0) / 1e6)
    # Cached can never exceed the prompt it is part of.
    assert price_tokens("gpt-5", 100, 0, 500) == pytest.approx(100 * 0.125 / 1e6)
    assert price_tokens("no-such-model", 10, 10) is None
    assert price_tokens("gpt-5", 0, 0) == 0.0


def test_the_codex_model_keeps_the_account_rate():
    rates = rates_for("gpt-5.6-sol")
    assert (rates.input, rates.cached_input, rates.output) == (5.0, 0.5, 30.0)
    assert "account price list" in rates.source


def test_codex_estimate_cost_prices_through_the_shared_list():
    from workflow_prismadv.utils.codex_runner import estimate_cost

    accounting = {"input_tokens": 1_000_000, "cached_input_tokens": 500_000, "output_tokens": 10_000}
    result = estimate_cost(accounting, "gpt-5.6-sol")
    assert result["priced"] and result["usd"] == pytest.approx(0.5 * 5.0 + 0.5 * 0.5 + 0.01 * 30.0)
    assert "as of" in result["source"]
    assert estimate_cost(accounting, "unpriced-model")["priced"] is False


def _chat_result(model_name: str | None, input_tokens: int, output_tokens: int, cache_read: int = 0) -> LLMResult:
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "input_token_details": {"cache_read": cache_read}},
        response_metadata={"model_name": model_name} if model_name else {},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_langchain_handler_prices_openai_and_gemini_alike():
    handler = UsageCallbackHandler()
    handler.on_llm_end(_chat_result("gpt-5-2025-08-07", 1000, 100, cache_read=200), run_id=uuid4())
    # Gemini responses may omit the model; it is remembered from the start event.
    run_id = uuid4()
    handler.on_chat_model_start({}, [], run_id=run_id, invocation_params={"model": "models/gemini-2.5-flash"})
    handler.on_llm_end(_chat_result(None, 1000, 100), run_id=run_id)
    assert handler.successful_requests == 2 and handler.uncosted_requests == 0
    expected = ((800 * 1.25 + 200 * 0.125 + 100 * 10.0) + (1000 * 0.30 + 100 * 2.50)) / 1e6
    assert handler.total_cost == pytest.approx(expected)
    assert handler.prompt_tokens == 2000 and handler.prompt_tokens_cached == 200


def test_langchain_handler_counts_unpriced_calls_instead_of_charging_zero():
    handler = UsageCallbackHandler()
    handler.on_llm_end(_chat_result("some-future-model", 10, 10), run_id=uuid4())
    assert handler.uncosted_requests == 1 and handler.total_cost == 0.0
    with get_usage_callback() as cb:
        assert isinstance(cb, UsageCallbackHandler)


def test_module_lists_models(monkeypatch):
    assert "gpt-5" in pricing.known_models() and "gemini-2.5-pro" in pricing.known_models()


def test_gpt_6_sol_medium_prices_as_gpt_6_sol():
    assert canonical_model("gpt-6-sol-medium") == "gpt-6-sol"
    assert canonical_model("gpt-6-sol") != canonical_model("gpt-5.6-sol")
    rates = rates_for("gpt-6-sol-medium")
    assert (rates.input, rates.cached_input, rates.output) == (2.0, 0.2, 10.0)


def test_gpt_6_sol_is_built_without_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from prismadv.llm_backend.entry import get_langchain_model
    sol = get_langchain_model("gpt-6-sol-medium", temperature=0.6)
    assert sol.model_name == "gpt-6-sol" and sol.reasoning == {"effort": "medium"}
    assert sol.temperature is None
    # LangChain itself drops temperature for gpt-5 reasoning models; a plain model keeps it.
    assert get_langchain_model("gpt-5-low", temperature=0.6).temperature is None
    assert get_langchain_model("gpt-4o", temperature=0.6).temperature == 0.6
