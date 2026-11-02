"""mini-swe-agent episodes are priced from their tokens on the project price list."""
from workflow_prismadv.utils.swe_agent_usage import model_from_trajectory, usage_from_trajectory


def _trajectory(model, calls):
    return {
        "info": {"model_stats": {"instance_cost": 0.999, "api_calls": len(calls)},
                 "config": {"model": {"model_name": model}}},
        "messages": [{"role": "assistant", "extra": {"response": {"usage": {
            "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
            "prompt_tokens_details": {"cached_tokens": cached}}}}} for p, c, cached in calls],
    }


def test_known_model_is_priced_from_tokens_not_from_minis_figure():
    usage = usage_from_trajectory(_trajectory("gemini/gemini-3.7-flash", [(1000, 100, 0), (2000, 50, 1000)]), 12.0)
    assert usage["requests"] == 2 and usage["prompt_tokens"] == 3000 and usage["cached_prompt_tokens"] == 1000
    # 2000 fresh * 0.75 + 1000 cached * 0.075 + 150 out * 3.75, per 1M
    assert abs(usage["cost"] - (2000 * 0.75 + 1000 * 0.075 + 150 * 3.75) / 1e6) < 1e-12
    assert usage["uncosted_requests"] == 0 and "cost_source" not in usage and usage["elapsed_s"] == 12.0


def test_unknown_model_falls_back_to_minis_figure_and_says_so():
    usage = usage_from_trajectory(_trajectory("openai/some-future-model", [(10, 5, 0)]))
    assert usage["cost"] == 0.999 and usage["cost_source"].startswith("mini-swe-agent")
    assert usage["uncosted_requests"] == 0  # litellm did price it, just not our list


def test_model_is_read_from_config_then_from_responses():
    assert model_from_trajectory(_trajectory("openai/gpt-5", [])) == "openai/gpt-5"
    assert model_from_trajectory({"messages": [{"extra": {"response": {"model": "gpt-5-2025-08-07"}}}]}) == "gpt-5-2025-08-07"
    assert model_from_trajectory({}) is None


def test_failed_attempts_are_billed_with_the_successful_one():
    from workflow_prismadv.utils.swe_agent_usage import cost_summary_from_trajectory
    ok = _trajectory("gemini/gemini-3.7-flash", [(1000, 100, 0)])
    failed = _trajectory("gemini/gemini-3.7-flash", [(500, 50, 0), (500, 50, 0)])
    stage = cost_summary_from_trajectory(ok, 30.0, [failed])["constraint_generation"]
    assert stage["attempts"] == 2 and stage["requests"] == 3 and stage["prompt_tokens"] == 2000
    assert abs(stage["cost"] - (2000 * 0.75 + 200 * 3.75) / 1e6) < 1e-12
    assert stage["elapsed_s"] == 30.0
    assert cost_summary_from_trajectory(ok)["constraint_generation"]["attempts"] == 1
