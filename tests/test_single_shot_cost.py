"""Single-shot cost: a failed call still reports its bill, and the bill waits for the script's next run."""
import pytest

from prismadv.llm.langchain.models.prismadv.costs import run_with_cost
from prismadv.llm_backend.llm_pricing_info import price_tokens
from workflow_prismadv.eid_bench_baselines.pipelines.run_codex import append_failed_attempts, cost_all_attempts, read_failed_attempts
from workflow_prismadv.eid_bench_baselines.pipelines.run_single_shot import (
    failure_log_path,
    failure_record,
    make_output_filename,
    usd_without_cache,
)

ENTRY = {"cost": 0.0162435, "prompt_tokens": 7683, "completion_tokens": 2795, "cached_prompt_tokens": 0,
         "reasoning_tokens": 1916, "requests": 3, "elapsed_s": 7.8}


def test_run_with_cost_records_usage_when_the_call_raises():
    summary = {}

    def broken():
        raise ValueError("not JSON")

    with pytest.raises(ValueError):
        run_with_cost("constraint_generation", broken, summary)
    assert summary["constraint_generation"]["requests"] == 0
    assert "elapsed_s" in summary["constraint_generation"]


def test_no_cache_price_matches_the_logged_price_when_nothing_was_cached():
    assert usd_without_cache(ENTRY, "gemini-3.7-flash") == round(price_tokens("gemini-3.7-flash", 7683, 2795, 0), 6)
    assert usd_without_cache(ENTRY, "gemini-3.7-flash") == pytest.approx(ENTRY["cost"], abs=1e-6)
    assert usd_without_cache(ENTRY, "no-such-model") is None


def test_failure_record_carries_the_bill_in_the_codex_shape():
    record = failure_record(ENTRY, "gemini-3.7-flash", KeyError("constraints"), 9.1234)
    assert record["outcome"] == "KeyError: 'constraints'"
    assert record["llm_calls"] == 3 and record["usd"] == ENTRY["cost"]
    assert record["wall_clock_seconds"] == 9.123
    # A failure before any call (no usage recorded) is logged with no bill.
    assert failure_record({}, "gemini-3.7-flash", RuntimeError("spark"), 1.0)["usd"] is None


def test_failed_scripts_are_charged_to_the_next_run_of_the_same_replicate(tmp_path):
    log = failure_log_path(tmp_path, "gemini-3.7-flash", 2)
    assert log.name == ".single_shot--gemini-3.7-flash--r2.failed_attempts.jsonl"
    assert failure_log_path(tmp_path, "gemini-3.7-flash", 1) != log
    assert make_output_filename("gemini-3.7-flash", 2).startswith("single_shot--gemini-3.7-flash--r2--")
    append_failed_attempts(log, [failure_record(ENTRY, "gemini-3.7-flash", ValueError("x"), 1.0)])
    prior = read_failed_attempts(log)
    total = cost_all_attempts(0.01, 0.01, prior)
    assert total["attempts"] == 2 and total["usd"] == round(0.01 + ENTRY["cost"], 6)


def test_few_shot_keeps_its_own_failure_log(tmp_path):
    few = failure_log_path(tmp_path, "gpt-6-sol-medium", 1, "few_shot")
    assert few.name == ".few_shot--gpt-6-sol-medium.failed_attempts.jsonl"
    assert few != failure_log_path(tmp_path, "gpt-6-sol-medium", 1)
