"""Codex cost: the no-cache price, and failed attempts charged to the script that finally succeeds."""
import subprocess

import pytest

from workflow_prismadv.eid_bench_baselines.pipelines.run_codex import (
    append_failed_attempts,
    cost_all_attempts,
    failed_attempts_path,
    read_failed_attempts,
)
from workflow_prismadv.utils import codex_runner
from workflow_prismadv.utils.codex_runner import CodexTimeoutError, cost_without_cache, estimate_cost, run_codex_call
from prismadv.llm_backend.llm_pricing_info import price_tokens


def test_cost_without_cache_prices_every_input_token_fresh():
    accounting = {"input_tokens": 1_000_000, "cached_input_tokens": 900_000, "output_tokens": 10_000}
    assert cost_without_cache(accounting, "gpt-5.6-sol") == round(price_tokens("gpt-5.6-sol", 1_000_000, 10_000, 0), 6)
    assert cost_without_cache(accounting, "gpt-5.6-sol") > estimate_cost(accounting, "gpt-5.6-sol")["usd"]
    assert cost_without_cache(accounting, "no-such-model") is None


def test_failed_attempts_wait_beside_the_artifact_and_add_up(tmp_path):
    artifact = tmp_path / "codex--gpt-5.6-sol--medium.yaml"
    log = failed_attempts_path(artifact)
    assert log.name == ".codex--gpt-5.6-sol--medium.failed_attempts.jsonl"
    append_failed_attempts(log, [{"attempt": 1, "usd": 0.2, "usd_without_cache": 1.0}])
    append_failed_attempts(log, [{"attempt": 1, "usd": 0.3, "usd_without_cache": 1.5}])
    failed = read_failed_attempts(log)
    assert [f["usd"] for f in failed] == [0.2, 0.3]
    total = cost_all_attempts(0.5, 2.0, failed)
    assert total == {"attempts": 3, "usd": 1.0, "usd_without_cache": 4.5,
                     "failed_usd": 0.5, "failed_usd_without_cache": 2.5}


def test_a_killed_attempt_is_recorded_on_the_exception(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
    with pytest.raises(CodexTimeoutError) as caught:
        run_codex_call("task", work_dir=tmp_path, codex_home=tmp_path, timeout_seconds=5)
    [record] = caught.value.failed_attempts
    assert record["attempt"] == 1 and record["outcome"].startswith("CodexTimeoutError")
