"""A codex exec attempt that passes its time limit is killed and not retried."""
import subprocess

import pytest

from workflow_prismadv.utils import codex_runner
from workflow_prismadv.utils.codex_runner import CodexTimeoutError, run_codex_call


def test_timeout_raises_and_does_not_retry(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
    with pytest.raises(CodexTimeoutError):
        run_codex_call("task", work_dir=tmp_path, codex_home=tmp_path, timeout_seconds=5)
    assert calls == [5]


def test_default_timeout_is_passed_to_subprocess(tmp_path, monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
    with pytest.raises(CodexTimeoutError):
        run_codex_call("task", work_dir=tmp_path, codex_home=tmp_path)
    assert seen["timeout"] == codex_runner.TIMEOUT_SECONDS
