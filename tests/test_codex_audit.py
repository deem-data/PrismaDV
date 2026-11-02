"""The codex workspace audit: what counts as reaching outside the folder."""
from pathlib import Path

from workflow_prismadv.utils.codex_runner import (
    SCOPE_RULE,
    audit_events,
    outside_workspace_hits,
    with_scope_rule,
)


def _cmd(text: str) -> dict:
    return {"type": "item.completed", "item": {"type": "command_execution", "command": text}}


def _run(text: str, work_dir: Path, **kwargs) -> list[str]:
    return outside_workspace_hits([_cmd(text)], work_dir, **kwargs)


def test_workspace_and_system_paths_are_fine(tmp_path):
    ws = tmp_path / "codex_ws"
    ws.mkdir()
    assert _run(f"/usr/bin/bash -lc 'sed -n 1,3p {ws}/data/tables/t.csv 2>/dev/null'", ws) == []
    assert _run("python deequ_check.py --table t --constraints c.txt", ws) == []
    assert _run("rg -n 'foo' . | head", ws) == []


def test_search_of_the_workspace_parent_is_flagged(tmp_path):
    ws = tmp_path / "codex_ws"
    ws.mkdir()
    # The case this check exists for: the parent of a /tmp workspace is shared.
    hits = _run("/usr/bin/bash -lc 'rg -n \"constraints.json\" . /tmp 2>/dev/null | head'", ws)
    assert len(hits) == 1 and hits[0].startswith("[/tmp]")


def test_absolute_paths_elsewhere_are_flagged(tmp_path):
    ws = tmp_path / "codex_ws"
    ws.mkdir()
    hits = _run("rg --files /sc-scratch/some-project | head", ws)
    assert hits and "/sc-scratch/some-project" in hits[0]


def test_home_and_parent_traversal_are_flagged(tmp_path):
    ws = tmp_path / "codex_ws"
    ws.mkdir()
    assert _run("ls ~/", ws)[0].startswith("[~]")
    assert _run("cat $HOME/.kaggle/kaggle.json", ws)[0].startswith("[~]")
    assert _run("cd .. && ls", ws)[0].startswith("[..]")
    assert _run("ls ../", ws)[0].startswith("[..]")
    # A version range or an ellipsis in prose is not a traversal.
    assert _run("echo 'rows 1..3 done...'", ws) == []


def test_allowed_roots_and_staged_tokens_are_skipped(tmp_path):
    ws = tmp_path / "codex_ws"
    venv = tmp_path / "repo" / ".venv"
    (venv / "bin").mkdir(parents=True)
    ws.mkdir()
    assert _run(f"{venv}/bin/python -c 'import pydeequ'", ws, allowed_roots=(venv,)) == []
    assert _run(f"cat {tmp_path}/repo/README.md", ws, allowed_roots=(venv,))
    staged = frozenset({"/Users/someone/data/t.csv"})
    assert _run("grep -c '/Users/someone/data/t.csv' task.txt", ws, staged=staged) == []


def test_only_commands_are_inspected(tmp_path):
    ws = tmp_path / "codex_ws"
    ws.mkdir()
    # A file the agent legitimately read may mention any path in its output.
    event = {"type": "item.completed",
             "item": {"type": "command_execution", "command": "cat task.txt",
                      "aggregated_output": "see /sc-scratch/elsewhere and /tmp/other"}}
    assert outside_workspace_hits([event], ws) == []
    reasoning = {"type": "item.completed", "item": {"type": "reasoning", "text": "look in /tmp"}}
    assert outside_workspace_hits([reasoning], ws) == []


def test_forbidden_component_audit_still_scans_outputs(tmp_path):
    event = {"type": "item.completed",
             "item": {"type": "command_execution", "command": "rg foo /sc-scratch",
                      "aggregated_output": "/sc-scratch/repo/data_processed/x.yaml:1: foo"}}
    assert audit_events([event], ("data_processed",)) == ["[data_processed] rg foo /sc-scratch"]


def test_scope_rule_names_the_shared_parent():
    rendered = with_scope_rule("# Task\n\nWrite constraints.")
    assert rendered.endswith(SCOPE_RULE.strip() + "\n")
    for phrase in ("/tmp", "`..`", "$HOME", "discarded"):
        assert phrase in SCOPE_RULE


def test_llm_calls_come_from_the_rollout_when_there_is_one(tmp_path):
    import json
    from workflow_prismadv.utils.codex_runner import find_rollout, rollout_calls, thread_id_from_events, token_accounting

    events = [{"type": "thread.started", "thread_id": "abc-123"},
              {"type": "item.completed", "item": {"type": "agent_message"}},
              {"type": "item.completed", "item": {"type": "command_execution"}},
              {"type": "item.completed", "item": {"type": "command_execution"}},
              {"type": "turn.completed", "usage": {"input_tokens": 300, "cached_input_tokens": 100,
                                                    "output_tokens": 30, "reasoning_output_tokens": 5}}]
    home = tmp_path / "home"
    rollout = home / "sessions" / "2026" / "09" / "03" / "rollout-2026-09-03T10-00-00-abc-123.jsonl"
    rollout.parent.mkdir(parents=True)
    lines = [{"type": "session_meta", "payload": {}},
             {"type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": {
                 "input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10, "reasoning_output_tokens": 0, "total_tokens": 110}}}},
             {"type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": {
                 "input_tokens": 200, "cached_input_tokens": 100, "output_tokens": 20, "reasoning_output_tokens": 5, "total_tokens": 220}}}},
             {"type": "response_item", "payload": {"type": "message"}}]
    rollout.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    assert thread_id_from_events(events) == "abc-123"
    assert find_rollout(home, "abc-123") == rollout
    assert find_rollout(home, "other") is None
    assert [c["input_tokens"] for c in rollout_calls(rollout)] == [100, 200]

    exact = token_accounting(events, rollout)
    assert exact["llm_calls"] == 2 and exact["llm_calls_source"] == "rollout token_count events"
    assert exact["llm_calls_estimate"] == 3 and exact["turns"] == 1
    assert sum(c["input_tokens"] for c in exact["calls"]) == exact["input_tokens"] == 300

    estimate_only = token_accounting(events)          # an --ephemeral run: no rollout
    assert estimate_only["llm_calls"] is None and estimate_only["llm_calls_estimate"] == 3


def test_slashes_inside_expressions_are_not_paths(tmp_path):
    ws = tmp_path / "codex_ws"
    ws.mkdir()
    assert _run("rg --files -g '!/.git' .", ws) == []
    assert _run("python - <<'PY'\nprint(df.Poster_Link.str.fullmatch(r'https://.+').mean())\nPY", ws) == []
    assert _run("python - <<'PY'\nD.Poster_Link.str.startswith('https://m.media-amazon.com/')\nPY", ws) == []
    assert _run("python - <<'PY'\nx = (df['a'] > 1).sum()/df['b'].eq(1).sum()\nPY", ws) == []


def test_paths_inside_python_are_still_flagged(tmp_path):
    ws = tmp_path / "codex_ws"
    ws.mkdir()
    for code in ("open('/tmp/other/constraints.json')", 'Path("/sc-scratch/x")',
                 "os.listdir( /home/someone )", "pd.read_csv(p,sep=',',x=[1,'/tmp/y'])"):
        hits = _run(f"python - <<'PY'\n{code}\nPY", ws)
        assert hits, code
    assert _run("cat /etc/hosts | rg -n x /tmp", ws)[0].startswith("[/tmp]")
