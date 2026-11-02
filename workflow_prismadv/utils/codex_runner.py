"""Invoke the Codex CLI headlessly and recover the constraints it writes."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from prismadv.llm_backend.llm_pricing_info import describe, price_tokens

DEFAULT_MODEL = "gpt-6-sol"
DEFAULT_CODEX_HOME = Path.home() / ".codex-eidbench"
TASK_INSTRUCTION = "Solve the task specified in task.txt."
MAX_RETRIES = 3

TIMEOUT_SECONDS = 1800

DEFAULT_EFFORT_LABEL = "default"

ISOLATION_BWRAP = "bwrap"
ISOLATION_AUDIT = "audit"

DEFAULT_ISOLATION = ISOLATION_AUDIT

SCOPE_RULE = """
# Workspace boundary

Your workspace is the current folder, and for this task nothing else exists.

- Read, list, search and write only inside this folder: `./data`, `./code`,
  `deequ_check.py`, `task.txt`, and whatever you create here.
- Do not touch anything outside it. That means no parent directory (`..`), no
  other folder under /tmp, no home directory (`~`, $HOME), and no absolute path
  anywhere else on the filesystem.
- Do not search the wider filesystem -- no `rg`, `grep`, `find`, `ls` or
  `python` over `/`, `/tmp`, `..`, `~` or any path outside this folder -- for
  related data, prompts, examples, other runs, or previous answers. There is
  nothing out there you are allowed to use, and a run that looks is discarded.
- Running `python`, `deequ_check.py` and the tools on your PATH is fine; the
  boundary is about what you read and write, not which interpreter runs.

Everything you need is here. Work only from it.
"""

SYSTEM_ROOTS = ("/dev", "/proc", "/sys", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt")


def with_scope_rule(prompt_content: str) -> str:
    return prompt_content.rstrip() + "\n\n" + SCOPE_RULE.strip() + "\n"

DEFAULT_REASONING_EFFORT = "medium"

PATH_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.~-]*(?:/[A-Za-z0-9_.~-]+)+/?")

# Staged files scanned to learn which paths the agent was handed.
STAGED_SCAN_SUFFIXES = (
    ".txt", ".py", ".md", ".json", ".yaml", ".yml", ".sql", ".ipynb", ".cfg", ".toml", ".csv",
)
STAGED_SCAN_MAX_BYTES = 4_000_000


class WorkspaceEscapeError(RuntimeError):
    """The agent reached outside its workspace, so the run cannot be trusted.

    Raised instead of returning constraints: an artifact produced by a run that
    may have read the answer key is worse than no artifact, because nothing
    downstream can tell the two apart.
    """

WORKSPACE_TABLES_REL = "data/tables"
DEEQU_CHECKER_NAME = "deequ_check.py"
DEEQU_JARS_NAME = "deequ_jars.txt"
DEEQU_CHECKER_SOURCE = '''"""Run PyDeequ constraint code against the example tables and report results."""
import argparse
import json
import os
import sys
from pathlib import Path

import prismadv  # noqa: F401

os.environ.pop("OPENAI_API_KEY", None)

import pandas as pd  # noqa: E402
from prismadv.dq_manager import DeequDataQualityManager  # noqa: E402

HERE = Path(__file__).resolve().parent
TABLES_DIR = HERE / "data" / "tables"
JARS_FILE = HERE / "deequ_jars.txt"


def build_spark():
    """A Spark session whose Deequ classpath is pinned to already-resolved jars.

    The default session config asks Spark to fetch `spark.jars.packages` through
    Ivy. That cannot work here: Ivy writes its resolution cache under $HOME, and
    inside the agent's sandbox only the workspace is writable, so resolution dies
    before the JVM starts and every check reports FAILS for a reason that has
    nothing to do with the constraint. The jars were resolved once by the driver
    that staged this file; naming them directly removes the resolution step, and
    with it the network and $HOME-write dependencies.
    """
    from pyspark.sql import SparkSession

    jars = [line.strip() for line in JARS_FILE.read_text(encoding="utf-8").split() if line.strip()]
    if not jars:
        raise RuntimeError(f"{JARS_FILE.name} is empty; the Deequ classpath cannot be pinned")
    # `spark.jars` takes URIs; the driver classpath takes plain filesystem paths.
    local = [jar[len("file://"):] if jar.startswith("file://") else jar for jar in jars]
    return (
        SparkSession.builder.appName("deequ-local")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.jars", ",".join(jars))
        .config("spark.driver.extraClassPath", ":".join(local))
        .config("spark.executor.extraClassPath", ":".join(local))
        .config("spark.driver.host", "localhost")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def load_codes(path: Path) -> list[str]:
    """Accept JSON, or one constraint per line so quoting never blocks a check."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.lstrip()[0] in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            print(f"warning: {path.name} is not valid JSON ({exc}); "
                  f"falling back to one-constraint-per-line parsing.")
            payload = None
        if payload is not None:
            if isinstance(payload, dict):
                payload = payload.get("constraints", [])
            codes = []
            for item in payload:
                if isinstance(item, str):
                    codes.append(item)
                else:
                    codes.append(item.get("code_for_constraint") or item.get("suggestion"))
            return [code for code in codes if code]
    codes = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        codes.append(line)
    return codes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, help="Table name (file stem under data/tables).")
    parser.add_argument("--constraints", required=True,
                        help='JSON: ["<code>", ...] or {"constraints": [{"code_for_constraint": ...}]}')
    args = parser.parse_args()

    csv_path = TABLES_DIR / f"{args.table}.csv"
    if not csv_path.exists():
        available = sorted(p.stem for p in TABLES_DIR.glob("*.csv"))
        print(f"no such table: {args.table}. available: {available}")
        return 2

    codes = load_codes(Path(args.constraints))
    if not codes:
        print("no constraint code found in the given file")
        return 2

    dataframe = pd.read_csv(csv_path)
    manager = DeequDataQualityManager()
    spark_df, spark = manager.spark_df_from_pandas_df(dataframe, spark_session=build_spark())
    results = manager.validate_constraints_with_reasons(spark, spark_df, codes, isolated_check=True)

    held = 0
    for code, (is_valid, reason) in zip(codes, results):
        held += int(bool(is_valid))
        print(f"[{'HOLDS' if is_valid else 'FAILS'}] {code}")
        if not is_valid:
            print(f"          reason: {reason}")
    print(f"\\n{held}/{len(codes)} constraint(s) hold on table '{args.table}' "
          f"({len(dataframe)} rows).")

    try:
        spark.sparkContext._gateway.shutdown_callback_server()
    except Exception:
        pass
    try:
        spark.stop()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
'''


def effort_label(reasoning_effort: str | None) -> str:
    """Filename token for the reasoning-effort dimension."""
    return reasoning_effort or DEFAULT_EFFORT_LABEL


def write_deequ_checker(work_dir: Path, deequ_jars: tuple[str, ...] = ()) -> Path:
    """Drop the PyDeequ runner beside the staged tables so the agent can execute checks.

    `deequ_jars` is the classpath the checker pins itself to; without it the
    checker would fall back to Ivy resolution, which cannot succeed inside the
    agent's sandbox. Refused rather than defaulted: a checker staged without a
    classpath fails on every invocation the agent makes, and that failure reads
    downstream as an agent that chose not to verify its constraints.
    """
    work_dir = Path(work_dir)
    if not deequ_jars:
        raise ValueError(
            "write_deequ_checker needs the resolved Deequ jar paths; without them the "
            "staged runner cannot start a Spark session inside the agent sandbox"
        )
    (work_dir / DEEQU_JARS_NAME).write_text("\n".join(deequ_jars) + "\n", encoding="utf-8")
    checker_path = work_dir / DEEQU_CHECKER_NAME
    checker_path.write_text(DEEQU_CHECKER_SOURCE, encoding="utf-8")
    return checker_path


def resolved_deequ_jars(spark) -> tuple[str, ...]:
    """The Deequ classpath Ivy resolved for a live session, as URIs.

    Read off the driver's own session rather than globbing the Ivy cache: the
    cache holds every jar ever resolved on this machine, while `spark.jars` is
    exactly the set this Deequ/Spark pair asked for.
    """
    jars = spark.sparkContext.getConf().get("spark.jars") or ""
    resolved = tuple(jar.strip() for jar in jars.split(",") if jar.strip())
    if not resolved:
        raise RuntimeError(
            "the driver's Spark session reports no resolved jars, so the agent's staged "
            "runner has no Deequ classpath to pin to"
        )
    return resolved


@lru_cache(maxsize=1)
def deequ_classpath() -> tuple[str, ...]:
    """The Deequ classpath, for a caller with no Spark session of its own to read.

    Ivy resolution is the expensive part and its answer does not change within a
    process, so it happens once. The session is deliberately left running rather
    than stopped: `getOrCreate` hands it back to whatever the caller builds next,
    and stopping it here would only force a second, identical resolution.
    """
    import pandas as pd

    from prismadv.dq_manager import DeequDataQualityManager

    _, spark = DeequDataQualityManager().spark_df_from_pandas_df(pd.DataFrame({"_probe": [0]}))
    return resolved_deequ_jars(spark)


def assert_no_answer_key(work_dir: Path, forbidden_parts: tuple[str, ...]) -> None:
    """Fail loudly if staging put a path holding the benchmark's answer key in reach."""
    work_dir = Path(work_dir)
    leaked = [
        path
        for path in work_dir.rglob("*")
        if any(part in forbidden_parts for part in path.relative_to(work_dir).parts)
    ]
    if leaked:
        raise RuntimeError(f"answer-key files staged into the agent workspace: {leaked[:5]}")


def codex_version() -> str:
    """Record the CLI build, which pins agent behaviour for reproducibility."""
    try:
        completed = subprocess.run(
            ["codex", "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL
        )
        return completed.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def codex_command(model_name: str, work_dir: Path, reasoning_effort: str | None = None,
                  *, network_access: bool = False) -> list[str]:
    command = [
        "codex",
        "exec",
        "--model", model_name,
        "-C", str(work_dir),
        "--sandbox", "workspace-write",
        "--skip-git-repo-check",
        # Not --ephemeral: the persisted rollout under CODEX_HOME/sessions is the
        # only record of the individual model requests (one `token_count` event
        # each); the --json stream reports usage once per turn.
        "--ignore-user-config",
        "--color", "never",
        "--json",
    ]
    if network_access:
        # Not for the agent's benefit -- for the JVM's. codex's workspace-write
        # sandbox blocks socket creation outright, and PySpark talks to its own
        # JVM over a loopback socket, so with this off the Deequ runner dies at
        # `findLocalInetAddress` before any constraint is ever evaluated. The
        # variants without a staged runner leave it off, so the extra reach is
        # confined to the rows that cannot work without it.
        command += ["-c", "sandbox_workspace_write.network_access=true"]
    if reasoning_effort:
        command += ["-c", f'model_reasoning_effort="{reasoning_effort}"']
    command.append(TASK_INSTRUCTION)
    return command


def find_bwrap() -> Path | None:
    """Locate bubblewrap: the system one, else the copy the Codex CLI ships."""
    system = shutil.which("bwrap")
    if system:
        return Path(system)
    codex = shutil.which("codex")
    if not codex:
        return None
    # `codex` on PATH is a symlink into the package, so resolving it lands on the
    # package root; search that first since it is small, then the install root.
    for root in (Path(codex).resolve().parent.parent, Path(codex).parent.parent):
        for candidate in root.glob("**/codex-resources/bwrap"):
            if candidate.is_file():
                return candidate
    return None


def _resolve_existing(paths) -> list[Path]:
    seen, out = set(), []
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw).resolve()
        if path.exists() and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def contain_command(
    command: list[str],
    *,
    work_dir: Path,
    bwrap: Path,
    readable: tuple[Path | str, ...] = (),
    writable: tuple[Path | str, ...] = (),
    readable_at: tuple[tuple[Path | str, Path | str], ...] = (),
) -> list[str]:
    """Wrap `command` in a mount namespace holding only what the run needs.

    The benchmark tree is not bound in, so the answer key is not merely unread
    but absent: paths under it fail with ENOENT no matter how the agent builds
    them. This is what makes the deequ variants comparable to the prompt-only
    one, whose isolation rests on the agent having no reason to look.

    Note the interpreter is bound at its own path rather than by binding its
    parent, so a venv living inside the benchmark repo exposes itself and not
    its siblings.
    """
    argv = [str(bwrap)]
    for system_dir in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"):
        if Path(system_dir).exists():
            argv += ["--ro-bind", system_dir, system_dir]
    argv += [
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--tmpfs", str(Path.home()),
    ]
    for path in _resolve_existing(readable):
        argv += ["--ro-bind", str(path), str(path)]
    # (source, destination) pairs, for material the run needs to find somewhere
    # other than where it lives -- the Ivy cache belongs under $HOME, and $HOME
    # inside the namespace is the workspace.
    for source, destination in readable_at:
        source = Path(source).resolve()
        if source.exists():
            argv += ["--ro-bind", str(source), str(destination)]
    for path in _resolve_existing([work_dir, *writable]):
        argv += ["--bind", str(path), str(path)]
    argv += [
        "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--share-net",          # the model API is remote; the filesystem is what we are fencing
        "--die-with-parent",
        "--setenv", "HOME", str(work_dir),
        "--chdir", str(work_dir),
        "--",
    ]
    return argv + command


def build_command(
    model_name: str,
    work_dir: Path,
    reasoning_effort: str | None = None,
    *,
    isolation: str = DEFAULT_ISOLATION,
    readable: tuple[Path | str, ...] = (),
    writable: tuple[Path | str, ...] = (),
    readable_at: tuple[tuple[Path | str, Path | str], ...] = (),
    network_access: bool = False,
) -> list[str]:
    """The codex invocation, contained unless containment was explicitly declined."""
    command = codex_command(model_name, work_dir, reasoning_effort,
                            network_access=network_access)
    if isolation == ISOLATION_AUDIT:
        return command
    if isolation != ISOLATION_BWRAP:
        raise ValueError(f"unknown isolation mode: {isolation!r}")
    bwrap = find_bwrap()
    if bwrap is None:
        raise RuntimeError(
            "isolation='bwrap' requested but bubblewrap was not found (looked on PATH and "
            "inside the Codex CLI install). Install bubblewrap, or pass isolation='audit' "
            "to fall back to post-hoc detection -- which does not contain reads, and so "
            "weakens any row that stages data into the workspace."
        )
    # Bind the whole Node install: the agent runs `codex`, which is a symlink in
    # its bin/ executed by the node binary sitting beside it.
    codex_path = shutil.which("codex")
    tool_roots = [Path(codex_path).parent.parent] if codex_path else []
    return contain_command(
        command,
        work_dir=work_dir,
        bwrap=bwrap,
        readable=tuple([*tool_roots, *readable]),
        writable=writable,
        readable_at=readable_at,
    )


class CodexTimeoutError(RuntimeError):
    """One codex exec attempt ran past its time limit and was killed.

    Not retried: a run that hit the limit once will most likely hit it again,
    and every retry bills another full attempt.
    """


class DeequToolBrokenError(RuntimeError):
    """The agent reached for the staged runner and the runner never answered.

    Raised instead of returning constraints, on the same reasoning as
    `WorkspaceEscapeError`: an artifact from a run whose only verification tool
    was dead is worse than no artifact, because downstream it is identical to a
    run where the agent simply chose not to verify. That is precisely how the
    first codex_percolumn sweep recorded 35 runs and 2418 failed checks without
    anything noticing -- every run exited 0 and wrote a well-formed file.
    """
DEEQU_VERDICT_MARKERS = ("[HOLDS]", "[FAILS]")


def deequ_tool_health(events: list[dict[str, Any]]) -> dict[str, Any]:
    """How the staged PyDeequ runner actually behaved in the agent's hands.

    Read off the trajectory rather than probed beforehand. A pre-run probe has to
    reconstruct the agent's sandbox from the outside and is only ever an
    approximation of it; the trajectory *is* the sandbox, observed. It also costs
    nothing, and it keeps working when the CLI's sandbox flags change under us.
    """
    invocations = verdicts = 0
    failures: list[str] = []
    for event in events:
        item = event.get("item") or {}
        if item.get("type") != "command_execution" or event.get("type") != "item.completed":
            continue
        if DEEQU_CHECKER_NAME not in (item.get("command") or ""):
            continue
        output = item.get("aggregated_output") or ""
        # Reading or printing the file is not running it; only a verdict counts.
        if not any(marker in output for marker in DEEQU_VERDICT_MARKERS):
            if item.get("exit_code"):
                invocations += 1
                failures.append(output.strip()[-400:])
            continue
        invocations += 1
        verdicts += sum(output.count(marker) for marker in DEEQU_VERDICT_MARKERS)
    return {
        "invocations": invocations,
        "verdict_lines": verdicts,
        "usable": invocations == 0 or verdicts > 0,
        "failure_tail": failures[-1] if failures and not verdicts else None,
    }


def assert_deequ_tool_usable(health: dict[str, Any]) -> None:
    """Refuse a run in which the agent tried to check its work and could not.

    Deliberately silent when the agent never invoked the runner at all: the
    prompt offers the tool rather than mandating it, so declining to use it is a
    result about the agent. Failing on every use is a result about the harness,
    and those two must not land in the same artifact looking alike.
    """
    if health["usable"]:
        return
    raise DeequToolBrokenError(
        f"the agent invoked {DEEQU_CHECKER_NAME} {health['invocations']} time(s) and never "
        f"got a verdict back, so every constraint it 'verified' was checked against a "
        f"broken tool. Refusing to write constraints.\n"
        f"--- last runner output (tail) ---\n{health.get('failure_tail')}"
    )


def parse_events(stdout: str) -> list[dict[str, Any]]:
    """Each --json line is one event; non-JSON progress lines are dropped."""
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def usage_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("type") == "turn.completed" and "usage" in event:
            return event["usage"]
    return {}


# Field names differ across CLI versions; accept any spelling for each quantity.
_USAGE_ALIASES = {
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "cached_input_tokens": ("cached_input_tokens", "cache_read_input_tokens",
                            "cached_tokens", "input_tokens_cached"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "reasoning_tokens": ("reasoning_tokens", "reasoning_output_tokens"),
    "total_tokens": ("total_tokens",),
}


def _dig(payload: Any, names: tuple[str, ...]) -> int | None:
    """Find the first of `names` anywhere in a nested usage payload."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in names and isinstance(value, (int, float)):
                return int(value)
        for value in payload.values():
            found = _dig(value, names)
            if found is not None:
                return found
    return None


def thread_id_from_events(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") == "thread.started" and event.get("thread_id"):
            return str(event["thread_id"])
    return None


def find_rollout(codex_home: Path | str, thread_id: str) -> Path | None:
    """The session rollout codex wrote for `thread_id`, or None (e.g. an --ephemeral run)."""
    sessions = Path(codex_home) / "sessions"
    if not sessions.is_dir():
        return None
    matches = sorted(sessions.rglob(f"rollout-*-{thread_id}.jsonl"))
    return matches[-1] if matches else None


def rollout_calls(rollout_path: Path | str) -> list[dict[str, Any]]:
    """One usage dict per model request, read off the rollout's `token_count` events.

    Each event carries `last_token_usage` for the response that just came
    back, so their number is the number of API calls the agent made and their
    sum matches the turn total the --json stream reports.
    """
    calls: list[dict[str, Any]] = []
    with open(rollout_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload") or {}
            if record.get("type") == "event_msg" and payload.get("type") == "token_count":
                last = (payload.get("info") or {}).get("last_token_usage")
                if isinstance(last, dict):
                    calls.append({name: last.get(name) for name in (
                        "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                        "output_tokens", "reasoning_output_tokens", "total_tokens")})
    return calls


def token_accounting(events: list[dict[str, Any]], rollout_path: Path | str | None = None) -> dict[str, Any]:
    """Per-turn and total token counts for one codex call.

    Summed across every `turn.completed`, because a run that hits a retry or a
    continuation reports more than one and the last alone would undercount.
    Cached input is tracked separately from fresh input: it is billed at a
    different rate, so a total that blends them cannot be priced.
    """
    turns = [event["usage"] for event in events
             if event.get("type") == "turn.completed" and "usage" in event]
    totals = {name: 0 for name in _USAGE_ALIASES}
    seen = {name: False for name in _USAGE_ALIASES}
    for usage in turns:
        for name, aliases in _USAGE_ALIASES.items():
            value = _dig(usage, aliases)
            if value is not None:
                totals[name] += value
                seen[name] = True
    accounting = {name: (totals[name] if seen[name] else None) for name in _USAGE_ALIASES}
    accounting["turns"] = len(turns)
    fresh = accounting["input_tokens"]
    cached = accounting["cached_input_tokens"]
    # Whether `input_tokens` already includes the cached portion varies by
    # version, so record the ratio rather than asserting a decomposition.
    accounting["cache_hit_rate"] = (
        round(cached / fresh, 4) if fresh and cached is not None and fresh > 0 else None
    )
    accounting["raw_turns"] = turns
    accounting.update(activity_counts(events))
    # Exact request count from the rollout when there is one; otherwise the
    # item-based estimate (one assistant message, tool call or file edit per
    # model response, which undercounts only when a response batches tool calls).
    calls = rollout_calls(rollout_path) if rollout_path else []
    accounting["llm_calls"] = len(calls) if calls else None
    accounting["llm_calls_source"] = "rollout token_count events" if calls else "estimate from items"
    accounting["llm_calls_estimate"] = (accounting["agent_messages"] + accounting["command_executions"]
                                        + accounting["file_changes"])
    accounting["calls"] = calls
    accounting["rollout_path"] = str(rollout_path) if rollout_path else None
    return accounting


def activity_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    """How much work the agent did, as far as the event stream reveals it.

    The --json stream reports usage once per turn; the request count comes
    from the session rollout (see `rollout_calls`). These are the observable
    proxies in the stream itself -- assistant messages, tool invocations, file
    edits -- named for what they count.
    """
    item_types: dict[str, int] = {}
    for event in events:
        item = event.get("item") or {}
        kind = item.get("type")
        if kind and event.get("type") == "item.completed":
            item_types[kind] = item_types.get(kind, 0) + 1
    return {
        "agent_messages": item_types.get("agent_message", 0),
        "command_executions": item_types.get("command_execution", 0),
        "file_changes": item_types.get("file_change", 0),
        "item_counts": item_types,
    }


def estimate_cost(accounting: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Dollar cost from the project price list, or an explicit "unpriced".

    Absent an entry the token counts still stand on their own; the artifact
    says so rather than carrying a $0 that reads as cheap.
    """
    rates = describe(model_name)
    if not rates:
        return {"usd": None, "priced": False,
                "reason": f"no price for {model_name!r} in prismadv.llm_backend.llm_pricing_info"}
    usd = price_tokens(
        model_name,
        accounting.get("input_tokens") or 0,
        accounting.get("output_tokens") or 0,
        accounting.get("cached_input_tokens") or 0,
    )
    return {"usd": round(usd, 6), "priced": True, "rates": rates,
            "source": f"{rates['source']} (as of {rates['as_of']})"}


def cost_without_cache(accounting: dict[str, Any], model_name: str) -> float | None:
    """Dollar cost with every input token at the fresh-input rate.

    Codex resends its growing conversation each turn, so most input is served
    from the provider's prompt cache and billed at the cached rate. This is the
    same tokens priced as if nothing were cached, the basis on which the
    methods are compared.
    """
    if not describe(model_name):
        return None
    return round(price_tokens(model_name, accounting.get("input_tokens") or 0,
                              accounting.get("output_tokens") or 0, 0), 6)


def attempt_record(events: list[dict[str, Any]], codex_home: Path | str, model_name: str,
                   *, attempt: int, outcome: str, seconds: float | None) -> dict[str, Any]:
    """Tokens and cost of one codex exec attempt, read off its rollout when there is one."""
    thread_id = thread_id_from_events(events)
    rollout = find_rollout(codex_home, thread_id) if thread_id else None
    accounting = token_accounting(events, rollout)
    return {
        "attempt": attempt,
        "outcome": outcome[:500],
        "wall_clock_seconds": round(seconds, 3) if seconds is not None else None,
        "rollout_path": str(rollout) if rollout else None,
        "llm_calls": accounting.get("llm_calls") or accounting.get("llm_calls_estimate"),
        "input_tokens": accounting.get("input_tokens"),
        "cached_input_tokens": accounting.get("cached_input_tokens"),
        "output_tokens": accounting.get("output_tokens"),
        "reasoning_tokens": accounting.get("reasoning_tokens"),
        "usd": estimate_cost(accounting, model_name).get("usd"),
        "usd_without_cache": cost_without_cache(accounting, model_name),
    }


class CodexAttemptsFailed(RuntimeError):
    """Every attempt failed; `failed_attempts` holds what each one billed."""

    def __init__(self, message: str, failed_attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.failed_attempts = failed_attempts


def codex_env(
    codex_home: Path | str | None = None,
    *,
    path_prepend: Path | str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Auth comes from CODEX_HOME, so the API key is kept out of the agent's shell."""
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home or env.get("CODEX_HOME") or DEFAULT_CODEX_HOME)
    env.pop("OPENAI_API_KEY", None)
    if path_prepend is not None:
        env["PATH"] = f"{path_prepend}{os.pathsep}{env.get('PATH', '')}"
    # Passed explicitly rather than left to the project's .env: under containment
    # the repo is not in the namespace, so dotenv finds nothing to load and any
    # variable the staged runner needs has to arrive through the environment.
    env.update(extra or {})
    return env


def staged_path_tokens(work_dir: Path) -> frozenset[str]:
    """Path tokens contained in the material staged for the agent to read.

    A path the agent was handed cannot evidence an escape, and the prompt names
    plenty: `task.txt` embeds the project source and the absolute path of the
    clean data it came from. Snapshot this before the run, never after -- a file
    the agent writes mid-run is not something we gave it.
    """
    tokens: set[str] = set()
    for path in Path(work_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in STAGED_SCAN_SUFFIXES:
            continue
        try:
            if path.stat().st_size > STAGED_SCAN_MAX_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        tokens.update(PATH_TOKEN_PATTERN.findall(text))
    return frozenset(tokens)


def forbidden_path_hits(
    blob: str, forbidden: tuple[str, ...], staged: frozenset[str] = frozenset()
) -> list[str]:
    """Which forbidden entries appear as path components somewhere in `blob`.

    Each entry is a path fragment, matched against the components of a
    path-shaped token rather than against the raw text. Bare substring matching
    cannot express this: `.env` occurs inside `os.environ`, which the staged
    PyDeequ runner contains, and `corrupted` occurs in any sentence about
    corrupt rows -- both of which are things the agent is meant to read.

    Tokens in `staged` are skipped for the same reason: they came from the
    prompt and the staged files, so seeing one back in an event says only that
    the agent read what it was given.
    """
    needles = [tuple(part for part in entry.strip("/").split("/") if part) for entry in forbidden]
    matched = set()
    for token in PATH_TOKEN_PATTERN.findall(blob):
        if token in staged:
            continue
        parts = [part for part in token.split("/") if part]
        for entry, needle in zip(forbidden, needles):
            width = len(needle)
            if any(tuple(parts[i:i + width]) == needle for i in range(len(parts) - width + 1)):
                matched.add(entry)
    return sorted(matched)


def audit_events(
    events: list[dict[str, Any]],
    forbidden: tuple[str, ...],
    staged: frozenset[str] = frozenset(),
) -> list[str]:
    """Flag any event that mentions a path the agent has no business touching.

    Scans the serialised event rather than only `command_execution.command`:
    a file read that never becomes a shell command would otherwise be invisible
    to this check entirely.

    This is detection, not containment -- a path built from a variable or
    reached with a bare `cd ..` still slips through. Under ISOLATION_BWRAP it is
    the second line of defence behind a mount namespace that removes the
    benchmark tree outright; under ISOLATION_AUDIT it is the only one, and
    should be read as a floor on leakage rather than a guarantee.
    """
    hits = []
    for event in events:
        blob = json.dumps(event, ensure_ascii=False, default=str)
        matched = forbidden_path_hits(blob, forbidden, staged)
        if not matched:
            continue
        item = event.get("item") or {}
        detail = str(item.get("command") or item.get("type") or event.get("type", "event"))
        hits.append(f"[{','.join(matched)}] {detail[:240]}")
    return hits

_PARENT_TRAVERSAL = re.compile(r"(?:^|[\s\"'=:(])(?:\.\./|\.\.(?=[\s\"')]|$))|\bcd\s+\.\.")
_HOME_REFERENCE = re.compile(r"(?:^|[\s\"'=:(])~(?:/|(?=[\s\"')]|$))|\$HOME\b|\$\{HOME\}")

_PATH_LEAD = set(" \t\n\r\"'`=([,:;&|<>{")


def command_text(event: dict[str, Any]) -> str:
    """The shell text of a command_execution event, or '' for anything else."""
    item = event.get("item") or {}
    if item.get("type") != "command_execution":
        return ""
    command = item.get("command")
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command or "")


def _under(path: str, roots: tuple[str, ...]) -> bool:
    parts = [p for p in path.split("/") if p and p != "."]
    for root in roots:
        root_parts = [p for p in root.split("/") if p]
        if parts[:len(root_parts)] == root_parts:
            return True
    return False


def outside_workspace_hits(
    events: list[dict[str, Any]],
    work_dir: Path | str,
    allowed_roots: tuple[Path | str, ...] = (),
    staged: frozenset[str] = frozenset(),
) -> list[str]:
    work = str(Path(work_dir).resolve())
    roots = (work, *SYSTEM_ROOTS, *(str(Path(r).resolve()) for r in allowed_roots))
    hits = []
    for event in events:
        text = command_text(event)
        if not text:
            continue
        reasons = []
        for match in PATH_TOKEN_PATTERN.finditer(text):
            token = match.group(0)
            if not token.startswith("/") or token in staged:
                continue
            if match.start() > 0 and text[match.start() - 1] not in _PATH_LEAD:
                continue
            if not _under(token, roots):
                reasons.append(token)
        if _HOME_REFERENCE.search(text):
            reasons.append("~")
        if _PARENT_TRAVERSAL.search(text):
            reasons.append("..")
        if reasons:
            hits.append(f"[{','.join(dict.fromkeys(reasons))}] {text[:240]}")
    # A command shows up as a started and a completed event; report it once.
    return list(dict.fromkeys(hits))


def run_codex_call(
    prompt_content: str,
    *,
    model_name: str = DEFAULT_MODEL,
    work_dir: Path,
    codex_home: Path | str | None = None,
    path_prepend: Path | str | None = None,
    audit_forbidden: tuple[str, ...] = (),
    reasoning_effort: str | None = None,
    isolation: str = DEFAULT_ISOLATION,
    readable: tuple[Path | str, ...] = (),
    writable: tuple[Path | str, ...] = (),
    readable_at: tuple[tuple[Path | str, Path | str], ...] = (),
    extra_env: dict[str, str] | None = None,
    network_access: bool = False,
    require_deequ_tool: bool = False,
    timeout_seconds: float | None = TIMEOUT_SECONDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop the prompt as task.txt, invoke codex exec, parse constraints.json."""
    work_dir = Path(work_dir)
    (work_dir / "task.txt").write_text(prompt_content, encoding="utf-8")
    (work_dir / "AGENTS.md").write_text(SCOPE_RULE.strip() + "\n", encoding="utf-8")
    constraints_path = work_dir / "constraints.json"
    if constraints_path.exists():
        constraints_path.unlink()

    staged = staged_path_tokens(work_dir)
    env = codex_env(codex_home, path_prepend=path_prepend, extra=extra_env)
    allowed_roots: tuple[Path | str, ...] = tuple(
        root for root in (
            Path(path_prepend).parent if path_prepend else None,
            env["CODEX_HOME"],
            Path.home() / ".ivy2",
            *readable,
            *(src for src, _ in readable_at),
            *(dst for _, dst in readable_at),
        ) if root
    )
    command = build_command(
        model_name, work_dir, reasoning_effort,
        isolation=isolation,
        readable=readable,
        readable_at=readable_at,
        network_access=network_access,
        # Codex writes sessions and logs into CODEX_HOME, so it cannot be read-only.
        writable=(env["CODEX_HOME"], *writable),
    )
    last_err: Exception | None = None
    last_stderr = ""
    attempts = 0
    wall_clock_total = 0.0
    # Every attempt that did not produce the result is still billed; each one
    # is recorded here and travels with the result or with the exception.
    failed_attempts: list[dict[str, Any]] = []
    for _ in range(MAX_RETRIES):
        events: list[dict[str, Any]] = []
        started = time.monotonic()
        try:
            attempts += 1
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                partial = exc.stdout.decode(errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                events = parse_events(partial)
                raise CodexTimeoutError(
                    f"codex exec attempt {attempts} passed the {timeout_seconds:.0f} s limit "
                    f"and was killed; its tokens are billed but not recorded here "
                    f"(see the rollout under {env['CODEX_HOME']}/sessions)."
                ) from exc
            attempt_seconds = time.monotonic() - started
            wall_clock_total += attempt_seconds
            last_stderr = completed.stderr[-2000:]
            events = parse_events(completed.stdout)
            raw_constraints = json.loads(constraints_path.read_text(encoding="utf-8"))["constraints"]
            escapes = audit_events(events, audit_forbidden, staged) if audit_forbidden else []
            outside = outside_workspace_hits(events, work_dir, allowed_roots, staged)
            if escapes or outside:
                detail = "\n".join(f"   {hit}" for hit in (escapes + outside)[:8])
                raise WorkspaceEscapeError(
                    f"{len(escapes)} event(s) named the benchmark tree and {len(outside)} "
                    f"command(s) reached outside the workspace under isolation={isolation!r}; "
                    f"refusing to write constraints.\n{detail}"
                )
            tool_health = deequ_tool_health(events)
            if require_deequ_tool:
                assert_deequ_tool_usable(tool_health)
            thread_id = thread_id_from_events(events)
            rollout = find_rollout(env["CODEX_HOME"], thread_id) if thread_id else None
            accounting = token_accounting(events, rollout)
            trajectory = {
                "codex_version": codex_version(),
                "model": model_name,
                "reasoning_effort": reasoning_effort,
                "command": command,
                "usage": usage_from_events(events),
                "token_accounting": accounting,
                "cost": estimate_cost(accounting, model_name),
                "cost_without_cache_usd": cost_without_cache(accounting, model_name),
                "failed_attempts": failed_attempts,
                "wall_clock_seconds": round(attempt_seconds, 3),
                "wall_clock_seconds_including_retries": round(wall_clock_total, 3),
                "attempts": attempts,
                "timeout_seconds": timeout_seconds,
                "returncode": completed.returncode,
                "isolation": isolation,
                "workspace_escapes": escapes,
                "outside_workspace": outside,
                "deequ_tool_health": tool_health,
                "events": events,
                "stderr_tail": completed.stderr[-2000:],
            }
            return raw_constraints, trajectory
        except (WorkspaceEscapeError, DeequToolBrokenError, CodexTimeoutError) as exc:
            failed_attempts.append(attempt_record(
                events, env["CODEX_HOME"], model_name, attempt=attempts,
                outcome=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - started))
            exc.failed_attempts = failed_attempts
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            failed_attempts.append(attempt_record(
                events, env["CODEX_HOME"], model_name, attempt=attempts,
                outcome=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - started))
    raise CodexAttemptsFailed(
        f"codex exec failed after {MAX_RETRIES} attempts; last error: {last_err}\n"
        f"--- codex stderr tail ---\n{last_stderr}",
        failed_attempts,
    )
