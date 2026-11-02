"""Run the Codex CLI agent to produce EIDBench-synth constraints.

Third agentic baseline alongside mini-swe-agent and PocketFlow. The agent gets a
workspace holding the observed table under ``data/``, the program as a ``.py``
file under ``code/``, and a PyDeequ runner, so it can profile the data, read or
run the program, and execute constraint code against the table before settling
on ``constraints.json``.

Unlike the other rows, the prompt is not the shared mini-swe-agent string: it is
read from ``prompts/codex.md``, a document meant to be opened and edited. This
row therefore does *not* hold the prompt fixed against the others -- the prompt
is the variable it exists to move -- so it is read as an ablation on instruction
detail rather than as another scaffold comparison.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import oyaml as yaml

from prismadv.data_models import Constraints
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_real_baselines.single_shot import column_from_code
from workflow_prismadv.eid_bench_baselines.prompts import (
    PROMPT_PATH,
    load_prompt_template,
    render_prompt,
)
from workflow_prismadv.utils.codex_runner import (
    DEEQU_CHECKER_NAME,
    DEFAULT_EFFORT_LABEL,
    DEFAULT_ISOLATION,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    WORKSPACE_TABLES_REL,
    assert_no_answer_key,
    effort_label,
    resolved_deequ_jars,
    run_codex_call,
    with_scope_rule,
    write_deequ_checker,
)

# Trailing field of the pre-deterministic filename scheme, e.g. 20260824_143012.
TIMESTAMP_PATTERN = re.compile(r"\d{8}_\d{6}")

BASELINE_METHOD = "codex"

# Second line of defence, matched as path components rather than as substrings.
# Under the default bwrap isolation none of these exist inside the namespace at
# all; if one still shows up in an event, either containment was declined or an
# assumption behind it is wrong, and the run is rejected rather than recorded.
AUDIT_FORBIDDEN = (
    "benchmarks",
    "data_processed",
    "errors",
    "files_with_corrupted_new_data",
    ".env",
)

FORBIDDEN_PARTS = ("errors", "files_with_corrupted_new_data")
STAGED_TABLE_NAME = "observed_data"
# The program is staged as a file too, assertions stripped exactly as in the
# prompt, so the agent can run or trace it rather than only read it.
CODE_DIRNAME = "code"


def interpreter_support_roots(project_root: Path, venv: Path) -> tuple[Path, ...]:
    """Everything outside the venv that the venv still needs to function.

    A venv is not self-contained here: `bin/python` is a symlink to a uv-managed
    interpreter under $HOME, and the project is installed via a .pth naming the
    repo root, so `prismadv` is imported from the source tree. Both live outside
    the venv directory and would be absent from the namespace, leaving a venv
    whose python silently is not on PATH and whose package does not import.

    Each is bound at its own path, so the benchmark tree around them stays gone.
    """
    roots = [project_root / "prismadv"]
    interpreter = (venv / "bin" / "python").resolve()
    if not interpreter.is_relative_to(venv):
        roots.append(interpreter.parent.parent)
    return tuple(root for root in roots if root.exists())


def resolve_interpreter(project_root: Path) -> tuple[Path, str]:
    """The venv the staged PyDeequ runner needs, plus the SPARK_VERSION it wants.

    Left unchecked this fails silently: a missing venv simply drops off PATH,
    the agent falls back to a system python without pydeequ, and every check it
    runs reports FAILS. The agent then iterates against pure noise and the row
    looks like a genuine bad result.
    """
    venv = project_root / ".venv"
    interpreter = venv / "bin" / "python"
    if not interpreter.exists():
        raise RuntimeError(
            f"the staged runner needs the project venv at {venv} (pyspark + pydeequ), "
            f"but {interpreter} does not exist. Create it with `uv sync` "
            f"before running codex."
        )
    # Existence is not enough: a venv built from the wrong lockfile, or on a
    # Python the project does not support, still has an interpreter here. It
    # would pass a mere exists() check and then fail every constraint at import
    # time inside the agent's workspace, where the failure reads as a bad result
    # rather than a broken environment.
    # pydeequ refuses to import without SPARK_VERSION, and picks its Deequ jar by
    # it, so derive it from the pyspark actually installed rather than hardcoding.
    version_probe = subprocess.run(
        [str(interpreter), "-c",
         "import pyspark; print('.'.join(pyspark.__version__.split('.')[:2]))"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if version_probe.returncode != 0:
        raise RuntimeError(
            f"the venv at {venv} cannot import pyspark, so the staged runner would "
            f"report every constraint as failing:\n{version_probe.stderr.strip()[-400:]}"
        )
    spark_version = version_probe.stdout.strip()
    probe = subprocess.run(
        [str(interpreter), "-c", "import pyspark, pydeequ"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env={**os.environ, "SPARK_VERSION": spark_version},
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"the venv at {venv} cannot import pydeequ under SPARK_VERSION="
            f"{spark_version}, so the staged runner would report every constraint "
            f"as failing:\n{probe.stderr.strip()[-400:]}"
        )
    return venv, spark_version


def make_output_filename(model_name, baseline_method=BASELINE_METHOD,
                         reasoning_effort=None, replicate=None) -> str:
    """Name an artifact by what identifies the run, and nothing else.

    Variant, model and reasoning effort are the dimensions a row is compared
    along, so they name the file. No timestamp: it would fork the identity,
    leaving a rerun sitting next to the run it was meant to replace at every
    processed-data label. When the run happened is provenance, and lives in the
    payload as `created_at`.

    `replicate` is the one case where repeating an identical configuration is
    the point rather than an accident -- measuring run-to-run variance. It is
    therefore part of the identity too, so replicates coexist instead of one
    skipping over another. Replicate 1 keeps the bare name, so a variance study
    extends an existing single run rather than orphaning it.
    """
    suffix = f"--r{replicate}" if replicate and int(replicate) > 1 else ""
    return f"{baseline_method}--{model_name}--{effort_label(reasoning_effort)}{suffix}.yaml"


def staged_code_rel(script_stem: str) -> str:
    """Workspace-relative path the program is staged at."""
    return f"{CODE_DIRNAME}/{script_stem}.py"


def build_detail_prompt(source_code, example_rows, prompt_path=None, script_stem="program") -> str:
    """Render the detail-prompt document with this script's program and data.

    The other variants append their tooling paragraph to a shared string; this
    one hands the same facts to the document as placeholders, so where the
    workspace is described -- and whether it is described before or after the
    program, or twice -- is a decision made by editing the document.
    """
    checker_cmd = (f"python {DEEQU_CHECKER_NAME} --table {STAGED_TABLE_NAME} "
                   f"--constraints <your_constraints_file>")
    return render_prompt(
        {
            "CODE": source_code,
            "CODE_PATH": f"./{staged_code_rel(script_stem)}",
            "DATA_SAMPLE": example_rows,
            "TABLE_PATH": f"./{WORKSPACE_TABLES_REL}/{STAGED_TABLE_NAME}.csv",
            "TABLE_NAME": STAGED_TABLE_NAME,
            "CHECKER_CMD": checker_cmd,
        },
        path=prompt_path,
    )


def warn_if_prompt_changed(artifact_path: Path, prompt_path=None) -> bool:
    """Say so when a skipped artifact was written under different wording.

    A warning rather than an automatic rerun: overwriting results costs money and
    is the caller's decision. But saying nothing is worse than either, because
    the failure is invisible -- you edit the prompt, rerun the sweep, watch it
    print "Skipping" for every script, and read the unchanged numbers as evidence
    that the edit made no difference.

    Compares the template, not the rendered prompt. The rendered text embeds
    `df.sample(3)`, which is redrawn every run, so comparing it would report a
    change every single time and the warning would be worth nothing.
    """
    try:
        with open(artifact_path, encoding="utf-8") as handle:
            recorded = (yaml.safe_load(handle) or {}).get("prompt_template")
    except Exception as exc:  # noqa: BLE001
        print(f"   (could not read the prompt recorded in {artifact_path.name}: {exc})")
        return False
    if recorded is None:
        print(f"   NOTE: {artifact_path.name} predates prompt recording, so whether it used "
              f"the current wording cannot be established. Rerun with --overwrite to be sure.")
        return True
    if recorded.strip() == load_prompt_template(prompt_path).strip():
        return False
    print(f"   WARNING: the prompt has changed since {artifact_path.name} was written, so this "
          f"artifact does not reflect the current wording. Rerun with --overwrite to apply it.")
    return True


def stage_deequ_workspace(observed_data_path: Path, work_dir: Path,
                          deequ_jars: tuple[str, ...],
                          script_stem: str, source_code: str) -> Path:
    """Copy the observed table and the program into the workspace, with the PyDeequ runner.

    `deequ_jars` is the classpath the driver's own Spark session resolved. The
    staged runner pins itself to it, because inside the agent's sandbox Ivy can
    neither write its cache under $HOME nor reach Maven Central.
    """
    tables_dir = work_dir / WORKSPACE_TABLES_REL
    tables_dir.mkdir(parents=True, exist_ok=True)
    staged_table = tables_dir / f"{STAGED_TABLE_NAME}.csv"
    shutil.copy(observed_data_path, staged_table)
    staged_code = work_dir / staged_code_rel(script_stem)
    staged_code.parent.mkdir(parents=True, exist_ok=True)
    staged_code.write_text(source_code, encoding="utf-8")
    write_deequ_checker(work_dir, deequ_jars)
    assert_no_answer_key(work_dir, FORBIDDEN_PARTS)
    return staged_table


_BACKTICKED = re.compile(r"`([^`]+)`")


def column_in_code(code: str, known_columns) -> str | None:
    """The column a check is about, read from the check itself.

    Backticked SQL names first (`.satisfies("`Blood Pressure` RLIKE ...")`),
    then quoted names as the real-benchmark runner reads them, then any known
    column that appears as a whole word in an unquoted SQL expression
    (`.satisfies("Age >= 0", ...)`), longest name first.
    """
    for name in _BACKTICKED.findall(code):
        if name in known_columns:
            return name
    column = column_from_code(code, set(known_columns))
    if column is not None:
        return column
    for name in sorted(known_columns, key=len, reverse=True):
        if re.search(rf"(?<![\w`]){re.escape(name)}(?![\w`])", code):
            return name
    return None


def normalize_raw_constraints(raw_constraints, known_columns) -> tuple[list[dict], dict]:
    """Read constraints.json entries in the formats the staged checker accepts.

    The checker takes a plain code string or a dict with `code_for_constraint`
    (or `suggestion`); the real-benchmark runner reads a missing column from the
    check itself. The synth runner applies the same rules, so an output the
    agent's own tool accepted is not discarded here. A check that names no
    known column (`.hasSize(...)`) is dropped, as in the real runner.
    """
    kept, stats = [], {"entries": len(raw_constraints), "plain_strings": 0,
                       "column_inferred": 0, "dropped_no_column": 0, "dropped_no_code": 0}
    for item in raw_constraints:
        if isinstance(item, str):
            stats["plain_strings"] += 1
            item = {"code_for_constraint": item}
        code = item.get("code_for_constraint") or item.get("suggestion")
        if not code:
            stats["dropped_no_code"] += 1
            continue
        column = item.get("column_name") or item.get("column")
        if column and not isinstance(column, str):
            # A multi-column check named as a list (`["a", "b"]`): read the column from the check, as for a missing one.
            stats["column_not_a_string"] = stats.get("column_not_a_string", 0) + 1
            column = None
        if not column:
            column = column_in_code(code, known_columns)
            if column is None:
                stats["dropped_no_column"] += 1
                continue
            stats["column_inferred"] += 1
        kept.append({**item, "column_name": column, "code_for_constraint": code})
    return kept, stats


def failed_attempts_path(artifact_path: Path) -> Path:
    """Where billed attempts wait for the run of the same script that succeeds."""
    return artifact_path.parent / f".{artifact_path.stem}.failed_attempts.jsonl"


def append_failed_attempts(path: Path, records) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({**record, "logged_at": datetime.now().isoformat(timespec="seconds")}) + "\n")


def read_failed_attempts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cost_all_attempts(accepted_usd, accepted_usd_without_cache, failed: list[dict]) -> dict:
    """The script's full bill: the accepted attempt plus every failed one before it."""
    return {
        "attempts": 1 + len(failed),
        "usd": round((accepted_usd or 0) + sum(f.get("usd") or 0 for f in failed), 6),
        "usd_without_cache": round((accepted_usd_without_cache or 0)
                                   + sum(f.get("usd_without_cache") or 0 for f in failed), 6),
        "failed_usd": round(sum(f.get("usd") or 0 for f in failed), 6),
        "failed_usd_without_cache": round(sum(f.get("usd_without_cache") or 0 for f in failed), 6),
    }


def superseded_artifacts(constraints_path: Path, baseline_method: str, model_name: str,
                         reasoning_effort=None) -> list[Path]:
    """Timestamped artifacts from before filenames became deterministic.

    They share this run's variant and model but carry a trailing timestamp, so
    they would survive a rerun and be read back as an independent result. The
    effort dimension did not exist when they were written, so they only count
    as superseded by a run that also leaves it at the CLI default.
    """
    if effort_label(reasoning_effort) != DEFAULT_EFFORT_LABEL:
        return []
    stale = []
    for path in constraints_path.glob(f"{baseline_method}--{model_name}--*.yaml"):
        suffix = path.stem.split("--")[2:]
        if len(suffix) == 1 and TIMESTAMP_PATTERN.fullmatch(suffix[0]):
            stale.append(path)
    return sorted(stale)


def run_codex(dataset_name, subtask_name, processed_data_label, model_name=DEFAULT_MODEL,
              codex_home=None, script_stems=None,
              reasoning_effort=DEFAULT_REASONING_EFFORT, overwrite=False,
              isolation=DEFAULT_ISOLATION, replicate=None,
              prompt_path=None, dump_prompt=None):
    baseline_method = BASELINE_METHOD
    project_root = get_project_root()
    project_manager = ProjectManager(project_root=project_root, dataset_name=dataset_name)
    dq_manager = DeequDataQualityManager()
    observed_data_path = project_manager.get_observed_data_path(subtask_name, processed_data_label)
    train_data = FileLoader.load_csv(observed_data_path)
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    example_rows = train_data.sample(3).to_dict(orient="records")

    # The runner needs the repo interpreter (pyspark + pydeequ) on PATH. Under
    # containment it is bound in on its own, so the repo root around it stays
    # empty -- PATH names a real interpreter and nothing else.
    interpreter_root, spark_version = resolve_interpreter(project_root)
    path_prepend = interpreter_root / "bin"
    # Read off the session just built above: it has already paid for Ivy
    # resolution, and its `spark.jars` is exactly the classpath this Deequ/Spark
    # pair needs. The agent's copy pins to these paths rather than resolving
    # again, which it could not do inside its sandbox.
    deequ_jars = resolved_deequ_jars(spark_train)

    script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name=subtask_name)
    if script_stems:
        script_path_list = [path for path in script_path_list if path.stem in set(script_stems)]
        if not script_path_list:
            raise ValueError(f"no script in {subtask_name} matches {sorted(set(script_stems))}")
    try:
        _run_scripts(
            script_path_list=script_path_list,
            project_manager=project_manager,
            project_root=project_root,
            dq_manager=dq_manager,
            spark_train=spark_train,
            spark_train_data=spark_train_data,
            observed_data_path=observed_data_path,
            example_rows=example_rows,
            dataset_name=dataset_name,
            subtask_name=subtask_name,
            processed_data_label=processed_data_label,
            baseline_method=baseline_method,
            model_name=model_name,
            codex_home=codex_home,
            reasoning_effort=reasoning_effort,
            overwrite=overwrite,
            isolation=isolation,
            replicate=replicate,
            interpreter_root=interpreter_root,
            spark_version=spark_version,
            path_prepend=path_prepend,
            deequ_jars=deequ_jars,
            prompt_path=prompt_path,
            dump_prompt=dump_prompt,
        )
    finally:
        # py4j's gateway threads are not daemons, so a session left open keeps
        # the interpreter alive after the work is done -- the process then sits
        # until an external timeout kills it. The staged `deequ_check.py` calls
        # os._exit() to dodge exactly this; the driver has to shut down cleanly
        # instead, because a batch runs this once per dataset.
        stop_spark(spark_train)


def stop_spark(spark) -> None:
    """Release the Spark session and its callback server, never raising."""
    try:
        spark.sparkContext._gateway.shutdown_callback_server()
    except Exception:  # noqa: BLE001
        pass
    try:
        spark.stop()
    except Exception:  # noqa: BLE001
        pass


def _run_scripts(*, script_path_list, project_manager, project_root, dq_manager, spark_train,
                 spark_train_data, observed_data_path, example_rows, dataset_name, subtask_name,
                 processed_data_label, baseline_method, model_name, codex_home,
                 reasoning_effort, overwrite, isolation, replicate,
                 interpreter_root, spark_version, path_prepend, deequ_jars,
                 prompt_path=None, dump_prompt=None):
    """One codex call per script, writing an artifact for each."""
    for script_path in script_path_list:
        constraints_path = project_manager.get_source_constraints_path(subtask_name, script_path.stem)
        constraints_path.mkdir(parents=True, exist_ok=True)
        constraint_output_path = constraints_path / make_output_filename(
            model_name, baseline_method, reasoning_effort, replicate
        )
        # Only replicate 1 can supersede a legacy timestamped artifact; a later
        # replicate is a new sample, not a replacement for an older run.
        stale = (superseded_artifacts(constraints_path, baseline_method, model_name,
                                      reasoning_effort)
                 if not replicate or int(replicate) <= 1 else [])
        # Not for `--dump-prompt`: it writes no artifact, so an existing one is
        # no reason to skip -- and skipping would make the flag silently do
        # nothing exactly when you most want to read the prompt, on a script that
        # has already been run.
        if (constraint_output_path.exists() or stale) and not overwrite and dump_prompt is None:
            existing = constraint_output_path if constraint_output_path.exists() else stale[0]
            print(f"Constraints file {existing} already exists. Skipping...")
            # The prompt is a file that is meant to be edited, so "already
            # exists" is a weak statement here: the artifact may have been
            # written under wording that no longer exists anywhere. Skipping
            # silently would make an edit look applied when the sweep never
            # re-ran.
            warn_if_prompt_changed(existing, prompt_path)
            continue
        source_code, assertions = FileLoader.load_py_file(script_path).extract_assertions()
        assert "# ASSERTION_START" not in str(source_code)

        print(f"Running {baseline_method} for dataset: {dataset_name}, subtask: {subtask_name}, "
              f"processed_data_label: {processed_data_label}, script: {script_path.stem}, "
              f"model: {model_name}")
        prompt_content = build_detail_prompt(source_code, example_rows, prompt_path,
                                             script_stem=script_path.stem)
        # Appended even though the row owns its prompt: the scope rule is what
        # `audit_events` measures against afterwards, so an edit to the document
        # must not be able to silently remove it.
        prompt_content = with_scope_rule(prompt_content)

        if dump_prompt is not None:
            # The rendered prompt is otherwise only ever written inside a temp dir
            # that is deleted on the way out, so the exact text an agent received
            # is recoverable only by digging through a trajectory. This writes it
            # where it can be read, and skips the run.
            target = Path(dump_prompt)
            if target.is_dir() or len(script_path_list) > 1:
                target.mkdir(parents=True, exist_ok=True)
                target = target / f"task--{dataset_name}--{script_path.stem}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(prompt_content, encoding="utf-8")
            print(f"Wrote prompt: {target}  ({len(prompt_content)} chars, no codex run)")
            continue
        output = {}

        with tempfile.TemporaryDirectory(prefix=f"{baseline_method}_synth_") as temp_dir:
            work_dir = Path(temp_dir)
            staged = stage_deequ_workspace(observed_data_path, work_dir, deequ_jars,
                                           script_path.stem, source_code)
            print(f"Staged observed table at {staged}, program at "
                  f"{staged_code_rel(script_path.stem)}, with {DEEQU_CHECKER_NAME}")
            failed_log = failed_attempts_path(constraint_output_path)
            try:
                raw_constraints, trajectory = run_codex_call(
                prompt_content,
                model_name=model_name,
                work_dir=work_dir,
                codex_home=codex_home,
                path_prepend=path_prepend,
                audit_forbidden=AUDIT_FORBIDDEN,
                reasoning_effort=reasoning_effort,
                isolation=isolation,
                readable=(
                    (interpreter_root,
                     *interpreter_support_roots(project_root, interpreter_root),
                     # The pinned classpath lives in the Ivy jar store; under a
                     # mount namespace it has to be bound at its own path or the
                     # checker's `spark.jars` entries all point at nothing.
                     Path.home() / ".ivy2" / "jars")
                    if interpreter_root else ()
                ),
                # Ivy resolves the Deequ jar under $HOME, and $HOME in the
                # namespace is the workspace; without the cache every run would
                # re-fetch 34MB of jars through the proxy.
                readable_at=((Path.home() / ".ivy2", work_dir / ".ivy2"),),
                extra_env={"SPARK_VERSION": spark_version} if spark_version else None,
                # PySpark reaches its own JVM over a loopback socket, and codex's
                # workspace-write sandbox blocks socket creation outright, so the
                # staged runner needs it opened.
                network_access=True,
                # Refuse the artifact if the agent reached for the staged runner
                # and never got a verdict out of it.
                require_deequ_tool=True,
                )
            except Exception as exc:
                # Billed even though it produced nothing; kept so the run that
                # finally succeeds for this script carries the cost.
                append_failed_attempts(failed_log, getattr(exc, "failed_attempts", []))
                raise

        try:
            raw_constraints_as_written = raw_constraints
            raw_constraints, normalization = normalize_raw_constraints(
                raw_constraints, set(spark_train_data.columns))
            code_list_for_constraints = [item["code_for_constraint"] for item in raw_constraints]
            code_list_for_constraints_valid = dq_manager.filter_valid_constraints_on_spark(
                code_list_for_constraints, spark_train, spark_train_data)
            constraints = Constraints.from_deequ_output(raw_constraints, code_list_for_constraints_valid)
        except Exception as exc:
            # The agent's run completed and was billed; only reading its output failed.
            accounting = trajectory.get("token_accounting") or {}
            append_failed_attempts(failed_log, trajectory.get("failed_attempts", []) + [{
                "attempt": trajectory.get("attempts"),
                "outcome": f"output not readable: {type(exc).__name__}: {exc}"[:500],
                "wall_clock_seconds": trajectory.get("wall_clock_seconds"),
                "rollout_path": accounting.get("rollout_path"),
                "llm_calls": accounting.get("llm_calls") or accounting.get("llm_calls_estimate"),
                "input_tokens": accounting.get("input_tokens"),
                "cached_input_tokens": accounting.get("cached_input_tokens"),
                "output_tokens": accounting.get("output_tokens"),
                "reasoning_tokens": accounting.get("reasoning_tokens"),
                "usd": (trajectory.get("cost") or {}).get("usd"),
                "usd_without_cache": trajectory.get("cost_without_cache_usd"),
            }])
            raise

        output['baseline_method'] = baseline_method
        output['model_name'] = model_name
        output['created_at'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        # The prompt is a file that is meant to change, so an artifact that does
        # not carry the text cannot be compared with one written before an edit.
        # Stored in full rather than as a path or a hash: the point of the row is
        # which wording produced which constraints.
        output['prompt_source'] = str(Path(prompt_path or PROMPT_PATH))
        # Both: the template is what an edit changes and what a rerun should be
        # compared against; the rendered text is what this agent actually read,
        # sample rows and all.
        output['prompt_template'] = load_prompt_template(prompt_path)
        output['prompt_text'] = prompt_content
        output['reasoning_effort'] = reasoning_effort
        output['isolation'] = isolation
        output['staged_code_path'] = staged_code_rel(script_path.stem)
        output['replicate'] = int(replicate) if replicate else 1
        output['token_accounting'] = trajectory.get('token_accounting')
        output['cost'] = trajectory.get('cost')
        output['cost_without_cache_usd'] = trajectory.get('cost_without_cache_usd')
        # Earlier processes that failed on this script, then retries inside this one.
        output['failed_attempts'] = read_failed_attempts(failed_log) + (trajectory.get('failed_attempts') or [])
        output['cost_all_attempts'] = cost_all_attempts(
            (trajectory.get('cost') or {}).get('usd'), trajectory.get('cost_without_cache_usd'),
            output['failed_attempts'])
        output['wall_clock_seconds'] = trajectory.get('wall_clock_seconds')
        output['attempts'] = trajectory.get('attempts')
        # Top-level, beside the other run-identifying fields, so scoring can tell
        # a row where the agent verified its constraints from one where it only
        # appeared to. Buried in `trajectory` it is nobody's job to look at.
        output['deequ_tool_health'] = trajectory.get('deequ_tool_health')
        output['raw_constraints'] = raw_constraints
        output['constraint_normalization'] = normalization
        output['raw_constraints_as_written'] = raw_constraints_as_written
        output['trajectory'] = trajectory
        output['constraints'] = constraints.to_dict()
        with open(constraint_output_path, "w", encoding="utf-8") as f:
            yaml.dump(output, f, allow_unicode=True)
        print(f"Wrote: {constraint_output_path}")
        # Folded into the artifact above, so the waiting records are done.
        failed_log.unlink(missing_ok=True)
        # Only once the replacement is safely on disk, so a failed run cannot
        # leave the label with no artifact at all.
        for path in stale:
            if path != constraint_output_path:
                path.unlink()
                print(f"Removed superseded timestamped artifact: {path}")
