#!/usr/bin/env python3
"""Run the Codex CLI agent to produce ICDBench constraints.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import textwrap
import traceback
from datetime import datetime
from pathlib import Path

import oyaml as yaml
import pandas as pd

from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_baselines.pipelines.run_codex import (
    append_failed_attempts,
    cost_all_attempts,
    failed_attempts_path,
    interpreter_support_roots,
    make_output_filename,
    read_failed_attempts,
    resolve_interpreter,
)
from workflow_prismadv.icd_bench_experiments import ALL_EVALUATION_CASES
from workflow_prismadv.icd_bench_experiments.prompts import (
    PROMPT_PATH,
    load_prompt_template,
    render_prompt,
)
from workflow_prismadv.utils.codex_runner import (
    DEEQU_CHECKER_NAME,
    DEFAULT_CODEX_HOME,
    DEFAULT_ISOLATION,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    ISOLATION_AUDIT,
    ISOLATION_BWRAP,
    WORKSPACE_TABLES_REL,
    assert_no_answer_key,
    deequ_classpath,
    run_codex_call,
    with_scope_rule,
    write_deequ_checker,
)
from workflow_prismadv.utils.replicates import add_replicate_argument, normalize_replicate

BASELINE_METHOD = "codex"

# Matched as path components. The case modules sit under benchmarks/ICDBench and
# hold the answer key; data_processed/icd_bench holds the other rows' outputs.
AUDIT_FORBIDDEN = (
    "benchmarks",
    "ICDBench",
    "data_processed",
    "icd_bench",
    ".env",
)

FORBIDDEN_PARTS = ("benchmarks", "ICDBench", "icd_bench")
STAGED_TABLE_NAME = "observed_data"
CODE_REL = "code/snippet.py"
# The other ICDBench baselines show `head(20)` of the sample in the prompt.
PROMPT_SAMPLE_ROWS = 20


def case_id(evaluation_case) -> str:
    """`<module>.<Class>`, the directory name every ICDBench row writes under."""
    return f"{evaluation_case.__class__.__module__.split('.')[-1]}.{evaluation_case.__class__.__name__}"


def constraints_dir(evaluation_case) -> Path:
    return get_project_root() / "data_processed" / "icd_bench" / case_id(evaluation_case) / "constraints"


def output_path(evaluation_case, model_name=DEFAULT_MODEL, reasoning_effort=DEFAULT_REASONING_EFFORT,
                replicate=None) -> Path:
    """`codex--<model>--<effort>[--rN].yaml`, the synth row's name."""
    return constraints_dir(evaluation_case) / make_output_filename(
        model_name, BASELINE_METHOD, reasoning_effort, replicate)


def select_cases(names=None) -> list:
    """Every case, or those whose module or class name is in `names`."""
    if not names:
        return list(ALL_EVALUATION_CASES)
    wanted = set(names)
    chosen = [case for case in ALL_EVALUATION_CASES
              if wanted & {case_id(case), case_id(case).split(".")[0], case.__class__.__name__}]
    found = {name for case in chosen
             for name in (case_id(case), case_id(case).split(".")[0], case.__class__.__name__)}
    if missing := wanted - found:
        raise ValueError(f"no ICDBench case matches {sorted(missing)}")
    return chosen


def snippet(evaluation_case) -> str:
    return textwrap.dedent(evaluation_case.downstream_code()).strip() + "\n"


def build_prompt(evaluation_case, prompt_path=None) -> str:
    data_sample = pd.DataFrame(evaluation_case.sample_data()).head(PROMPT_SAMPLE_ROWS).to_string(index=False)
    prompt = render_prompt(
        {
            "CODE": snippet(evaluation_case).rstrip(),
            "CODE_PATH": f"./{CODE_REL}",
            "DATA_SAMPLE": data_sample,
            "TARGET_COLUMN": evaluation_case.target_column(),
            "TABLE_PATH": f"./{WORKSPACE_TABLES_REL}/{STAGED_TABLE_NAME}.csv",
            "TABLE_NAME": STAGED_TABLE_NAME,
            "CHECKER_CMD": (f"python {DEEQU_CHECKER_NAME} --table {STAGED_TABLE_NAME} "
                            f"--constraints <your_constraints_file>"),
        },
        prompt_path,
    )
    # Appended outside the document, as in the EIDBench rows: the audit measures
    # against this rule, so an edit to the document must not remove it.
    return with_scope_rule(prompt)


def stage_workspace(evaluation_case, work_dir: Path, deequ_jars: tuple[str, ...]) -> None:
    """The sample data, the snippet and the PyDeequ runner; nothing else of the case."""
    tables_dir = work_dir / WORKSPACE_TABLES_REL
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(evaluation_case.sample_data()).to_csv(tables_dir / f"{STAGED_TABLE_NAME}.csv", index=False)
    code_path = work_dir / CODE_REL
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(snippet(evaluation_case), encoding="utf-8")
    write_deequ_checker(work_dir, deequ_jars)
    assert_no_answer_key(work_dir, FORBIDDEN_PARTS)


def constraint_codes(raw_constraints) -> list[str]:
    """The code of every entry, in the formats the staged checker accepts."""
    codes = []
    for item in raw_constraints or []:
        code = item if isinstance(item, str) else (item.get("code_for_constraint") or item.get("suggestion"))
        if code:
            codes.append(code)
    return codes


def run_case(evaluation_case, *, model_name, reasoning_effort, replicate, codex_home, isolation,
             prompt_path, project_root, interpreter_root, spark_version, deequ_jars) -> Path:
    artifact_path = output_path(evaluation_case, model_name, reasoning_effort, replicate)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_content = build_prompt(evaluation_case, prompt_path)
    failed_log = failed_attempts_path(artifact_path)

    with tempfile.TemporaryDirectory(prefix=f"{BASELINE_METHOD}_icd_") as temp_dir:
        work_dir = Path(temp_dir)
        stage_workspace(evaluation_case, work_dir, deequ_jars)
        try:
            raw_constraints, trajectory = run_codex_call(
                prompt_content,
                model_name=model_name,
                work_dir=work_dir,
                codex_home=codex_home,
                path_prepend=interpreter_root / "bin",
                audit_forbidden=AUDIT_FORBIDDEN,
                reasoning_effort=reasoning_effort,
                isolation=isolation,
                readable=(interpreter_root, *interpreter_support_roots(project_root, interpreter_root),
                          Path.home() / ".ivy2" / "jars"),
                readable_at=((Path.home() / ".ivy2", work_dir / ".ivy2"),),
                extra_env={"SPARK_VERSION": spark_version},
                # PySpark reaches its own JVM over a loopback socket, which
                # codex's workspace-write sandbox blocks.
                network_access=True,
                require_deequ_tool=True,
            )
        except Exception as exc:
            # Billed even though it produced nothing; the run that finally
            # succeeds for this case carries the cost.
            append_failed_attempts(failed_log, getattr(exc, "failed_attempts", []))
            raise

    failed_attempts = read_failed_attempts(failed_log) + (trajectory.get("failed_attempts") or [])
    payload = {
        "baseline_method": BASELINE_METHOD,
        "model_name": model_name,
        "reasoning_effort": reasoning_effort,
        "replicate": normalize_replicate(replicate),
        "case": case_id(evaluation_case),
        "target_column": evaluation_case.target_column(),
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "prompt_source": str(Path(prompt_path or PROMPT_PATH)),
        "prompt_template": load_prompt_template(prompt_path),
        "prompt_text": prompt_content,
        "isolation": isolation,
        "token_accounting": trajectory.get("token_accounting"),
        "cost": trajectory.get("cost"),
        "cost_without_cache_usd": trajectory.get("cost_without_cache_usd"),
        "failed_attempts": failed_attempts,
        "cost_all_attempts": cost_all_attempts(
            (trajectory.get("cost") or {}).get("usd"), trajectory.get("cost_without_cache_usd"),
            failed_attempts),
        "wall_clock_seconds": trajectory.get("wall_clock_seconds"),
        "attempts": trajectory.get("attempts"),
        "deequ_tool_health": trajectory.get("deequ_tool_health"),
        "raw_constraints": raw_constraints,
        "trajectory": trajectory,
        # Every constraint the agent wrote, unfiltered: no validity check on
        # the sample and no target-column filter.
        "constraints": constraint_codes(raw_constraints),
    }
    with open(artifact_path, "w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, allow_unicode=True)
    failed_log.unlink(missing_ok=True)
    return artifact_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", default=None,
                        help="Run only this case (repeatable): module name, class name or <module>.<Class>. "
                             "Default: all 63.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        help=f"Codex model_reasoning_effort (default: {DEFAULT_REASONING_EFFORT}); "
                             f"part of the artifact name.")
    parser.add_argument("--codex-home", type=Path, default=DEFAULT_CODEX_HOME,
                        help="CODEX_HOME holding the credential, which keeps the API key out of the "
                             "agent's shell.")
    parser.add_argument("--isolation", choices=[ISOLATION_BWRAP, ISOLATION_AUDIT], default=DEFAULT_ISOLATION)
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help=f"Prompt document to run instead of {PROMPT_PATH}.")
    parser.add_argument("--dump-prompt", type=Path, default=None,
                        help="Write the rendered prompt of each selected case there and exit without "
                             "calling codex. A directory (or more than one case) gets one file per case.")
    parser.add_argument("--overwrite", action="store_true")
    add_replicate_argument(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cases = select_cases(args.case)

    if args.dump_prompt is not None:
        for evaluation_case in cases:
            target = Path(args.dump_prompt)
            if target.is_dir() or len(cases) > 1:
                target.mkdir(parents=True, exist_ok=True)
                target = target / f"task--{case_id(evaluation_case)}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            prompt = build_prompt(evaluation_case, args.prompt_file)
            target.write_text(prompt, encoding="utf-8")
            print(f"Wrote prompt: {target}  ({len(prompt)} chars, no codex run)")
        return 0

    todo = []
    for evaluation_case in cases:
        existing = output_path(evaluation_case, args.model_name, args.reasoning_effort, args.replicate)
        if existing.exists() and not args.overwrite:
            print(f"Constraints file {existing} already exists. Skipping...")
        else:
            todo.append(evaluation_case)
    if not todo:
        return 0

    project_root = get_project_root()
    interpreter_root, spark_version = resolve_interpreter(project_root)
    deequ_jars = deequ_classpath()

    failed = []
    for evaluation_case in todo:
        print(f"Running {BASELINE_METHOD} for ICDBench case {case_id(evaluation_case)} "
              f"(target column {evaluation_case.target_column()!r}), model {args.model_name}, "
              f"effort {args.reasoning_effort}, replicate {normalize_replicate(args.replicate)}", flush=True)
        try:
            path = run_case(
                evaluation_case,
                model_name=args.model_name,
                reasoning_effort=args.reasoning_effort,
                replicate=args.replicate,
                codex_home=args.codex_home,
                isolation=args.isolation,
                prompt_path=args.prompt_file,
                project_root=project_root,
                interpreter_root=interpreter_root,
                spark_version=spark_version,
                deequ_jars=deequ_jars,
            )
            print(f"Wrote: {path}", flush=True)
        except Exception:  # noqa: BLE001
            # One case's failure does not stop the others; a rerun picks it up.
            traceback.print_exc()
            failed.append(case_id(evaluation_case))
    if failed:
        print(f"{len(failed)} case(s) failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # deequ_classpath() leaves its Spark session running, and py4j's threads
    # keep the interpreter alive otherwise.
    os._exit(code)
