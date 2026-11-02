"""Run the Codex CLI agent to produce EIDBench-real project-level constraints."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import oyaml as yaml

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_baselines.pipelines.run_mini_swe_agent import (
    DEFAULT_SAMPLE_ROWS,
    sample_rows_by_table,
)
from workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation import (
    build_downstream_task_description,
)
from workflow_prismadv.eid_bench_real_baselines.prompts import (
    PROMPT_PATH,
    load_prompt_template,
    render_prompt,
)
from workflow_prismadv.eid_bench_real_baselines.single_shot import (
    constraints_with_validation,
    load_or_prepare_project_inputs,
)
from workflow_prismadv.utils.codex_runner import (
    DEEQU_CHECKER_NAME,
    DEFAULT_MODEL,
    WORKSPACE_TABLES_REL,
    assert_no_answer_key,
    deequ_classpath,
    run_codex_call,
    with_scope_rule,
    write_deequ_checker,
)
from workflow_prismadv.utils.replicates import artifact_filename, existing_artifacts, normalize_replicate

BASELINE_METHOD = "codex"

AUDIT_FORBIDDEN = (
    "benchmarks",
    "data_processed",
    "errors",
    "corrupted",
    ".env",
)
DATA_VARIANT = "observed"
CODE_DIRNAME = "code"
FORBIDDEN_PARTS = ("errors", "corrupted")


def make_output_filename(model_name: str, baseline_method: str = BASELINE_METHOD, replicate=None) -> str:
    return artifact_filename(baseline_method, model_name, replicate)


def existing_output_for_model(
    output_dir: Path, model_name: str, baseline_method: str = BASELINE_METHOD, replicate=None
) -> Path | None:
    """The newest artifact for this model *and* replicate; another replicate is a new sample."""
    candidates = existing_artifacts(output_dir, baseline_method, model_name, replicate)
    return candidates[-1] if candidates else None


def stage_workspace(
    project_manager: MultiTableProjectManager,
    prismadv_inputs: dict,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Copy the observed tables and the pipeline source in, and drop the PyDeequ runner beside them.

    Everything the agent touches is a scratch copy inside its own per-run temp
    dir, so nothing it does can reach the benchmark originals. Returns the staged
    tables dir and the staged code dir.
    """
    tables_dir = work_dir / WORKSPACE_TABLES_REL
    tables_dir.mkdir(parents=True, exist_ok=True)
    for table_name, source in project_manager.get_table_paths(DATA_VARIANT).items():
        if not source.is_file():
            raise RuntimeError(f"{DATA_VARIANT} table {table_name!r} not found: {source}")
        shutil.copy(source, tables_dir / source.name)

    code_dir = work_dir / CODE_DIRNAME
    for entry in prismadv_inputs["code_context"]["files"]:
        target = code_dir / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(entry["content"], encoding="utf-8")
    write_deequ_checker(work_dir, deequ_classpath())

    assert_no_answer_key(work_dir, FORBIDDEN_PARTS)
    return tables_dir, code_dir


def run_codex(
    *,
    project_root: Path,
    example_id: str,
    input_path: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    codex_home: Path | None = None,
    prompt_path: Path | None = None,
    dump_prompt: Path | None = None,
    reasoning_effort: str | None = None,
    overwrite: bool = False,
    replicate=None,
    dq_manager: DeequDataQualityManager | None = None,
) -> Path:
    from workflow_prismadv.eid_bench_real_experiments import utils as exp_utils

    baseline_method = BASELINE_METHOD
    replicate = normalize_replicate(replicate)
    project_root = project_root.resolve()
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    output_dir = exp_utils.constraints_dir(example_id, "project", project_root, create=True)
    existing = existing_output_for_model(output_dir, model_name, baseline_method, replicate)
    if existing is not None and not overwrite and dump_prompt is None:
        print(f"{baseline_method} EIDBench-real baseline (replicate {replicate}) already exists; "
              f"skipping: {existing}")
        return existing

    prismadv_inputs, resolved_input_path = load_or_prepare_project_inputs(
        project_manager, input_path=input_path, overwrite=False,
    )
    table_names = list(prismadv_inputs["context"]["script"]["reads"])
    data_dir = f"./{WORKSPACE_TABLES_REL}"
    prompt_content = render_prompt(
        {
            "CODE": prismadv_inputs["code_context"]["combined_with_line_numbers"],
            "CODE_DIR": f"./{CODE_DIRNAME}",
            "DATA_SAMPLE": yaml.safe_dump(
                sample_rows_by_table(project_manager, table_names, sample_rows=sample_rows),
                sort_keys=False,
            ),
            "TABLES_DESC": yaml.safe_dump(
                prismadv_inputs["table_profiles"], sort_keys=False
            ),
            "TASK": build_downstream_task_description(prismadv_inputs),
            "DATA_DIR": data_dir,
            "CHECKER_CMD": (
                f"python {DEEQU_CHECKER_NAME} --table <table_name> --constraints <file>"
            ),
        },
        prompt_path,
    )
    prompt_content = with_scope_rule(prompt_content)

    if dump_prompt is not None:
        target = Path(dump_prompt)
        if target.is_dir():
            target = target / f"task--{example_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(prompt_content, encoding="utf-8")
        print(f"Wrote prompt: {target}  ({len(prompt_content)} chars, no codex run)")
        return target

    path_prepend = project_root / ".venv" / "bin"

    with tempfile.TemporaryDirectory(prefix=f"{baseline_method}_") as tmp_dir:
        work_dir = Path(tmp_dir)
        staged_tables, staged_code = stage_workspace(project_manager, prismadv_inputs, work_dir)
        print(f"Staged {DATA_VARIANT} tables at {staged_tables}, source at {staged_code}, "
              f"with {DEEQU_CHECKER_NAME}")
        print(f"Running codex in {work_dir} for example={example_id}, model={model_name}")
        raw_constraints, trajectory = run_codex_call(
            prompt_content,
            model_name=model_name,
            work_dir=work_dir,
            codex_home=codex_home,
            path_prepend=path_prepend,
            audit_forbidden=AUDIT_FORBIDDEN,
            reasoning_effort=reasoning_effort,
            network_access=True,
            require_deequ_tool=True,
        )

    constraints = constraints_with_validation(
        project_manager,
        raw_constraints,
        prismadv_inputs,
        dq_manager=dq_manager,
    )

    output_path = output_dir / make_output_filename(model_name, baseline_method, replicate)
    payload = {
        "baseline_method": baseline_method,
        "model_name": model_name,
        "reasoning_effort": reasoning_effort,
        "replicate": replicate,
        # The prompt is a file that is meant to change, so an artifact that does
        # not carry the text cannot be compared with one written before an edit.
        "prompt_source": str(Path(prompt_path or PROMPT_PATH)),
        "prompt_template": load_prompt_template(prompt_path),
        "prompt_text": prompt_content,
        "input_artifact": str(resolved_input_path),
        "data_variant": DATA_VARIANT,
        "staged_code_files": [
            f"{CODE_DIRNAME}/{entry['path']}" for entry in prismadv_inputs["code_context"]["files"]
        ],
        "raw_constraints": raw_constraints,
        "trajectory": trajectory,
        "constraints": constraints.to_dict()["constraints"],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
    print(f"Wrote: {output_path}")
    return output_path
