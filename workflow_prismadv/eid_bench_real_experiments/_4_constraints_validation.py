#!/usr/bin/env python3
"""Validate EIDBench-real table-local constraints on clean and corrupted bundles."""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import oyaml as yaml

from prismadv.data_models import ConstraintsWithSources, ValidationResults
from prismadv.data_models.constraints_v2 import ser_column_group_key
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import MultiTableLoader
from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils


VALID_TARGETS = {"clean", "observed", "corrupted", "all"}
CONSTRAINT_ARTIFACT_GLOBS = (
    "prismadv_real_etl--*.yaml",
    "single_shot_real_etl--*.yaml",
)
JAVA_SECURITY_MANAGER_FLAG = "-Djava.security.manager=allow"
PROJECT_SCRIPT_ID = "project"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", default="omop_cdm_synthea")
    parser.add_argument("--script-id", default="patient_summary")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--constraint-path",
        type=Path,
        default=None,
        help="Constraint artifact to validate. Defaults to the latest known EIDBench-real constraint artifact.",
    )
    parser.add_argument(
        "--target",
        choices=sorted(VALID_TARGETS),
        default="all",
        help="Bundle target to validate. 'all' means clean plus all corrupted bundles.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Corruption label to validate. May be repeated. Only applies to corrupted/all targets.",
    )
    parser.add_argument(
        "--include-invalid-generated",
        action="store_true",
        help="Also validate constraints that failed clean-sample validation during generation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def ensure_java_security_manager_compat() -> None:
    """Set a JVM compatibility flag needed by some local Spark/JDK combinations."""
    current_options = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if JAVA_SECURITY_MANAGER_FLAG not in current_options:
        os.environ["JAVA_TOOL_OPTIONS"] = f"{JAVA_SECURITY_MANAGER_FLAG} {current_options}".strip()


def latest_constraint_path(example_id: str, script_id: str, project_root: Path) -> Path:
    constraints_dir = utils.constraints_dir(example_id, script_id, project_root)
    candidates = [
        path
        for artifact_glob in CONSTRAINT_ARTIFACT_GLOBS
        for path in constraints_dir.glob(artifact_glob)
    ]
    if not candidates:
        raise FileNotFoundError(f"no EIDBench-real constraint artifacts found in {constraints_dir}")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name))[-1]


def load_constraints(path: Path) -> ConstraintsWithSources:
    raw = load_yaml(path)
    if "constraints" not in raw:
        raise ValueError(f"constraint artifact is missing 'constraints': {path}")
    return ConstraintsWithSources.from_dict({"constraints": raw["constraints"]})


def validation_targets(
    project_manager: MultiTableProjectManager,
    target: str,
    labels: list[str] | None = None,
) -> list[tuple[str, str | None]]:
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {sorted(VALID_TARGETS)}")
    selected: list[tuple[str, str | None]] = []
    if target in {"clean", "all"}:
        selected.append(("clean", None))
    if target == "observed":
        selected.append(("observed", None))
    if target in {"corrupted", "all"}:
        corruption_labels = labels if labels is not None else utils.corruption_labels(project_manager)
        selected.extend(("corrupted", label) for label in corruption_labels)
    return selected


def validation_output_path(
    example_id: str,
    script_id: str,
    variant: str,
    corruption_label: str | None,
    constraint_path: Path,
    project_root: Path,
) -> Path:
    output_dir = utils.constraints_validation_dir(
        example_id,
        script_id,
        variant,
        project_root,
        corruption_label=corruption_label,
        create=True,
    )
    return output_dir / f"validation_results__{constraint_path.stem}.yaml"


def close_spark_session(spark_session: Any) -> None:
    if spark_session is None:
        return
    try:
        spark_session.sparkContext._gateway.close()
    finally:
        spark_session.stop()


def validate_table_constraints(
    dq_manager: DeequDataQualityManager,
    spark: Any,
    spark_df: Any,
    table_constraints: ConstraintsWithSources,
    *,
    valid_only: bool = True,
) -> tuple[ValidationResults, float]:
    code_map = table_constraints.get_suggestions_code_column_map(valid_only=valid_only)
    code_list = list(code_map.keys())
    result_dict = defaultdict(lambda: {"code": []})
    if not code_list:
        return ValidationResults.from_dict(result_dict), 0.0

    start = time.perf_counter()
    validation_results = dq_manager.validate_constraints_with_reasons(
        spark,
        spark_df,
        code_list,
        isolated_check=True,
    )
    elapsed = time.perf_counter() - start
    for code, (status, reason_if_failed) in zip(code_list, validation_results):
        code_info = code_map[code]
        column_key = code_info["column"]
        result_dict[column_key]["code"].append(
            {
                "suggestion": code,
                "status": status,
                "reason_if_failed": reason_if_failed,
                "level": code_info["level"],
            }
        )
    return ValidationResults.from_dict(result_dict), elapsed


def validation_summary(result: ValidationResults) -> dict[str, int]:
    passed_warning, failed_warning, failed_error, passed_error, non_compilable = result.check_result()
    return {
        "passed_warning": passed_warning,
        "failed_warning": failed_warning,
        "passed_error": passed_error,
        "failed_error": failed_error,
        "non_compilable": non_compilable,
        "total": passed_warning + failed_warning + passed_error + failed_error + non_compilable,
    }


def serialize_table_results(table_results: dict[str, ValidationResults]) -> dict[str, Any]:
    return {
        table_name: table_result.to_dict()["results"]
        for table_name, table_result in sorted(table_results.items())
    }


def validate_constraints_on_bundle(
    project_manager: MultiTableProjectManager,
    constraints: ConstraintsWithSources,
    *,
    script_id: str,
    variant: str,
    corruption_label: str | None = None,
    valid_only: bool = True,
    dq_manager: DeequDataQualityManager | None = None,
) -> dict[str, Any]:
    ensure_java_security_manager_compat()
    dq_manager = dq_manager or DeequDataQualityManager()
    table_names = read_table_names_for_script(project_manager, script_id)
    bundle = MultiTableLoader(project_manager).load_pandas(
        variant=variant,
        corruption_label=corruption_label,
        table_names=table_names,
    )
    grouped_constraints = constraints.group_by_table()
    table_results: dict[str, ValidationResults] = {}
    table_summaries: dict[str, dict[str, int]] = {}
    validation_seconds: dict[str, float] = {}
    spark_session = None

    try:
        for table_name in table_names:
            table_constraints = grouped_constraints.get(table_name)
            if table_constraints is None:
                continue
            spark_df, spark_session = dq_manager.spark_df_from_pandas_df(
                bundle.tables[table_name],
                spark_session=spark_session,
            )
            table_result, elapsed = validate_table_constraints(
                dq_manager,
                spark_session,
                spark_df,
                table_constraints,
                valid_only=valid_only,
            )
            table_results[table_name] = table_result
            table_summaries[table_name] = validation_summary(table_result)
            validation_seconds[table_name] = elapsed
    finally:
        close_spark_session(spark_session)

    totals = defaultdict(int)
    for summary in table_summaries.values():
        for key, value in summary.items():
            totals[key] += value

    return {
        "example_id": project_manager.example_id,
        "script_id": script_id,
        "variant": variant,
        "corruption_label": corruption_label,
        "valid_generated_constraints_only": valid_only,
        "tables": serialize_table_results(table_results),
        "summary_by_table": table_summaries,
        "summary": dict(totals),
        "validation_seconds_by_table": validation_seconds,
    }


def read_table_names_for_script(project_manager: MultiTableProjectManager, script_id: str) -> list[str]:
    if script_id != PROJECT_SCRIPT_ID:
        return list(project_manager.get_script_spec(script_id)["reads"])

    table_names = []
    for script_spec in project_manager.scripts.values():
        table_names.extend(script_spec["reads"])
    return sorted(set(table_names))


def write_validation_result(path: Path, result: dict[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"validation artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)


def run_validation(
    *,
    project_root: Path,
    example_id: str,
    script_id: str,
    constraint_path: Path | None = None,
    target: str = "all",
    labels: list[str] | None = None,
    valid_only: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    project_root = project_root.resolve()
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    constraint_path = constraint_path or latest_constraint_path(example_id, script_id, project_root)
    constraints = load_constraints(constraint_path)
    output_paths = []
    for variant, corruption_label in validation_targets(project_manager, target, labels):
        output_path = validation_output_path(
            example_id,
            script_id,
            variant,
            corruption_label,
            constraint_path,
            project_root,
        )
        if output_path.exists() and not overwrite:
            print(f"Validation artifact already exists. Skipping: {output_path}")
            output_paths.append(output_path)
            continue
        result = validate_constraints_on_bundle(
            project_manager,
            constraints,
            script_id=script_id,
            variant=variant,
            corruption_label=corruption_label,
            valid_only=valid_only,
        )
        result["constraint_artifact"] = str(constraint_path)
        write_validation_result(output_path, result, overwrite=True)
        output_paths.append(output_path)
        print(f"Wrote validation artifact: {output_path}")
    return output_paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    output_paths = run_validation(
        project_root=project_root,
        example_id=args.example_id,
        script_id=args.script_id,
        constraint_path=args.constraint_path,
        target=args.target,
        labels=args.label,
        valid_only=not args.include_invalid_generated,
        overwrite=args.overwrite,
    )
    print(
        yaml.safe_dump(
            {
                "example_id": args.example_id,
                "script_id": args.script_id,
                "output_paths": [str(path) for path in output_paths],
            },
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
