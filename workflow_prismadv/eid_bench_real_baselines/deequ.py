#!/usr/bin/env python3
"""Generate EIDBench-real project-level constraints via Deequ constraint suggestion."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from prismadv.data_models.constraints_v2 import (
    CodeEntry,
    ColumnConstraintsWithSources,
    ConstraintsWithSources,
)
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.dq_manager.deequ._constraint_suggestion import get_suggestion_for_spark_df
from prismadv.loader import MultiTableLoader
from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils
from workflow_prismadv.eid_bench_real_experiments._4_constraints_validation import (
    close_spark_session,
    ensure_java_security_manager_compat,
)
from workflow_prismadv.eid_bench_real_baselines.single_shot import (
    load_or_prepare_project_inputs,
    write_yaml,
)

DEFAULT_EXAMPLE_ID = "omop_cdm_synthea"
PROJECT_SCRIPT_ID = "project"
BASELINE_METHOD = "deequ"
MODEL_TAG = "deequ"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", default=DEFAULT_EXAMPLE_ID)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_TAG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def make_output_filename() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{BASELINE_METHOD}--{MODEL_TAG}--{timestamp}.yaml"


def existing_output(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob(f"{BASELINE_METHOD}--{MODEL_TAG}--*.yaml"))
    return candidates[-1] if candidates else None


def infer_deequ_constraints(
    project_manager: MultiTableProjectManager,
    table_names: list[str],
    *,
    dq_manager: DeequDataQualityManager,
) -> tuple[ConstraintsWithSources, list[dict[str, str]]]:
    """Per table: run Deequ suggestion on the clean data, validate on clean, assemble."""
    ensure_java_security_manager_compat()
    bundle = MultiTableLoader(project_manager).load_pandas(
        variant="clean",
        table_names=table_names,
    )
    result = ConstraintsWithSources()
    raw_constraints: list[dict[str, str]] = []
    spark_session = None
    try:
        for table_name in table_names:
            spark_df, spark_session = dq_manager.spark_df_from_pandas_df(
                bundle.tables[table_name],
                spark_session=spark_session,
            )
            suggestion = get_suggestion_for_spark_df(spark_session, spark_df)
            codes = [item["code_for_constraint"] for item in suggestion]
            if not codes:
                continue
            validation_results = dq_manager.validate_constraints_with_reasons(
                spark_session,
                spark_df,
                codes,
                isolated_check=True,
            )
            for item, (validity, reason_if_invalid) in zip(suggestion, validation_results):
                column_name = item["column_name"]
                code = item["code_for_constraint"]
                raw_constraints.append(
                    {"table_name": table_name, "column_name": column_name, "code_for_constraint": code}
                )
                key = f"{table_name}.{column_name}"
                if key not in result.data_map:
                    result.data_map[key] = ColumnConstraintsWithSources(
                        assumptions=[],
                        code=[],
                        table_name=table_name,
                        column_group=column_name,
                    )
                result.data_map[key].code.append(
                    CodeEntry(
                        suggestion=code,
                        validity=validity,
                        reason_if_invalid=reason_if_invalid,
                        level="error",
                    )
                )
    finally:
        close_spark_session(spark_session)
    return result, raw_constraints


def generate_deequ_constraints(
    *,
    project_root: Path,
    example_id: str,
    input_path: Path | None = None,
    overwrite: bool = False,
    dq_manager: DeequDataQualityManager | None = None,
) -> Path:
    project_root = project_root.resolve()
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    output_dir = utils.constraints_dir(example_id, PROJECT_SCRIPT_ID, project_root, create=True)
    existing = existing_output(output_dir)
    if existing is not None and not overwrite:
        print(f"Deequ EIDBench-real baseline already exists. Skipping: {existing}")
        return existing

    prismadv_inputs, resolved_input_path = load_or_prepare_project_inputs(
        project_manager,
        input_path=input_path,
        overwrite=False,
    )
    table_names = sorted(prismadv_inputs["table_profiles"])
    dq_manager = dq_manager or DeequDataQualityManager()
    constraints, raw_constraints = infer_deequ_constraints(
        project_manager, table_names, dq_manager=dq_manager
    )

    output_path = output_dir / make_output_filename()
    result = {
        "baseline_method": BASELINE_METHOD,
        "model_name": MODEL_TAG,
        "input_artifact": str(resolved_input_path),
        "raw_constraints": raw_constraints,
        "constraints": constraints.to_dict()["constraints"],
    }
    write_yaml(output_path, result)
    print(f"example_id: {example_id}")
    print(f"output_path: {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    generate_deequ_constraints(
        project_root=project_root,
        example_id=args.example_id,
        input_path=args.input_path,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
