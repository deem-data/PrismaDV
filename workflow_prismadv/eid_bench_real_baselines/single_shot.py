#!/usr/bin/env python3
"""Generate EIDBench-real project-level constraints with a single LLM prompt."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import oyaml as yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.data_models.constraints_v2 import (
    CodeEntry,
    ColumnConstraintsWithSources,
    ConstraintsWithSources,
)
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import MultiTableLoader
from prismadv.llm_backend.entry import get_langchain_model
from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import _3_0_prepare_prismadv_inputs as input_prep
from workflow_prismadv.eid_bench_real_experiments import utils
from workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation import (
    build_downstream_task_description,
)
from workflow_prismadv.eid_bench_real_experiments._4_constraints_validation import (
    close_spark_session,
    ensure_java_security_manager_compat,
)


DEFAULT_EXAMPLE_ID = "omop_cdm_synthea"
PROJECT_SCRIPT_ID = "project"
DEFAULT_MODEL_NAME = "gpt-5-mini"
MAX_RETRIES = 3


SINGLE_SHOT_REAL_ETL_PROMPT = """
Write PyDeequ data unit tests for a EIDBench-real project.

The project reads multiple named input tables. Generate table-local constraints only:
the code will run against one target table dataframe at a time, not against a joined
bundle. Do not generate cross-table constraints or foreign-key checks.

Input table profiles:
{tables_desc}

Sample rows by table:
{data_sample}

Project code:
{code_context}

Downstream task description:
{downstream_task_description}

Focus on Python code for PyDeequ Check objects only.

Return only a valid JSON object with this shape:
{{
  "constraints": [
    {{
      "table_name": "table_name",
      "column_name": "column_name",
      "code_for_constraint": ".isComplete(\\"column_name\\")"
    }}
  ]
}}
"""


ConstraintInferer = Callable[[dict[str, Any]], list[dict[str, Any]]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", default=DEFAULT_EXAMPLE_ID)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Prepared EIDBench-real input artifact. Defaults to constraints/project/prismadv_inputs.yaml.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def default_input_path(example_id: str, project_root: Path) -> Path:
    return utils.constraints_dir(example_id, PROJECT_SCRIPT_ID, project_root) / "prismadv_inputs.yaml"


def load_or_prepare_project_inputs(
    project_manager: MultiTableProjectManager,
    *,
    input_path: Path | None,
    overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    input_path = input_path or default_input_path(project_manager.example_id, project_manager.project_root)
    if input_path.exists() and not overwrite:
        return load_yaml(input_path), input_path

    prismadv_inputs = input_prep.build_prismadv_inputs(
        project_manager,
        PROJECT_SCRIPT_ID,
        variant="clean",
    )
    input_prep.write_prismadv_input_artifacts(
        input_path.parent,
        prismadv_inputs,
        overwrite=True,
    )
    return prismadv_inputs, input_path


def make_output_filename(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"single_shot_real_etl--{model_name}--{timestamp}.yaml"


def existing_output_for_model(output_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(output_dir.glob(f"single_shot_real_etl--{model_name}--*.yaml"))
    return candidates[-1] if candidates else None


def sample_rows_by_table(
    project_manager: MultiTableProjectManager,
    table_names: list[str],
    *,
    sample_rows: int,
) -> dict[str, list[dict[str, Any]]]:
    bundle = MultiTableLoader(project_manager).load_pandas(
        variant="clean",
        table_names=table_names,
    )
    samples = {}
    for table_name, dataframe in bundle.tables.items():
        if dataframe.empty or sample_rows <= 0:
            samples[table_name] = []
            continue
        n_rows = min(sample_rows, len(dataframe))
        samples[table_name] = dataframe.sample(n_rows, random_state=0).to_dict(orient="records")
    return samples


def build_llm_input_variables(
    project_manager: MultiTableProjectManager,
    prismadv_inputs: dict[str, Any],
    *,
    sample_rows: int,
) -> dict[str, Any]:
    table_names = list(prismadv_inputs["context"]["script"]["reads"])
    return {
        "tables_desc": yaml.safe_dump(prismadv_inputs["table_profiles"], sort_keys=False),
        "data_sample": yaml.safe_dump(
            sample_rows_by_table(project_manager, table_names, sample_rows=sample_rows),
            sort_keys=False,
        ),
        "code_context": prismadv_inputs["code_context"]["combined_with_line_numbers"],
        "downstream_task_description": build_downstream_task_description(prismadv_inputs),
    }


def infer_single_shot_constraints(
    input_variables: dict[str, Any],
    *,
    model_name: str,
    temperature: float,
    max_retries: int = MAX_RETRIES,
) -> list[dict[str, Any]]:
    prompt = ChatPromptTemplate.from_template(SINGLE_SHOT_REAL_ETL_PROMPT)
    llm = get_langchain_model(model_name, temperature=temperature)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    for attempt in range(max_retries):
        try:
            response = chain.invoke(input_variables)
            constraints = response["constraints"]
            if not isinstance(constraints, list):
                raise ValueError("single-shot response field 'constraints' must be a list")
            return constraints
        except Exception:
            if attempt == max_retries - 1:
                raise
    raise RuntimeError("unreachable")


def normalize_raw_constraints(
    raw_constraints: list[dict[str, Any]],
    table_profiles: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    table_columns = {
        table_name: {column["name"] for column in table_profile.get("columns", [])}
        for table_name, table_profile in table_profiles.items()
    }
    constraints_by_table: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen = set()
    for item in raw_constraints:
        table_name = item.get("table_name") or item.get("table")
        column_name = item.get("column_name") or item.get("column")
        suggestion = item.get("code_for_constraint") or item.get("suggestion")
        if not table_name or not column_name or not suggestion:
            continue
        if table_name not in table_columns or column_name not in table_columns[table_name]:
            continue
        marker = (table_name, column_name, suggestion)
        if marker in seen:
            continue
        seen.add(marker)
        constraints_by_table[table_name].append(
            {
                "table_name": str(table_name),
                "column_name": str(column_name),
                "suggestion": str(suggestion),
            }
        )
    return dict(constraints_by_table)


def constraints_with_validation(
    project_manager: MultiTableProjectManager,
    raw_constraints: list[dict[str, Any]],
    prismadv_inputs: dict[str, Any],
    *,
    dq_manager: DeequDataQualityManager | None = None,
) -> ConstraintsWithSources:
    ensure_java_security_manager_compat()
    dq_manager = dq_manager or DeequDataQualityManager()
    constraints_by_table = normalize_raw_constraints(raw_constraints, prismadv_inputs["table_profiles"])
    table_names = sorted(constraints_by_table)
    bundle = MultiTableLoader(project_manager).load_pandas(
        variant="clean",
        table_names=table_names,
    )
    result = ConstraintsWithSources()
    spark_session = None
    try:
        for table_name in table_names:
            table_constraints = constraints_by_table[table_name]
            spark_df, spark_session = dq_manager.spark_df_from_pandas_df(
                bundle.tables[table_name],
                spark_session=spark_session,
            )
            codes = [item["suggestion"] for item in table_constraints]
            validation_results = dq_manager.validate_constraints_with_reasons(
                spark_session,
                spark_df,
                codes,
                isolated_check=True,
            )
            for item, (validity, reason_if_invalid) in zip(table_constraints, validation_results):
                column_name = item["column_name"]
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
                        suggestion=item["suggestion"],
                        validity=validity,
                        reason_if_invalid=reason_if_invalid,
                        level="error",
                    )
                )
    finally:
        close_spark_session(spark_session)
    return result


def generate_single_shot_constraints(
    *,
    project_root: Path,
    example_id: str,
    input_path: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.6,
    sample_rows: int = 3,
    overwrite: bool = False,
    constraint_inferer: ConstraintInferer | None = None,
    dq_manager: DeequDataQualityManager | None = None,
) -> Path:
    project_root = project_root.resolve()
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    output_dir = utils.constraints_dir(example_id, PROJECT_SCRIPT_ID, project_root, create=True)
    existing_output = existing_output_for_model(output_dir, model_name)
    if existing_output is not None and not overwrite:
        print(f"Single-shot EIDBench-real baseline already exists. Skipping: {existing_output}")
        return existing_output

    prismadv_inputs, resolved_input_path = load_or_prepare_project_inputs(
        project_manager,
        input_path=input_path,
        overwrite=False,
    )
    input_variables = build_llm_input_variables(
        project_manager,
        prismadv_inputs,
        sample_rows=sample_rows,
    )
    inferer = constraint_inferer or (
        lambda variables: infer_single_shot_constraints(
            variables,
            model_name=model_name,
            temperature=temperature,
        )
    )
    raw_constraints = inferer(input_variables)
    constraints = constraints_with_validation(
        project_manager,
        raw_constraints,
        prismadv_inputs,
        dq_manager=dq_manager,
    )

    output_path = output_dir / make_output_filename(model_name)
    result = {
        "baseline_method": "single_shot_real_etl",
        "model_name": model_name,
        "temperature": temperature,
        "input_artifact": str(resolved_input_path),
        "raw_constraints": raw_constraints,
        "constraints": constraints.to_dict()["constraints"],
    }
    write_yaml(output_path, result)
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    output_path = generate_single_shot_constraints(
        project_root=project_root,
        example_id=args.example_id,
        input_path=args.input_path,
        model_name=args.model_name,
        temperature=args.temperature,
        sample_rows=args.sample_rows,
        overwrite=args.overwrite,
    )
    print(
        yaml.safe_dump(
            {
                "example_id": args.example_id,
                "script_id": PROJECT_SCRIPT_ID,
                "output_path": str(output_path),
            },
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
