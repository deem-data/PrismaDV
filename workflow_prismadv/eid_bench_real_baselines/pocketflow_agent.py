#!/usr/bin/env python3
"""PocketFlow agent baseline for EIDBench-real."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils
from workflow_prismadv.icd_bench_experiments.pocketflow.flow import create_code_generator_flow
from workflow_prismadv.icd_bench_experiments.pocketflow.utils.code_executor import execute_python
from workflow_prismadv.icd_bench_experiments.pocketflow.utils.call_llm import (
    get_message_log,
    reset_message_log,
)
from workflow_prismadv.eid_bench_real_baselines.single_shot import (
    build_llm_input_variables,
    constraints_with_validation,
    load_or_prepare_project_inputs,
    write_yaml,
)

DEFAULT_EXAMPLE_ID = "omop_cdm_synthea"
PROJECT_SCRIPT_ID = "project"
BASELINE_METHOD = "pocketflow"
MODEL_TAG = "gpt-5"
DEFAULT_SAMPLE_ROWS = 3
MAX_RETRIES = 3
MAX_ITERATIONS = 3


def build_problem(input_variables: dict[str, Any]) -> str:
    return f"""Write pydeequ data unit tests for a multi-table dataset consumed by a data pipeline.

Downstream task:
{input_variables['downstream_task_description']}

Pipeline code context:
{input_variables['code_context']}

Table profiles:
{input_variables['tables_desc']}

Sample rows per table:
{input_variables['data_sample']}

Generate a Python function `run_code()` (taking no arguments) that returns a JSON object with a
list of constraints. Each constraint is an object with keys "table_name", "column_name", and
"suggestion", where "suggestion" is executable pydeequ Check code for that column (e.g.
.isComplete("col")). Your code must not inspect any dataframe; only return the constraints.

Here is an example of the code to generate, please replace the example constraints with the ones
you generate:

def run_code():
    return {{
        "constraints": [
            {{"table_name": "patients", "column_name": "age", "suggestion": ".isComplete(\\"age\\")"}},
            {{"table_name": "patients", "column_name": "age", "suggestion": ".isNonNegative(\\"age\\")"}},
        ]
    }}
"""


def run_pocketflow(problem: str) -> list[dict[str, Any]]:
    os.environ["POCKETFLOW_MODEL"] = MODEL_TAG
    reset_message_log()
    shared = {
        "problem": problem,
        "test_cases": [],
        "function_code": "",
        "test_results": [],
        "iteration_count": 0,
        "max_iterations": MAX_ITERATIONS,
    }
    flow = create_code_generator_flow()
    flow.run(shared)
    result, error = execute_python(shared["function_code"], input={})
    if error or not isinstance(result, dict) or "constraints" not in result:
        raise RuntimeError(f"pocketflow produced no usable constraints (error={error})")
    constraints = result["constraints"]
    if not isinstance(constraints, list):
        raise RuntimeError("pocketflow 'constraints' field must be a list")
    return constraints


def generate_pocketflow_constraints(
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
    existing = sorted(output_dir.glob(f"{BASELINE_METHOD}--{MODEL_TAG}--*.yaml"))
    if existing and not overwrite:
        print(f"PocketFlow EIDBench-real baseline already exists. Skipping: {existing[-1]}")
        return existing[-1]

    prismadv_inputs, resolved_input_path = load_or_prepare_project_inputs(
        project_manager, input_path=input_path, overwrite=False
    )
    input_variables = build_llm_input_variables(
        project_manager, prismadv_inputs, sample_rows=DEFAULT_SAMPLE_ROWS
    )
    problem = build_problem(input_variables)

    raw_constraints: list[dict[str, Any]] = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_constraints = run_pocketflow(problem)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"Attempt {attempt} failed: {exc}")
            if attempt == MAX_RETRIES:
                raise

    dq_manager = dq_manager or DeequDataQualityManager()
    constraints = constraints_with_validation(
        project_manager, raw_constraints, prismadv_inputs, dq_manager=dq_manager
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{BASELINE_METHOD}--{MODEL_TAG}--{timestamp}.yaml"
    write_yaml(
        output_path,
        {
            "baseline_method": BASELINE_METHOD,
            "model_name": MODEL_TAG,
            "input_artifact": str(resolved_input_path),
            "raw_constraints": raw_constraints,
            "constraints": constraints.to_dict()["constraints"],
        },
    )
    messages_path = output_path.with_name(output_path.stem + ".messages.json")
    messages_path.write_text(json.dumps({"messages": get_message_log()}, indent=2))
    print(f"example_id: {example_id}")
    print(f"output_path: {output_path}")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", default=DEFAULT_EXAMPLE_ID)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_TAG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    generate_pocketflow_constraints(
        project_root=project_root,
        example_id=args.example_id,
        input_path=args.input_path,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
