#!/usr/bin/env python3
"""PocketFlow agent baseline for EIDBench-real."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils
from workflow_prismadv.utils.replicates import (
    add_replicate_argument,
    artifact_filename,
    existing_artifacts,
    normalize_replicate,
)
from workflow_prismadv.icd_bench_experiments.pocketflow.flow import create_code_generator_flow
from workflow_prismadv.icd_bench_experiments.pocketflow.utils.code_executor import execute_python
from workflow_prismadv.icd_bench_experiments.pocketflow.utils.call_llm import (
    get_message_log,
    get_usage_summary,
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
    return f"""Write PyDeequ data unit tests for a EIDBench-real project.

The project reads multiple named input tables and may consist of multiple
script files. Generate table-local constraints only: the validator runs
against one target table dataframe at a time, not against a joined table set.
Do not generate cross-table constraints or foreign-key checks.

Below are two examples that illustrate the expected behavior for
multi-table, multi-file projects.
---------------------------------------

### First Example:

Input table profiles (abridged):
  customers:
    columns: [customer_id (int), name (string), age (int)]
  orders:
    columns: [order_id (int), customer_id (int), amount (float)]

Sample rows by table:
  customers:
    - {{customer_id: 1, name: "Alice", age: 25}}
    - {{customer_id: 2, name: "Bob",   age: 30}}
    - {{customer_id: 3, name: "Carol", age: 22}}
  orders:
    - {{order_id: 101, customer_id: 1, amount: 49.99}}
    - {{order_id: 102, customer_id: 2, amount: 120.00}}
    - {{order_id: 103, customer_id: 1, amount: 15.50}}

Project code (multiple files):
  # clean_customers.py
  df = customers[customers['age'] >= 0]
  df = df.dropna(subset=['name'])

  # aggregate_orders.py
  totals = orders.groupby('customer_id')['amount'].sum()

Desired output:

def run_code():
    return {{
        "constraints": [
            {{"table_name": "customers", "column_name": "age", "code_for_constraint": ".isComplete(\\"age\\")"}},
            {{"table_name": "customers", "column_name": "age", "code_for_constraint": ".isNonNegative(\\"age\\")"}},
            {{"table_name": "customers", "column_name": "name", "code_for_constraint": ".isComplete(\\"name\\")"}},
            {{"table_name": "orders", "column_name": "customer_id", "code_for_constraint": ".isComplete(\\"customer_id\\")"}},
            {{"table_name": "orders", "column_name": "amount", "code_for_constraint": ".isNonNegative(\\"amount\\")"}},
        ]
    }}

### Second Example:

Input table profiles (abridged):
  products:
    columns: [product_id (int), name (string), category (string)]
  inventory:
    columns: [product_id (int), stock (int)]

Sample rows by table:
  products:
    - {{product_id: 1, name: "Apple", category: "fruit"}}
    - {{product_id: 2, name: null,    category: "fruit"}}
    - {{product_id: 3, name: "Bread", category: "bakery"}}
  inventory:
    - {{product_id: 1, stock: 100}}
    - {{product_id: 2, stock: 50}}
    - {{product_id: 3, stock: 200}}

Project code (multiple files):
  # notify.py
  unique_names = products['name'].dropna().unique().tolist()
  send_notifications(unique_names)

  # stock_check.py
  low_stock = inventory[inventory['stock'] < 10]

Desired output:

def run_code():
    return {{
        "constraints": [
            {{"table_name": "products", "column_name": "name", "code_for_constraint": ".hasNumberOfDistinctValues(\\"name\\", lambda x: x > 0)"}},
            {{"table_name": "inventory", "column_name": "stock", "code_for_constraint": ".isComplete(\\"stock\\")"}},
            {{"table_name": "inventory", "column_name": "stock", "code_for_constraint": ".isNonNegative(\\"stock\\")"}},
        ]
    }}

---------------------------------------

NOW I WILL GIVE YOU THE ACTUAL TASK DETAILS.

Input table profiles:
{input_variables['tables_desc']}

Sample rows by table:
{input_variables['data_sample']}

Project code (multiple files):
{input_variables['code_context']}

Downstream task description:
{input_variables['downstream_task_description']}

Focus on Python code for PyDeequ Check objects only. Generate a Python function
`run_code()` (taking no arguments) that returns a JSON object with a single key
"constraints" mapping to a list with the shape shown in the examples above. Make
sure each entry has `table_name`, `column_name`, and `code_for_constraint`
fields. Do not include cross-table or foreign-key checks. Your code must not
inspect any dataframe; only return the constraints.
"""


def run_pocketflow(problem: str, model_name: str = MODEL_TAG) -> list[dict[str, Any]]:
    os.environ["POCKETFLOW_MODEL"] = model_name
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
    model_name: str = MODEL_TAG,
    overwrite: bool = False,
    replicate=None,
    dq_manager: DeequDataQualityManager | None = None,
) -> Path:
    replicate = normalize_replicate(replicate)
    project_root = project_root.resolve()
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    output_dir = utils.constraints_dir(example_id, PROJECT_SCRIPT_ID, project_root, create=True)
    existing = existing_artifacts(output_dir, BASELINE_METHOD, model_name, replicate)
    if existing and not overwrite:
        print(f"PocketFlow EIDBench-real baseline (replicate {replicate}) already exists. "
              f"Skipping: {existing[-1]}")
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
            raw_constraints = run_pocketflow(problem, model_name)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"Attempt {attempt} failed: {exc}")
            if attempt == MAX_RETRIES:
                raise

    dq_manager = dq_manager or DeequDataQualityManager()
    constraints = constraints_with_validation(
        project_manager, raw_constraints, prismadv_inputs, dq_manager=dq_manager
    )

    output_path = output_dir / artifact_filename(BASELINE_METHOD, model_name, replicate)
    write_yaml(
        output_path,
        {
            "baseline_method": BASELINE_METHOD,
            "model_name": model_name,
            "replicate": replicate,
            "input_artifact": str(resolved_input_path),
            "raw_constraints": raw_constraints,
            "constraints": constraints.to_dict()["constraints"],
            "cost_summary": {"constraint_generation": get_usage_summary()},
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
    add_replicate_argument(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    generate_pocketflow_constraints(
        project_root=project_root,
        example_id=args.example_id,
        input_path=args.input_path,
        model_name=args.model_name,
        overwrite=args.overwrite,
        replicate=args.replicate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
