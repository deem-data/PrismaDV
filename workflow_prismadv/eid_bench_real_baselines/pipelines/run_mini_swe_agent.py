"""Run mini-swe-agent to produce EIDBench-real project-level constraints.

Mirrors `workflow_prismadv/eid_bench_baselines/pipelines/run_mini_swe_agent.py`
but adapted to the multi-table, multi-script project scope used by
EIDBench-real. The agent receives:

  * 2 multi-table few-shot examples (lifted from the few-shot baseline prompt)
  * Per-table profiles (dtype, completeness, samples)
  * 3 sample rows per table
  * The concatenated project source files (line-numbered)
  * The downstream task description

and writes a `constraints.json` file in its working directory with the same
shape used by the few-shot / zero-shot baselines (so the existing validator
loop in :mod:`workflow_prismadv.eid_bench_real_baselines.single_shot` can be
reused for Spark/Deequ validation).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import oyaml as yaml
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import MultiTableLoader
from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_baselines.single_shot import (
    constraints_with_validation,
    load_or_prepare_project_inputs,
)
from workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation import (
    build_downstream_task_description,
)


BASELINE_METHOD = "swe_agent_real_etl"
DEFAULT_COST_LIMIT = 0.50
DEFAULT_SAMPLE_ROWS = 3
MAX_RETRIES = 3


SWE_AGENT_REAL_ETL_PROMPT = """\
Write PyDeequ data unit tests for a EIDBench-real project.

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
{{
  "constraints": [
    {{"table_name": "customers", "column_name": "age",         "code_for_constraint": ".isComplete(\\"age\\")"}},
    {{"table_name": "customers", "column_name": "age",         "code_for_constraint": ".isNonNegative(\\"age\\")"}},
    {{"table_name": "customers", "column_name": "name",        "code_for_constraint": ".isComplete(\\"name\\")"}},
    {{"table_name": "orders",    "column_name": "customer_id", "code_for_constraint": ".isComplete(\\"customer_id\\")"}},
    {{"table_name": "orders",    "column_name": "amount",      "code_for_constraint": ".isNonNegative(\\"amount\\")"}}
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
{{
  "constraints": [
    {{"table_name": "products",  "column_name": "name",  "code_for_constraint": ".hasNumberOfDistinctValues(\\"name\\", lambda x: x > 0)"}},
    {{"table_name": "inventory", "column_name": "stock", "code_for_constraint": ".isComplete(\\"stock\\")"}},
    {{"table_name": "inventory", "column_name": "stock", "code_for_constraint": ".isNonNegative(\\"stock\\")"}}
  ]
}}

---------------------------------------

NOW I WILL GIVE YOU THE ACTUAL TASK DETAILS.

Input table profiles:
{tables_desc}

Sample rows by table:
{data_sample}

Project code (multiple files):
{code_context}

Downstream task description:
{downstream_task_description}

Focus on Python code for PyDeequ Check objects only. Create a file called
`constraints.json` in the current working directory with the shape shown in
the examples above. Make sure each entry has `table_name`, `column_name`, and
`code_for_constraint` fields. Do not include cross-table or foreign-key
checks. Do not run additional shell or python commands; just write the file.
"""


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
    samples: dict[str, list[dict[str, Any]]] = {}
    for table_name, dataframe in bundle.tables.items():
        if dataframe.empty or sample_rows <= 0:
            samples[table_name] = []
            continue
        n_rows = min(sample_rows, len(dataframe))
        samples[table_name] = dataframe.sample(n_rows, random_state=0).to_dict(orient="records")
    return samples


def build_prompt_content(
    project_manager: MultiTableProjectManager,
    prismadv_inputs: dict[str, Any],
    *,
    sample_rows: int,
) -> str:
    table_names = list(prismadv_inputs["context"]["script"]["reads"])
    input_variables = {
        "tables_desc": yaml.safe_dump(prismadv_inputs["table_profiles"], sort_keys=False),
        "data_sample": yaml.safe_dump(
            sample_rows_by_table(project_manager, table_names, sample_rows=sample_rows),
            sort_keys=False,
        ),
        "code_context": prismadv_inputs["code_context"]["combined_with_line_numbers"],
        "downstream_task_description": build_downstream_task_description(prismadv_inputs),
    }
    prompt = ChatPromptTemplate.from_template(SWE_AGENT_REAL_ETL_PROMPT)
    return prompt.invoke(input_variables).to_messages()[0].content


def route_for_model(model_name: str) -> str:
    """Map a bare model name to a litellm route mini-swe-agent understands."""
    if "/" in model_name:
        return model_name
    if model_name.startswith("gemini-"):
        return f"gemini/{model_name}"
    if model_name.startswith("claude-"):
        return f"anthropic/{model_name}"
    # default: OpenAI route (matches EID convention)
    return f"openai/{model_name}"


def make_output_filename(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{BASELINE_METHOD}--{model_name}--{timestamp}.yaml"


def existing_output_for_model(output_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(output_dir.glob(f"{BASELINE_METHOD}--{model_name}--*.yaml"))
    return candidates[-1] if candidates else None


def run_mini_swe_agent_call(
    prompt_content: str,
    *,
    model_route: str,
    cost_limit: float,
    work_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop the prompt as task.txt, invoke mini, parse constraints.json."""
    task_file = work_dir / "task.txt"
    task_file.write_text(prompt_content, encoding="utf-8")
    constraints_path = work_dir / "constraints.json"
    trajectory_path = work_dir / "trajectory.json"
    if constraints_path.exists():
        constraints_path.unlink()
    if trajectory_path.exists():
        trajectory_path.unlink()

    command = [
        "mini",
        "--model", model_route,
        "--task", "Solve the task specified in task.txt.",
        "--output", str(trajectory_path),
        "--cost-limit", f"{cost_limit:.2f}",
        "--yolo",
        "--exit-immediately",
    ]
    cwd_before = os.getcwd()
    try:
        os.chdir(work_dir)
        last_err: Exception | None = None
        for _ in range(MAX_RETRIES):
            try:
                subprocess.run(command, check=False, capture_output=True, text=True)
                raw_constraints = json.loads(constraints_path.read_text(encoding="utf-8"))[
                    "constraints"
                ]
                trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                return raw_constraints, trajectory
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise RuntimeError(
            f"mini-swe-agent failed after {MAX_RETRIES} attempts; last error: {last_err}"
        )
    finally:
        os.chdir(cwd_before)


def run_mini_swe_agent_real_etl(
    *,
    project_root: Path,
    example_id: str,
    input_path: Path | None = None,
    model_name: str = "gemini-2.5-flash",
    cost_limit: float = DEFAULT_COST_LIMIT,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    overwrite: bool = False,
    dq_manager: DeequDataQualityManager | None = None,
) -> Path:
    from workflow_prismadv.eid_bench_real_experiments import utils as exp_utils

    project_root = project_root.resolve()
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    output_dir = exp_utils.constraints_dir(example_id, "project", project_root, create=True)
    existing = existing_output_for_model(output_dir, model_name)
    if existing is not None and not overwrite:
        print(f"swe-agent EIDBench-real baseline already exists; skipping: {existing}")
        return existing

    prismadv_inputs, resolved_input_path = load_or_prepare_project_inputs(
        project_manager, input_path=input_path, overwrite=False,
    )
    prompt_content = build_prompt_content(
        project_manager, prismadv_inputs, sample_rows=sample_rows,
    )

    model_route = route_for_model(model_name)
    with tempfile.TemporaryDirectory(prefix="swe_agent_real_etl_") as tmp_dir:
        work_dir = Path(tmp_dir)
        print(
            f"Running mini-swe-agent in {work_dir} for example={example_id}, "
            f"model={model_name} ({model_route}), cost_limit=${cost_limit:.2f}"
        )
        raw_constraints, trajectory = run_mini_swe_agent_call(
            prompt_content,
            model_route=model_route,
            cost_limit=cost_limit,
            work_dir=work_dir,
        )

    constraints = constraints_with_validation(
        project_manager,
        raw_constraints,
        prismadv_inputs,
        dq_manager=dq_manager,
    )

    output_path = output_dir / make_output_filename(model_name)
    payload = {
        "baseline_method": BASELINE_METHOD,
        "model_name": model_name,
        "model_route": model_route,
        "cost_limit": cost_limit,
        "input_artifact": str(resolved_input_path),
        "raw_constraints": raw_constraints,
        "trajectory": trajectory,
        "constraints": constraints.to_dict()["constraints"],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
    print(f"Wrote: {output_path}")
    return output_path
