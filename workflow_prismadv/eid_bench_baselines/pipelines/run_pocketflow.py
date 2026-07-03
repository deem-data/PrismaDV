"""Run the PocketFlow code-generation agent to produce EIDBench-synth constraints."""
import json
import os
from datetime import datetime

import oyaml as yaml

from prismadv.data_models import Constraints
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.icd_bench_experiments.pocketflow.flow import create_code_generator_flow
from workflow_prismadv.icd_bench_experiments.pocketflow.utils.code_executor import execute_python
from workflow_prismadv.icd_bench_experiments.pocketflow.utils.call_llm import (
    get_message_log,
    reset_message_log,
)

MODEL_TAG = "gpt-5"
MAX_ITERATIONS = 3
MAX_RETRIES = 3

POCKETFLOW_PROBLEM = """Write a pydeequ data unit test for a dataset consumed by the following program.

Here is a sample of data that can be processed by the program:
{data_sample}

Here is the data profiling results:
{columns_desc}

Here is the program:
{code}

Focus on the python code for the Check object from pydeequ only. Generate a Python function
`run_code()` (taking no arguments) that returns a JSON object with a single key "constraints"
mapping to a list of objects, each with keys "column_name" and "code_for_constraint", where
"code_for_constraint" is executable pydeequ Check code for that column. Your code must not inspect
any dataframe; only return the constraints.

Here is an example of the code to generate, please replace the example constraints with the ones
you generate:

def run_code():
    return {{
        "constraints": [
            {{"column_name": "some_column", "code_for_constraint": ".isComplete('some_column')"}},
            {{"column_name": "another_column", "code_for_constraint": ".isPositive('another_column')"}},
        ]
    }}
"""


def run_pocketflow_flow(problem):
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


def make_output_filename(model_name) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"pocketflow--{model_name}--{timestamp}.yaml"


def run_pocketflow(dataset_name, subtask_name, processed_data_label, model_name=MODEL_TAG):
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)

    dq_manager = DeequDataQualityManager()
    train_data = FileLoader.load_csv(
        project_manager.get_observed_data_path(subtask_name, processed_data_label)
    )
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    example_rows = train_data.sample(3).to_dict(orient="records")

    script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name=subtask_name)
    for script_path in script_path_list:
        constraints_path = project_manager.get_constraints_path(
            subtask_name, processed_data_label, script_path.stem
        )
        constraints_path.mkdir(parents=True, exist_ok=True)
        constraint_output_path = constraints_path / make_output_filename(model_name)

        skip = False
        for path in constraints_path.glob("*.yaml"):
            parts = path.stem.split("--")
            if len(parts) >= 2 and parts[0] == "pocketflow" and parts[1] == model_name:
                print(f"Constraints file for pocketflow/{model_name} already exists ({path}). Skipping...")
                skip = True
        if skip:
            continue

        source_code, assertions = FileLoader.load_py_file(script_path).extract_assertions()
        assert "# ASSERTION START" not in str(source_code)

        column_desc_dict = DeequInspectorManager().spark_df_to_column_desc_dict(spark_train, spark_train_data)
        column_desc = yaml.dump(column_desc_dict, default_flow_style=False, sort_keys=False)
        problem = POCKETFLOW_PROBLEM.format(
            code=source_code,
            columns_desc=column_desc,
            data_sample=example_rows,
        )

        print(f"Running PocketFlow for dataset: {dataset_name}, subtask: {subtask_name}, "
              f"processed_data_label: {processed_data_label}, script: {script_path.stem}, model: {model_name}")

        raw_constraints = []
        for attempt in range(MAX_RETRIES):
            try:
                raw_constraints = run_pocketflow_flow(problem)
                break
            except Exception as e:  # noqa: BLE001
                if attempt < MAX_RETRIES - 1:
                    print(f"Error during PocketFlow run: {e}. Retrying ({attempt + 1}/{MAX_RETRIES})...")
                    continue
                raise

        code_list_for_constraints = [item["code_for_constraint"] for item in raw_constraints]
        code_list_for_constraints_valid = dq_manager.filter_valid_constraints_on_spark(
            code_list_for_constraints, spark_train, spark_train_data
        )
        constraints = Constraints.from_deequ_output(raw_constraints, code_list_for_constraints_valid)
        constraints.save_to_yaml(constraint_output_path)
        messages_path = constraint_output_path.with_name(constraint_output_path.stem + ".messages.json")
        messages_path.write_text(json.dumps({"messages": get_message_log()}, indent=2))
