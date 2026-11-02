"""Run the PocketFlow code-generation agent to produce EIDBench-synth constraints."""
import json
import os
import time

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
    get_usage_summary,
    reset_message_log,
)
from workflow_prismadv.utils.spark_cleanup import close_spark_session
from workflow_prismadv.utils.replicates import select_scripts, artifact_filename, existing_artifacts, normalize_replicate

METHOD = "pocketflow"
MODEL_TAG = "gpt-5"
MAX_ITERATIONS = 3
MAX_RETRIES = 3

POCKETFLOW_PROBLEM = """Write a pydeequ data unit test for a dataset consumed by the following program.

Here are two examples of data samples and the corresponding constraints that should be generated.
---------------------------------------

### First Example:

Here is sample of data that can be processed by the program:

 name  age
Alice   25
  Bob   30
Carol   22
 Dave   40
  Eve   28

Here is the program:

for row['age'] in df.iterrows():
    if row['age'] < 0:
        raise ValueError("Age cannot be negative")

Here is the desired output:

def run_code():
    return {{
        "constraints": [
            {{"column_name": "age", "code_for_constraint": ".isComplete('age')"}},
            {{"column_name": "age", "code_for_constraint": ".isNonNegative('age')"}},
        ]
    }}

### Second Example:

Here is sample of data that can be processed by the program:

 name  age
Alice   25
 NULL   30
Carol   22
 NULL   40
  Eve   28

Here is the program:

unique_names = df['name'].dropna().unique().tolist()
send_notifications(unique_names)

Here is the desired output:

def run_code():
    return {{
        "constraints": [
            {{"column_name": "name", "code_for_constraint": ".hasNumberOfDistinctValues('name', lambda x: x > 0)"}},
        ]
    }}

---------------------------------------

NOW I WILL GIVE YOU THE ACTUAL TASK DETAILS.

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


def run_pocketflow_flow(problem, model_name=MODEL_TAG):
    os.environ["POCKETFLOW_MODEL"] = model_name
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


def make_output_filename(model_name, replicate=None) -> str:
    return artifact_filename(METHOD, model_name, replicate)


def run_pocketflow(dataset_name, subtask_name, processed_data_label, model_name=MODEL_TAG, replicate=None,
                   script_stems=None):
    replicate = normalize_replicate(replicate)
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)

    dq_manager = DeequDataQualityManager()
    train_data = FileLoader.load_csv(
        project_manager.get_observed_data_path(subtask_name, processed_data_label)
    )
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    example_rows = train_data.sample(3).to_dict(orient="records")

    script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name=subtask_name)
    for script_path in select_scripts(script_path_list, script_stems):
        constraints_path = project_manager.get_source_constraints_path(subtask_name, script_path.stem)
        constraints_path.mkdir(parents=True, exist_ok=True)
        constraint_output_path = constraints_path / make_output_filename(model_name, replicate)
        existing = existing_artifacts(constraints_path, METHOD, model_name, replicate)
        if existing:
            print(f"Constraints file {existing[-1]} already exists (replicate {replicate}). Skipping...")
            continue

        # Same span as single/few-shot's wall_clock_seconds: source load, profiling, every attempt, filter_valid.
        started = time.perf_counter()
        source_code, assertions = FileLoader.load_py_file(script_path).extract_assertions()
        assert "# ASSERTION_START" not in str(source_code)

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
        # Reset outside the retry loop so a retried task reports the cost of every attempt.
        reset_message_log()
        for attempt in range(MAX_RETRIES):
            try:
                raw_constraints = run_pocketflow_flow(problem, model_name)
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
        constraints.cost_summary = {"constraint_generation": get_usage_summary()}
        constraints.save_to_yaml(constraint_output_path, extra={
            "replicate": replicate,
            "wall_clock_seconds": round(time.perf_counter() - started, 3),
        })
        messages_path = constraint_output_path.with_name(constraint_output_path.stem + ".messages.json")
        messages_path.write_text(json.dumps({"messages": get_message_log()}, indent=2))
    close_spark_session(spark_train)
