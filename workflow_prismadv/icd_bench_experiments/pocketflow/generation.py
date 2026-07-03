from pathlib import Path

import oyaml as yaml

from prismadv.utils import get_project_root
from workflow_prismadv.icd_bench_experiments import ALL_EVALUATION_CASES
from workflow_prismadv.icd_bench_experiments.pocketflow.flow import create_code_generator_flow
from workflow_prismadv.icd_bench_experiments.pocketflow.utils.code_executor import execute_python


def find_pocketflow_constraints_output_path(evaluation_case):
    target_dir = evaluation_case.__class__.__module__.split('.')[-1] + "." + evaluation_case.__class__.__name__
    constraints_output_dir = (
            get_project_root() / "data_processed" / "icd_bench" / target_dir / "constraints"
    )
    Path(constraints_output_dir).mkdir(parents=True, exist_ok=True)
    constraints_output_path = (constraints_output_dir / f"pocketflow_constraints.yaml")
    return constraints_output_path


for evaluation_case in ALL_EVALUATION_CASES:
    constraints_output_path = find_pocketflow_constraints_output_path(evaluation_case)

    if constraints_output_path.exists():
        print(f"Constraints already exist at {constraints_output_path}. Skipping generation.")
        continue

    case_id = f"{evaluation_case.__class__.__module__}.{evaluation_case.__class__.__name__}"
    print(f"Starting PocketFlow Code Generator for {case_id}...")
    import pandas as pd

    data_sample = pd.DataFrame(evaluation_case.sample_data()).head(20).to_string(index=False)

    problem = f"""`
Write a pydeequ data unit test for a dataset consumed by a program snippet.

Here is sample of data that can be processed by the program snippet:

{data_sample}

Here is the the snippet of the program:

{evaluation_case.downstream_code()}

Focus on the python code for the Check object from pydeequ only. Generate a Python function that returns a JSON object with a list of strings, where each string contains the executable code for a constraint on the Check object. Return constraints for the {evaluation_case.target_column()} column only. Note that your code to generate should not inspect a dataframe, but only return the constraints required for a PyDeequ unit test.

Here is an example of the code to generate, please replace the example constraints with the ones you generated:

def run_code():
  return {{
  	"constraints": [
		".isComplete('some_column')",
		".isPositive('some_column')",
  ]
}}
"""

    shared = {
        "problem": problem,
        "test_cases": [],
        "function_code": "",
        "test_results": [],
        "iteration_count": 0,
        "max_iterations": 3
    }

    generated_constraints = {"constraints": []}

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            flow = create_code_generator_flow()
            flow.run(shared)
            result, _ = execute_python(shared['function_code'], input={})
            if result:
                generated_constraints = result
            break  # Exit loop if successful

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print("Retrying...")
            else:
                print("All retries failed. Giving up.")

    print(generated_constraints)
    with open(constraints_output_path, "w") as f:
        yaml.dump(generated_constraints, f, default_flow_style=False, sort_keys=False)
        print(f"... constraints saved to {constraints_output_path}")
