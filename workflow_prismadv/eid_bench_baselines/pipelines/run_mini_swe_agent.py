import json
import os
import subprocess
import tempfile
from datetime import datetime

import oyaml as yaml
from langchain_core.prompts import ChatPromptTemplate

from prismadv.data_models import Constraints
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

TASK_PROMPT = """Write a pydeequ data unit test for a dataset consumed by the following program.

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

{{
    "constraints": [
        {{
            'column_name': 'age',
            'code_for_constraint': ".isComplete('age')",
        }},
        {{
            'column_name': 'age',
            'code_for_constraint': ".isNonNegative('age')",
        }},   
    ]
}}

# Second Example:

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

{{
    "constraints": [
        {{
            'column_name': 'name',
            'code_for_constraint': ".hasNumberOfDistinctValues('name', lambda x: x > 0)",
        }},
        
    ]
}}


---------------------------------------

NOW I WILL GIVE YOU THE ACTUAL TASK DETAILS.

Here is sample of data that can be processed by the program:

{data_sample}

Here is the program:

{code}

Focus on the python code for the Check object from pydeequ only. Create a file called `constraints.json` as output.

Here is an example output:

{{
    "constraints": [
        {{
            'column_name': 'some_column',
            'code_for_constraint': ".isComplete('some_column')",
        }},
        {{
            'column_name': 'another_column',
            'code_for_constraint': ".isPositive('another_column')",
        }}
    ]
}}

Please make sure the constraints.json is the output file in the current folder.
"""


def run_mini_swe_agent(dataset_name, subtask_name, processed_data_label, model_name):
    max_retries = 3
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
            setup = path.stem.split("--")[0]
            existing_name = path.stem.split("--")[1]
            if setup == "mini_swe_agent" and existing_name == model_name:
                print(f"Constraints file {constraint_output_path} already exists. Skipping...")
                skip = True
        if skip:
            continue
        source_code, assertions = FileLoader.load_py_file(script_path).extract_assertions()
        assert "# ASSERTION START" not in str(source_code)

        input_variables = {
            "code": source_code,
            "data_sample": example_rows,
        }
        print(f"Running SWE-agent for dataset: {dataset_name}, ")
        print(f"subtask: {subtask_name}, processed_data_label: {processed_data_label}, "
              f"script: {script_path.stem}, model: {model_name}")
        prompt = ChatPromptTemplate.from_template(TASK_PROMPT)
        prompt_content = prompt.invoke(input_variables).to_messages()[0].content
        output = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            task_file_path = os.path.join(temp_dir, "task.txt")
            with open(task_file_path, "w", encoding="utf-8") as f:
                f.write(prompt_content)

            print(f"Running Mini SWE Agent in {temp_dir}")
            for attempt in range(max_retries):
                try:
                    command = f"mini --model \"openai/{model_name}\" --task \"Solve the task specified in task.txt.\" --output trajectory.json --cost-limit 0.50 --yolo --exit-immediately"
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    print(result)
                    with open("./constraints.json", "r", encoding="utf-8") as f:
                        raw_constraints = json.load(f)['constraints']
                    with open("./trajectory.json", "r", encoding="utf-8") as f:
                        trajectory_container = json.load(f)

                    code_list_for_constraints = [item["code_for_constraint"] for item in raw_constraints]
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise e

            code_list_for_constraints_valid = dq_manager.filter_valid_constraints_on_spark(
                code_list_for_constraints, spark_train, spark_train_data)
            constraints = Constraints.from_deequ_output(raw_constraints, code_list_for_constraints_valid)

            output['raw_constraints'] = raw_constraints
            output['trajectory'] = trajectory_container
            output['constraints'] = constraints.to_dict()
            with open(constraint_output_path, "w", encoding="utf-8") as f:
                yaml.dump(output, f, allow_unicode=True)


def make_output_filename(model_name) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"mini_swe_agent--{model_name}--{timestamp}.yaml"
