from datetime import datetime

import oyaml as yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.data_models import Constraints
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.llm_backend.entry import get_langchain_model
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

SINGLE_PROMPT = """
Write a pydeequ data unit test for a dataset consumed by the following program.

Here is sample of data that can be processed by the program:

{data_sample}

Here is the data profiling results:
{columns_desc}

Here is the program:

{code}

Focus on the python code for the Check object from pydeequ only.

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

The output must be a valid JSON object with a single key "constraints" that maps to a list of constraints. No other text should be included outside the JSON object.
"""


def run_single_shot(dataset_name, subtask_name, processed_data_label, model_name):
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
            if setup == "single_shot" and existing_name == model_name:
                print(f"Constraints file {constraint_output_path} already exists. Skipping...")
                skip = True
        if skip:
            continue
        source_code, assertions = FileLoader.load_py_file(script_path).extract_assertions()
        assert "# ASSERTION START" not in str(source_code)

        column_desc_dict = DeequInspectorManager().spark_df_to_column_desc_dict(spark_train, spark_train_data)
        column_desc = yaml.dump(column_desc_dict, default_flow_style=False, sort_keys=False)
        input_variables = {
            "code": source_code,
            "columns_desc": column_desc,
            "data_sample": example_rows,
        }
        print(f"Running PrismaDV for dataset: {dataset_name}, ")
        print(f"subtask: {subtask_name}, processed_data_label: {processed_data_label}, "
              f"script: {script_path.stem}, model: {model_name}")

        prompt = ChatPromptTemplate.from_template(SINGLE_PROMPT)
        llm = get_langchain_model(model_name, temperature=0.6)
        parser = JsonOutputParser()

        chain = prompt | llm | parser
        for attempt in range(max_retries):
            try:
                raw_constraints = chain.invoke(input_variables)['constraints']
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error during LLM invocation: {e}. Retrying ({attempt + 1}/{max_retries})...")
                    continue
                else:
                    raise e
        code_list_for_constraints = [item["code_for_constraint"] for item in raw_constraints]
        code_list_for_constraints_valid = dq_manager.filter_valid_constraints_on_spark(
            code_list_for_constraints, spark_train, spark_train_data)
        constraints = Constraints.from_deequ_output(raw_constraints, code_list_for_constraints_valid)
        constraints.save_to_yaml(constraint_output_path)


def make_output_filename(model_name) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"single_shot--{model_name}--{timestamp}.yaml"
