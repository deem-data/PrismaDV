from workflow_prismadv.eid_bench_baselines.pipelines.run_single_shot import run_prompt_baseline
from workflow_prismadv.utils.replicates import artifact_filename

METHOD = "few_shot"

FEW_SHOT_PROMPT = """
Write a pydeequ data unit test for a dataset consumed by the following program.

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


def run_few_shot(dataset_name, subtask_name, processed_data_label, model_name, replicate=None,
                 script_stems=None):
    # The single-shot loop with the few-shot prompt, so both rows log cost, failures and formats the same way.
    return run_prompt_baseline(METHOD, FEW_SHOT_PROMPT, dataset_name, subtask_name, processed_data_label,
                               model_name, replicate, script_stems)


def make_output_filename(model_name, replicate=None) -> str:
    return artifact_filename(METHOD, model_name, replicate)
