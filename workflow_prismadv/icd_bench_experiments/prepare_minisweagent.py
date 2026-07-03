"""Setup for SWE-Agent evaluation"""
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from workflow_prismadv.icd_bench_experiments import find_minisweagent_task_output_path, ALL_EVALUATION_CASES

TASK_PROMPT = """
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
        ".isComplete('age')",
        ".isNonNegative('age')",
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
        ".hasNumberOfDistinctValues('name', lambda x: x > 0)",
    ]
}} 


---------------------------------------

NOW I WILL GIVE YOU THE ACTUAL TASK DETAILS.

Here is sample of data that can be processed by the program:

{data_sample}

Here is the program:

{code}

Focus on the python code for the Check object from pydeequ only. Create a file called constraints.json as output. This JSON should have with list of strings, where each string contains the executable code for a constraint on the Check object. Return constraints for the {target_column} column only.

Here is an example output:

{{
    "constraints": [
        ".isComplete('some_column')",
        ".isPositive('another_column')",
    ]
}}    
"""

for evaluation_case in ALL_EVALUATION_CASES:
    task_output_path = find_minisweagent_task_output_path(evaluation_case)

    if task_output_path.exists():
        print(f"Description already exist at {task_output_path}. Skipping generation.")
        continue

    prompt = ChatPromptTemplate.from_template(TASK_PROMPT)

    chain = prompt

    data_sample = pd.DataFrame(evaluation_case.sample_data()).head(20).to_string(index=False)

    try:
        result = chain.invoke({
            "data_sample": data_sample,
            "code": evaluation_case.downstream_code(),
            "target_column": evaluation_case.target_column(),
        })

        task = result.messages[0].content

        with open(task_output_path, "w") as f:
            f.write(task)
            print(f"... task details saved to {task_output_path}")

    except Exception as e:
        print(
            f"Error generating constraints for {evaluation_case.__class__.__module__}.{evaluation_case.__class__.__name__}: {e}")
        result = {"constraints": []}
