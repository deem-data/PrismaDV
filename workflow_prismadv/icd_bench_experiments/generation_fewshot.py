"""Few-shot LLM constraint generation on ICDBench.

    python -m workflow_prismadv.icd_bench_experiments.generation_fewshot --model_name gpt-6-sol-medium
"""
import os
import sys

from workflow_prismadv.icd_bench_experiments import find_fewshot_prompt_constraints_output_path
from workflow_prismadv.icd_bench_experiments.llm_baseline import run_generation

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

Focus on the python code for the Check object from pydeequ only. Reply with a JSON object with list of strings, where each string contains the executable code for a constraint on the Check object. Return constraints for the {target_column} column only.

Here is an example output:

{{
    "constraints": [
        ".isComplete('some_column')",
        ".isPositive('another_column')",
    ]
}}    
"""

if __name__ == "__main__":
    code = run_generation("fewshot_prompt", FEW_SHOT_PROMPT, find_fewshot_prompt_constraints_output_path)
    sys.stdout.flush()
    os._exit(code)
