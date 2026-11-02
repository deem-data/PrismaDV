"""Zero-shot LLM constraint generation on ICDBench.

    python -m workflow_prismadv.icd_bench_experiments.generation_single_prompt --model_name gpt-6-sol-medium
"""
import os
import sys

from workflow_prismadv.icd_bench_experiments import find_single_prompt_constraints_output_path
from workflow_prismadv.icd_bench_experiments.llm_baseline import run_generation

SINGLE_PROMPT = """
Write a pydeequ data unit test for a dataset consumed by the following program.

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
    code = run_generation("single_prompt", SINGLE_PROMPT, find_single_prompt_constraints_output_path)
    sys.stdout.flush()
    os._exit(code)
