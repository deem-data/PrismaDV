"""Few-shot LLM constraint generation"""
import oyaml as yaml
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.llm_backend.entry import get_langchain_model
from workflow_prismadv.icd_bench_experiments import ALL_EVALUATION_CASES, find_fewshot_prompt_constraints_output_path

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

models = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-5-mini", "gpt-5", "gemini-2.5-flash", "gemini-2.5-pro"]

for model_name in models:
    for evaluation_case in ALL_EVALUATION_CASES:
        constraints_output_path = find_fewshot_prompt_constraints_output_path(evaluation_case, model_name)

        if constraints_output_path.exists():
            print(f"Constraints already exist at {constraints_output_path}. Skipping generation.")
            continue

        prompt = ChatPromptTemplate.from_template(FEW_SHOT_PROMPT)
        llm = get_langchain_model(model_name=model_name, temperature=0.6)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        data_sample = pd.DataFrame(evaluation_case.sample_data()).head(20).to_string(index=False)

        try:
            result = chain.invoke({
                "data_sample": data_sample,
                "code": evaluation_case.downstream_code(),
                "target_column": evaluation_case.target_column()
            })
        except Exception as e:
            print(
                f"Error generating constraints for {evaluation_case.__class__.__module__}.{evaluation_case.__class__.__name__}: {e}")
            result = {"constraints": []}

        print(result)

        with open(constraints_output_path, "w") as f:
            yaml.dump(result, f, default_flow_style=False, sort_keys=False)
            print(f"... constraints saved to {constraints_output_path}")
