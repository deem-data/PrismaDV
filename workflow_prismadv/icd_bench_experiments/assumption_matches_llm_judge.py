"""LLM-as-judge for evaluating assumption quality"""
import os
import warnings

import oyaml as yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning)
from . import find_constraints_output_path, instantiate_evaluation_case, ASSUMPTION_MATCH_PROMPT, RED, RESET

model_name = "gpt-4.1"

num_cases = 0
num_matches = 0

failed_case_names = []

constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench")
for case_dir in os.listdir(constraints_output_dir):

    num_cases += 1

    evaluation_case = instantiate_evaluation_case(case_dir)
    constraints_output_path = find_constraints_output_path(evaluation_case, model_name)

    if os.path.exists(constraints_output_path):
        with open(constraints_output_path, "r") as f:
            res_dict = yaml.safe_load(f)

        candidate_assumptions = []

        target_column = evaluation_case.target_column()
        if target_column in res_dict['constraints']:
            for assumption in res_dict['constraints'][target_column]['assumptions']:
                candidate_assumptions.append(assumption['text'])

        ground_truth = evaluation_case.assumption_in_natural_language()
        candidate_assumption_list = ""

        for index, candidate in enumerate(candidate_assumptions):
            candidate_assumption_list += f"- index: {index}; candidate_assumption: {candidate}\n"

        prompt = ChatPromptTemplate.from_template(ASSUMPTION_MATCH_PROMPT)
        llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.6)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({
            "ground_truth": ground_truth,
            "candidate_assumption_list": candidate_assumption_list
        })

        print(f"\n\n### {evaluation_case.__class__.__module__}.{evaluation_case.__class__.__qualname__}")
        print("### Ground truth:")
        print(ground_truth)

        if "chosen_assumption" in result and result["chosen_assumption"] != -1:
            num_matches += 1
            print("### Matching assumption:")
            print(candidate_assumptions[result["chosen_assumption"]])
        else:
            failed_case_names.append(f"{evaluation_case.__class__.__module__}.{evaluation_case.__class__.__qualname__}")
            print(f"{RED}No matching assumption found.{RESET}")
    else:
        print(f"{RED}Constraint file {constraints_output_path} not found .{RESET}")

print(f"\n\nFailed cases: {failed_case_names}")

print(f"\n\n{num_matches}/{num_cases} assumptions identified by model {model_name}.")
