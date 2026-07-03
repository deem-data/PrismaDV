"""Analyze constraint generation patterns across different methods"""
import os
import warnings

import oyaml as yaml

from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning)

from workflow_prismadv.icd_bench_experiments import find_diverse_constraints_output_path, instantiate_evaluation_case

num_constraints = {
    'pocketflow_constraints': 0,
    'prismadv_constraints--gemini-2.5-flash': 0,
    'prismadv_constraints--gemini-2.5-pro': 0,
    'prismadv_constraints--gpt-4.1': 0,
    'prismadv_constraints--gpt-5': 0,
    'prismadv_constraints--gpt-5-mini': 0,
    'prismadv_constraints--gpt-4.1-nano': 0,
    'prismadv_constraints--gpt-4o-mini': 0,
    'single_prompt_constraints--gemini-2.5-flash': 0,
    'single_prompt_constraints--gemini-2.5-pro': 0,
    'single_prompt_constraints--gpt-4.1': 0,
    'single_prompt_constraints--gpt-5': 0,
    'single_prompt_constraints--gpt-4.1-nano': 0,
    'single_prompt_constraints--gpt-4o-mini': 0,
    'single_prompt_constraints--gpt-5-mini': 0,
    'fewshot_prompt_constraints--gemini-2.5-flash': 0,
    'fewshot_prompt_constraints--gemini-2.5-pro': 0,
    'fewshot_prompt_constraints--gpt-4.1': 0,
    'fewshot_prompt_constraints--gpt-5': 0,
    'fewshot_prompt_constraints--gpt-4.1-nano': 0,
    'fewshot_prompt_constraints--gpt-4o-mini': 0,
    'fewshot_prompt_constraints--gpt-5-mini': 0,
}

num_cases = {
    'pocketflow_constraints': 0,
    'prismadv_constraints--gemini-2.5-flash': 0,
    'prismadv_constraints--gemini-2.5-pro': 0,
    'prismadv_constraints--gpt-4.1': 0,
    'prismadv_constraints--gpt-5': 0,
    'prismadv_constraints--gpt-5-mini': 0,
    'prismadv_constraints--gpt-4.1-nano': 0,
    'prismadv_constraints--gpt-4o-mini': 0,
    'single_prompt_constraints--gemini-2.5-flash': 0,
    'single_prompt_constraints--gemini-2.5-pro': 0,
    'single_prompt_constraints--gpt-4.1': 0,
    'single_prompt_constraints--gpt-5': 0,
    'single_prompt_constraints--gpt-4.1-nano': 0,
    'single_prompt_constraints--gpt-4o-mini': 0,
    'single_prompt_constraints--gpt-5-mini': 0,
    'fewshot_prompt_constraints--gemini-2.5-flash': 0,
    'fewshot_prompt_constraints--gemini-2.5-pro': 0,
    'fewshot_prompt_constraints--gpt-4.1': 0,
    'fewshot_prompt_constraints--gpt-5': 0,
    'fewshot_prompt_constraints--gpt-4.1-nano': 0,
    'fewshot_prompt_constraints--gpt-4o-mini': 0,
    'fewshot_prompt_constraints--gpt-5-mini': 0,
}

constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench")
for case_dir in os.listdir(constraints_output_dir):
    evaluation_case = instantiate_evaluation_case(case_dir)

    for approach in num_constraints.keys():
        constraints_output_path = find_diverse_constraints_output_path(evaluation_case, approach)
        if not os.path.exists(constraints_output_path):
            continue

        with open(constraints_output_path, "r") as f:
            res_dict = yaml.safe_load(f)

        try:
            target_column = evaluation_case.target_column()
            if approach.startswith('prismadv_constraints'):
                if target_column in res_dict['constraints']:
                    for code in res_dict['constraints'][target_column]['code']:
                        if not code['suggestion'].startswith('.hasDataType'):
                            num_constraints[approach] += 1
            else:
                num_constraints[approach] += len(res_dict['constraints'])

            num_cases[approach] += 1

        except Exception as e:
            print(f"Error processing {constraints_output_path} for {approach}: {e}")

for approach in num_constraints.keys():
    print(f"{approach}: {num_constraints[approach]} constraints / {num_cases[approach]} cases")
