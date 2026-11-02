"""Evaluate the few-shot LLM baseline on ICDBench.

Every constraint the model returned is scored, unfiltered (see `llm_baseline.py`).

    python -m workflow_prismadv.icd_bench_experiments.evaluation_fewshot --model_name gpt-6-sol-medium
"""
from workflow_prismadv.icd_bench_experiments import find_fewshot_prompt_constraints_output_path
from workflow_prismadv.icd_bench_experiments.llm_baseline import run_evaluation

if __name__ == "__main__":
    run_evaluation("fewshot_prompt", find_fewshot_prompt_constraints_output_path)
