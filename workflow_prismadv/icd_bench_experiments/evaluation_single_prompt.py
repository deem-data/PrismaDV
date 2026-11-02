"""Evaluate the zero-shot (single-prompt) LLM baseline on ICDBench.

Every constraint the model returned is scored, unfiltered (see `llm_baseline.py`).

    python -m workflow_prismadv.icd_bench_experiments.evaluation_single_prompt --model_name gpt-6-sol-medium
"""
from workflow_prismadv.icd_bench_experiments import find_single_prompt_constraints_output_path
from workflow_prismadv.icd_bench_experiments.llm_baseline import run_evaluation

if __name__ == "__main__":
    run_evaluation("single_prompt", find_single_prompt_constraints_output_path)
