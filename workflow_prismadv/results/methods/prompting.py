"""Prompting baselines: zero-shot (single prompt) and few-shot constraint generation with gemini-3.8-flash and
gpt-6-sol (medium), replicates r1-r3. Their constraints are validated with Deequ like PrismaDV's.

  synth  per task: general/constraints/<task>/{single_shot,few_shot}--<model>--*.yaml
  real   constraints/project/{single_shot,few_shot}--<model>[--r<N>]--<timestamp>.yaml
  icd    icd_bench_experiments/evaluation_{single_prompt,fewshot}.py --out

  uv run python -m workflow_prismadv.results.methods.prompting
"""
from workflow_prismadv.results.methods._common import Row, icd_arms, real_arms, synth_per_task

FAMILY = "prompting"
G38, G6 = "gemini-3.8-flash", "gpt-6-sol-medium"
ROWS = [Row("single-g38", "zero-shot [gemini-3.8]", True, True),
        Row("single-6sol", "zero-shot [gpt-6]", True, True),
        Row("few-g38", "few-shot [gemini-3.8]", True, True),
        Row("few-6sol", "few-shot [gpt-6]", True, True)]
ARTIFACT = {"single-g38": ("single_shot", G38), "single-6sol": ("single_shot", G6),
            "few-g38": ("few_shot", G38), "few-6sol": ("few_shot", G6)}


def synth():
    return synth_per_task({k: f"{p}--{m}--*" for k, (p, m) in ARTIFACT.items()})


def real():
    return real_arms(ARTIFACT)


def icd():
    return icd_arms({k: f"{p}--{m}" for k, (p, m) in ARTIFACT.items()})


if __name__ == "__main__":
    from workflow_prismadv.results.combined_table import main
    raise SystemExit(main(["--family", FAMILY]))
