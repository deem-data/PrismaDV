"""PrismaDV with gemini-3.8-flash and gpt-6-sol (medium), replicates r1-r3.

  synth  per task: general/constraints/<task>/prismadv--<model>--*.yaml
  real   constraints/project/prismadv--<model>[--r<N>]--<timestamp>.yaml
  icd    icd_bench_experiments/evaluation_prismadv.py --out

  uv run python -m workflow_prismadv.results.methods.prismadv
"""
from workflow_prismadv.results.methods._common import Row, icd_arms, real_arms, synth_per_task

FAMILY = "prismadv"
G38, G6 = "gemini-3.8-flash", "gpt-6-sol-medium"
ROWS = [Row("P-g38", "prismaDV [gemini-3.8]", True, True),
        Row("P-6sol", "prismaDV [gpt-6]", True, True)]
ARTIFACT = {"P-g38": ("prismadv", G38), "P-6sol": ("prismadv", G6)}


def synth():
    return synth_per_task({k: f"{p}--{m}--*" for k, (p, m) in ARTIFACT.items()})


def real():
    return real_arms(ARTIFACT)


def icd():
    return icd_arms({k: f"{p}--{m}" for k, (p, m) in ARTIFACT.items()})


if __name__ == "__main__":
    from workflow_prismadv.results.combined_table import main
    raise SystemExit(main(["--family", FAMILY]))
