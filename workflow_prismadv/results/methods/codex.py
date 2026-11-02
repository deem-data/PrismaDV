"""Codex CLI agent (gpt-6-sol, medium reasoning) writing constraints, replicates r1-r3.

  synth  per task: general/constraints/<task>/codex--gpt-6-sol--medium*.yaml
  real   constraints/project/codex--gpt-6-sol[--r<N>]--<timestamp>.yaml
  icd    icd_bench_experiments/evaluation_codex.py --out

  uv run python -m workflow_prismadv.results.methods.codex
"""
from workflow_prismadv.results.methods._common import Row, icd_arms, real_arms, synth_per_task

FAMILY = "codex"
ROWS = [Row("codex", "codex [gpt-6]", True, True)]


def synth():
    return synth_per_task({"codex": "codex--gpt-6-sol--medium*"})


def real():
    return real_arms({"codex": ("codex", "gpt-6-sol")})


def icd():
    return icd_arms({"codex": "codex--gpt-6-sol"})


if __name__ == "__main__":
    from workflow_prismadv.results.combined_table import main
    raise SystemExit(main(["--family", FAMILY]))
