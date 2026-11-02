"""Novelty detectors fitted on the observed data: one-class SVM, isolation forest, stats-novelty (MMD test).
Implementation: workflow_prismadv/task_agnostic_baselines/novelty_models.py (seeded; same on all benchmarks).

  synth  one result per label (eid_bench_baselines/{one_class_svm,isolation_forest,stats_novelty}.py)
  real   one artifact per app, one result per label (eid_bench_real_baselines/<same names>.py)
  icd    icd_bench_experiments/evaluation_{oneclasssvm,isolationforest,sergey}.py --out

  uv run python -m workflow_prismadv.results.methods.novelty     # this family's rows only
"""
from workflow_prismadv.results.methods._common import Row, icd_arms, real_arms, synth_per_label

FAMILY = "novelty"
ROWS = [Row("ocsvm", "one-class-svm", False, False),
        Row("iforest", "isolation-forest", False, False),
        Row("stats-nov", "stats-novelty", False, False)]
STEM = {"ocsvm": "one_class_svm", "iforest": "isolation_forest", "stats-nov": "stats_novelty"}


def synth():
    return synth_per_label({k: f"{v}_constraints" for k, v in STEM.items()})


def real():
    return real_arms({k: (v, v) for k, v in STEM.items()}, reps=(1,))


def icd():
    return icd_arms({k: f"{v}--{v}" for k, v in STEM.items()}, reps=(1,))


if __name__ == "__main__":
    from workflow_prismadv.results.combined_table import main
    raise SystemExit(main(["--family", FAMILY]))
