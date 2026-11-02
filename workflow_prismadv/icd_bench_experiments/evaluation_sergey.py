"""Evaluate the stats-novelty approach (column statistics + MMD permutation test) on ICDBench.

Same test as on EIDBench (workflow_prismadv/task_agnostic_baselines/novelty_models.py: seeded permutations, NaN
statistics read as 0): a batch is rejected when its column statistics differ from the case's sample (p < 0.05).
"""
import warnings

from workflow_prismadv.icd_bench_experiments.scoring import evaluate_task_agnostic, out_argument
from workflow_prismadv.task_agnostic_baselines.novelty_models import should_be_rejected_stats_novelty

warnings.filterwarnings("ignore", category=DeprecationWarning)


def fit(sample):
    return lambda batch: should_be_rejected_stats_novelty(sample, batch)


if __name__ == "__main__":
    args = out_argument(__doc__, "stats_novelty--stats_novelty--r1.tsv")
    evaluate_task_agnostic(fit, args.out)
