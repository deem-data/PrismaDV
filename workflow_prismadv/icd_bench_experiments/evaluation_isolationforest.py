"""Evaluate Isolation Forest on ICDBench.

Same detector as on EIDBench (workflow_prismadv/task_agnostic_baselines/novelty_models.py, seeded): fitted on each
case's sample; a batch is rejected when any of its rows is flagged.
"""
import warnings

from workflow_prismadv.icd_bench_experiments.scoring import evaluate_task_agnostic, out_argument
from workflow_prismadv.task_agnostic_baselines.novelty_models import learn_isolation_forest, should_be_rejected

warnings.filterwarnings("ignore", category=DeprecationWarning)


def fit(sample):
    detector = learn_isolation_forest(sample)
    return lambda batch: should_be_rejected(detector, batch)


if __name__ == "__main__":
    args = out_argument(__doc__, "isolation_forest--isolation_forest--r1.tsv")
    evaluate_task_agnostic(fit, args.out)
