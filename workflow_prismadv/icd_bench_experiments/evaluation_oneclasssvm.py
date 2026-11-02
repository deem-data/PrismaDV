"""Evaluate One-Class SVM on ICDBench.

Same detector as on EIDBench (workflow_prismadv/task_agnostic_baselines/novelty_models.py): fitted on each case's
sample; a batch is rejected when any of its rows is flagged.
"""
import warnings

from workflow_prismadv.icd_bench_experiments.scoring import evaluate_task_agnostic, out_argument
from workflow_prismadv.task_agnostic_baselines.novelty_models import learn_one_class_svm, should_be_rejected

warnings.filterwarnings("ignore", category=DeprecationWarning)


def fit(sample):
    detector = learn_one_class_svm(sample)
    return lambda batch: should_be_rejected(detector, batch)


if __name__ == "__main__":
    args = out_argument(__doc__, "one_class_svm--one_class_svm--r1.tsv")
    evaluate_task_agnostic(fit, args.out)
