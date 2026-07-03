"""Redyuk et al. (EDBT 2021) task-agnostic novelty-detection baseline.

A standalone, non-LLM batch-level data-quality validator. It is a baseline ONLY
for the *New Data Batches* scenario, since it models the data distribution of
previously-observed batches and never sees task code or task outcomes.
"""

from workflow_sifta.baselines.redyuk.data_profiler import (
    KNNNoveltyDetector,
    SupervisedKNNClassifier,
    compute_profile,
    infer_column_kinds,
)

__all__ = [
    "KNNNoveltyDetector",
    "SupervisedKNNClassifier",
    "compute_profile",
    "infer_column_kinds",
]
