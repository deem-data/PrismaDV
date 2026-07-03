"""Descriptive-statistics data profiler + one-class kNN novelty detector.

Vendored and lightly adapted from the reference implementation accompanying:

    S. Redyuk, Z. Kaoudi, V. Markl, S. Schelter.
    "Automating Data Quality Validation for Dynamic Data Ingestion." EDBT 2021.
    https://github.com/sergred/automating-data-quality-validation-data (demo.py)

The original computes a per-batch feature vector of descriptive statistics
(completeness, uniqueness, approx-distinct via HyperLogLog, most-frequent ratio,
numeric min/mean/max/std/sum, and a text "index of peculiarity"), MinMax-scales
the vectors, and fits a one-class kNN detector (mean distance to k neighbours).
A new batch is flagged BAD (reject) if it is an outlier, GOOD (pass) otherwise.

Adaptations vs. the original ``demo.py``:
  * ``dabl.detect_types`` is dropped -- in the original it is computed but never
    used by ``compute_for`` (free-string detection falls back to ``dtype``).
  * ``nltk.util.ngrams`` is replaced by an inline sliding-window helper (identical
    output, no nltk data download).
  * The per-column metric set ("schema") is fixed from a reference batch so every
    batch yields a constant-length vector even if pandas infers a column's dtype
    differently on a corrupted batch. Columns declared numeric are coerced with
    ``pd.to_numeric(errors="coerce")`` and aggregated with NaN-aware reducers; any
    residual NaN in the final vector is set to 0.0 (the scaler/kNN cannot ingest NaN).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from hyperloglog import HyperLogLog
from pyod.models.knn import KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Column "kinds" decide which metrics are computed.
NUMERIC = "numeric"
OBJECT = "object"
OTHER = "other"

GENERIC_METRICS = ["Completeness", "Uniqueness", "ApproxCountDistinct", "FrequentRatio"]
NUMERIC_METRICS = ["Mean", "Minimum", "Maximum", "StandardDeviation", "Sum"]
OBJECT_METRICS = ["PeculiarityIndex"]


def _ngrams(sequence: str, n: int):
    """Yield consecutive n-grams of ``sequence`` (matches ``nltk.util.ngrams``)."""
    return zip(*[sequence[i:] for i in range(n)])


def _completeness(x: pd.Series) -> float:
    return 1.0 - np.sum(pd.isna(x)) / x.shape[0]


def _uniqueness(x: pd.Series) -> float:
    singletons = [c for c in Counter(x).values() if c == 1]
    return 1.0 * np.sum(singletons) / x.shape[0]


def _approx_count_distinct(x: pd.Series) -> float:
    hll = HyperLogLog(0.01)
    for val in x:
        hll.add(str(val))
    return float(len(hll))


def _frequent_ratio(x: pd.Series) -> float:
    counts = Counter(x)
    if not counts:
        return 0.0
    return 1.0 * max(counts.values()) / x.shape[0]


def _peculiarity(x: pd.Series) -> float:
    """Max over rows of the 3-gram "index of peculiarity" (Redyuk et al.)."""
    # Build n-gram frequency tables over the whole (aggregated) column once.
    aggregated = " ".join(map(str, x))
    c2gr = Counter(_ngrams(aggregated, 2))
    c3gr = Counter(_ngrams(aggregated, 3))

    def index_for_word(word) -> float:
        t = []
        for xyz in _ngrams(str(word), 3):
            xy, yz = xyz[:2], xyz[1:]
            cxy, cyz = c2gr.get(xy, 0), c2gr.get(yz, 0)
            cxyz = c3gr.get(xyz, 0)
            # Counts are taken from the same aggregated string, so any tri-gram
            # present in a word has cxy, cyz, cxyz >= 1 (no log(0)).
            t.append(0.5 * (np.log(cxy) + np.log(cyz) - np.log(cxyz)))
        if not t:
            return 0.0
        return float(np.sqrt(np.mean(np.array(t) ** 2)))

    values = x.apply(index_for_word)
    return float(values.max()) if len(values) else 0.0


def infer_column_kinds(reference: pd.DataFrame) -> Dict[str, str]:
    """Classify each column of a reference batch as numeric/object/other.

    The result is reused for every batch so feature vectors stay constant-length.
    """
    kinds: Dict[str, str] = {}
    for col, dtype in zip(reference.columns, reference.dtypes):
        if str(dtype) in ("int64", "float64", "Int64", "Float64"):
            kinds[col] = NUMERIC
        elif str(dtype) == "object":
            kinds[col] = OBJECT
        else:
            kinds[col] = OTHER
    return kinds


def metrics_for_kind(kind: str) -> List[str]:
    metrics = list(GENERIC_METRICS)
    if kind == NUMERIC:
        metrics.extend(NUMERIC_METRICS)
    elif kind == OBJECT:
        metrics.extend(OBJECT_METRICS)
    return metrics


def _compute_metric(name: str, x: pd.Series) -> float:
    if name == "Completeness":
        return _completeness(x)
    if name == "Uniqueness":
        return _uniqueness(x)
    if name == "ApproxCountDistinct":
        return _approx_count_distinct(x)
    if name == "FrequentRatio":
        return _frequent_ratio(x)
    if name == "PeculiarityIndex":
        return _peculiarity(x)
    # Numeric reducers (NaN-aware). A column declared numeric from the reference
    # batch may still arrive as object on a corrupted batch (e.g. '?' tokens), so
    # always coerce non-numeric tokens to NaN.
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")
    if np.all(np.isnan(arr)):
        return 0.0
    if name == "Mean":
        return float(np.nanmean(arr))
    if name == "Minimum":
        return float(np.nanmin(arr))
    if name == "Maximum":
        return float(np.nanmax(arr))
    if name == "StandardDeviation":
        return float(np.nanstd(arr))
    if name == "Sum":
        return float(np.nansum(arr))
    raise ValueError(f"Unknown metric: {name}")


def compute_profile(
    batch: pd.DataFrame,
    column_kinds: Dict[str, str],
    return_labels: bool = False,
):
    """Compute the descriptive-statistics feature vector for one batch.

    Args:
        batch: The batch data.
        column_kinds: Fixed mapping column -> kind (from ``infer_column_kinds``).
        return_labels: If True, also return the list of ``{col}_{metric}`` labels.
    """
    profile: List[float] = []
    labels: List[str] = []
    for col, kind in column_kinds.items():
        metrics = metrics_for_kind(kind)
        series = batch[col] if col in batch.columns else pd.Series([np.nan])
        for m in metrics:
            value = _compute_metric(m, series)
            profile.append(value)
            labels.append(f"{col}_{m}")
    # Final safety net: the scaler/kNN cannot ingest NaN/inf.
    profile = [0.0 if (v is None or not np.isfinite(v)) else float(v) for v in profile]
    return (profile, labels) if return_labels else profile


class KNNNoveltyDetector:
    """One-class kNN novelty detector (MinMax-scaled), per Redyuk et al."""

    def __init__(self, contamination: float = 0.01, n_neighbors: int = 5):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.clf: Pipeline | None = None

    def fit(self, history: List[List[float]]) -> "KNNNoveltyDetector":
        learner = KNN(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors,
            method="mean",
            metric="euclidean",
            algorithm="ball_tree",
        )
        self.clf = Pipeline(
            [("scaler", MinMaxScaler()), ("learner", learner)]
        ).fit(np.asarray(history, dtype="float64"))
        return self

    def predict(self, X: List[List[float]]) -> np.ndarray:
        """Return 0 for inlier (GOOD/pass) and 1 for outlier (BAD/reject)."""
        assert self.clf is not None, "call .fit() first"
        return self.clf.predict(np.asarray(X, dtype="float64"))


class SupervisedKNNClassifier:
    """Plain (uniform-weight) supervised kNN classifier over labeled batches.

    Unlike the one-class novelty detector, this USES both classes: it is trained on
    historical batches labeled GOOD (0) / BAD (1) and classifies a new batch by the
    majority vote of its k nearest labeled neighbours. Features are MinMax-scaled.

    NB: this is no longer the Redyuk (one-class) method -- it is a separate
    supervised-kNN baseline.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.clf: Pipeline | None = None

    def fit(self, X: List[List[float]], y: List[int]) -> "SupervisedKNNClassifier":
        self.clf = Pipeline([
            ("scaler", MinMaxScaler()),
            ("learner", KNeighborsClassifier(n_neighbors=self.n_neighbors, weights="uniform")),
        ]).fit(np.asarray(X, dtype="float64"), np.asarray(y, dtype="int"))
        return self

    def predict(self, X: List[List[float]]) -> np.ndarray:
        """Return 0 for GOOD/pass and 1 for BAD/reject."""
        assert self.clf is not None, "call .fit() first"
        return self.clf.predict(np.asarray(X, dtype="float64"))
