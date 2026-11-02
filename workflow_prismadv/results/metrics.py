"""Macro and binary precision, recall and F1, computed with scikit-learn.

A unit is safe when the task runs correctly on the batch; the method predicts it safe when it does not alarm.
scikit-learn's precision_recall_fscore_support gives precision, recall and F1 for both classes (labels=[safe,
unsafe], so both always count, also in a dataset without units of one class); macro = their mean
(average="macro"). Binary = the safe class only (average="binary", pos_label=True), the metric of the original
submission. zero_division=0: a method that never passes, or never alarms, scores 0 there instead of NaN.

The aggregation is ours: per dataset first, then the mean over the datasets per replicate, then the mean and
sample SD over the replicates. ci_half_width turns the unrounded SD into a 95% t confidence interval.
"""
from __future__ import annotations

import math

import pandas as pd
from scipy.stats import t
from sklearn.metrics import precision_recall_fscore_support

METRICS = ("precision", "recall", "f1")
KINDS = ("macro", "binary")


def macro_scores(safe, predicted_safe) -> dict[str, float]:
    """Macro precision, recall and F1 of paired ground truth and predictions (True = safe), plus F1 with safe as
    the positive class."""
    p, r, f, _ = precision_recall_fscore_support(list(safe), list(predicted_safe), labels=[True, False],
                                                 average=None, zero_division=0)
    return {"precision": p.mean(), "recall": r.mean(), "f1": f.mean(), "f1_safe": f[0]}


def binary_scores(safe, predicted_safe) -> dict[str, float]:
    """Precision, recall and F1 with safe as the positive class, of paired ground truth and predictions."""
    p, r, f, _ = precision_recall_fscore_support(list(safe), list(predicted_safe), average="binary", pos_label=True,
                                                 zero_division=0)
    return {"precision": p, "recall": r, "f1": f}


def macro(rows: pd.DataFrame) -> dict[str, float]:
    """Macro precision, recall and F1 of one set of outcome rows, plus F1 with safe as the positive class."""
    return macro_scores(rows["safe"].astype(bool), ~rows["alarm"].astype(bool))


def binary(rows: pd.DataFrame) -> dict[str, float]:
    """Binary precision, recall and F1 (safe = positive class) of one set of outcome rows."""
    return binary_scores(rows["safe"].astype(bool), ~rows["alarm"].astype(bool))


SCORERS = {"macro": macro, "binary": binary}


def per_dataset(outcomes: pd.DataFrame, kind: str = "macro") -> pd.DataFrame:
    """Metrics of `kind` (macro | binary) per (benchmark, arm, replicate, dataset)."""
    return (outcomes.groupby(["benchmark", "arm", "rep", "dataset"])
            .apply(lambda g: pd.Series(SCORERS[kind](g))))


def per_replicate(outcomes: pd.DataFrame, kind: str = "macro") -> pd.DataFrame:
    """Metrics per (benchmark, arm, replicate): per dataset first, then the mean over the datasets."""
    return per_dataset(outcomes, kind).groupby(["benchmark", "arm", "rep"]).mean()


def dataset_summary(outcomes: pd.DataFrame, kind: str = "macro") -> pd.DataFrame:
    """Per (benchmark, arm, dataset): mean and sample SD of the per-dataset metrics over replicates, and the
    replicate count."""
    g = per_dataset(outcomes, kind).groupby(["benchmark", "arm", "dataset"])
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    out["reps"] = g.size()
    return out


def summary(outcomes: pd.DataFrame, kind: str = "macro") -> pd.DataFrame:
    """Mean and sample SD over replicates (SD is NaN for a single replicate) and the replicate count."""
    rep = per_replicate(outcomes, kind)
    g = rep.groupby(["benchmark", "arm"])
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    out["reps"] = g.size()
    return out


def ci_half_width(sd: float, n: int, level: float = 0.95) -> float:
    """Half-width of the t confidence interval of a mean of n values with sample SD sd: t(1 - a/2, n - 1) sd / sqrt(n)."""
    return t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n)
