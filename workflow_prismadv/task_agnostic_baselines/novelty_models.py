"""Model-based novelty detectors used as task-agnostic EIDBench baselines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import OneClassSVM

TRAIN_SAMPLE_CAP = 5000
PREDICT_SAMPLE_CAP = 20000
MAX_CAT_CARDINALITY = 50
RANDOM_STATE = 0


@dataclass
class NoveltyDetector:
    """A fitted novelty model plus the feature columns it was trained on."""

    pipeline: Pipeline
    numeric_cols: list[str]
    categorical_cols: list[str]

    @property
    def feature_cols(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols


def select_feature_columns(df: pd.DataFrame, *, max_cat_cardinality: int = MAX_CAT_CARDINALITY):
    """Pick numeric columns and low-cardinality categorical columns for encoding."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col
        for col in df.select_dtypes(include=["object", "category"]).columns
        if 0 < df[col].nunique(dropna=True) <= max_cat_cardinality
    ]
    return numeric_cols, categorical_cols


def _subsample(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap and len(df) > cap:
        return df.sample(cap, random_state=RANDOM_STATE)
    return df


def _to_finite_numeric(series: pd.Series) -> pd.Series:
    """Coerce to float; non-numeric strings and +/-inf become NaN (later imputed)."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _fit_detector(
    df: pd.DataFrame,
    estimator,
    estimator_name: str,
    *,
    train_sample_cap: int,
    max_cat_cardinality: int,
) -> NoveltyDetector | None:
    """Shared pipeline: numeric mean-impute + categorical one-hot, then ``estimator``."""
    numeric_cols, categorical_cols = select_feature_columns(
        df, max_cat_cardinality=max_cat_cardinality
    )
    if not numeric_cols and not categorical_cols:
        return None

    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="mean"), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers)),
        (estimator_name, estimator),
    ])

    feature_cols = numeric_cols + categorical_cols
    train = _subsample(df[feature_cols], train_sample_cap).copy()
    for col in numeric_cols:
        train[col] = _to_finite_numeric(train[col])
    pipeline.fit(train)
    return NoveltyDetector(
        pipeline=pipeline, numeric_cols=numeric_cols, categorical_cols=categorical_cols
    )


def learn_one_class_svm(
    df: pd.DataFrame,
    *,
    train_sample_cap: int = TRAIN_SAMPLE_CAP,
    max_cat_cardinality: int = MAX_CAT_CARDINALITY,
    nu: float = 0.01,
) -> NoveltyDetector | None:
    """Fit a One-Class SVM on the clean ``df``; ``None`` if no usable features."""
    return _fit_detector(
        df,
        OneClassSVM(kernel="rbf", gamma="scale", nu=nu),
        "ocsvm",
        train_sample_cap=train_sample_cap,
        max_cat_cardinality=max_cat_cardinality,
    )


def learn_isolation_forest(
    df: pd.DataFrame,
    *,
    train_sample_cap: int = TRAIN_SAMPLE_CAP,
    max_cat_cardinality: int = MAX_CAT_CARDINALITY,
    n_estimators: int = 10,
    contamination="auto",
) -> NoveltyDetector | None:
    """Fit an Isolation Forest on the clean ``df``; ``None`` if no usable features."""
    return _fit_detector(
        df,
        IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=RANDOM_STATE
        ),
        "isof",
        train_sample_cap=train_sample_cap,
        max_cat_cardinality=max_cat_cardinality,
    )


def _align_features(detector: NoveltyDetector, df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``df`` to the detector's trained feature columns and dtypes."""
    features = df.reindex(columns=detector.feature_cols)
    for col in detector.numeric_cols:
        features[col] = _to_finite_numeric(features[col])
    for col in detector.categorical_cols:
        features[col] = features[col].astype(object)
    return features


def should_be_rejected(
    detector: NoveltyDetector | None,
    df: pd.DataFrame,
    *,
    predict_sample_cap: int = PREDICT_SAMPLE_CAP,
) -> bool:
    """Return True if the detector flags any row of ``df`` as anomalous."""
    if detector is None:
        return False
    sample = _subsample(df, predict_sample_cap)
    features = _align_features(detector, sample)
    predictions = detector.pipeline.predict(features)
    return bool((predictions == -1).any())


STATS_NOVELTY_ALPHA = 0.05
STATS_NOVELTY_N_PERMUTATIONS = 1000


def _index_of_peculiarity(series: pd.Series) -> float:
    counts = series.value_counts(dropna=True)
    if len(counts) == 0:
        return np.nan
    return 1 - (counts.max() / counts.sum())


def compute_column_stats(df: pd.DataFrame) -> np.ndarray:
    """One 8-dim summary-stat row per column."""
    stats_list = []
    for col in df.columns:
        series = df[col]
        col_stats = [
            series.notnull().mean(),
            series.nunique(dropna=True),
        ]
        if series.notnull().any():
            col_stats.append(series.value_counts(dropna=True).max() / len(series.dropna()))
        else:
            col_stats.append(np.nan)
        if np.issubdtype(series.dtype, np.number):
            col_stats.extend([series.max(), series.mean(), series.min(), series.std(), np.nan])
        else:
            col_stats.extend([np.nan, np.nan, np.nan, np.nan, _index_of_peculiarity(series)])
        stats_list.append(col_stats)
    return np.array(stats_list, dtype=float)


def _gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    x_norm = np.sum(x ** 2, axis=1).reshape(-1, 1)
    y_norm = np.sum(y ** 2, axis=1).reshape(1, -1)
    dist = x_norm + y_norm - 2 * np.dot(x, y.T)
    return np.exp(-dist / (2 * sigma ** 2))


def _median_heuristic_sigma(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    dists = [
        np.linalg.norm(Z[i] - Z[j])
        for i in range(len(Z))
        for j in range(i + 1, len(Z))
    ]
    return np.median(dists)


def mmd(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_permutations: int = STATS_NOVELTY_N_PERMUTATIONS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """MMD statistic and permutation-test p-value (seeded for reproducibility)."""
    n = len(X)
    sigma = _median_heuristic_sigma(X, Y)
    mmd_stat = (
        _gaussian_kernel(X, X, sigma).mean()
        + _gaussian_kernel(Y, Y, sigma).mean()
        - 2 * _gaussian_kernel(X, Y, sigma).mean()
    )
    Z = np.vstack([X, Y])
    rng = np.random.RandomState(random_state)
    mmd_perms = []
    for _ in range(n_permutations):
        idx = rng.permutation(len(Z))
        Xp, Yp = Z[idx[:n]], Z[idx[n:]]
        mmd_perms.append(
            _gaussian_kernel(Xp, Xp, sigma).mean()
            + _gaussian_kernel(Yp, Yp, sigma).mean()
            - 2 * _gaussian_kernel(Xp, Yp, sigma).mean()
        )
    p_value = np.mean(np.array(mmd_perms) > mmd_stat)
    return mmd_stat, p_value


def should_be_rejected_stats_novelty(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    *,
    alpha: float = STATS_NOVELTY_ALPHA,
    n_permutations: int = STATS_NOVELTY_N_PERMUTATIONS,
) -> bool:
    """Reject ``test`` if its column-stats distribution differs from ``reference``."""
    x = compute_column_stats(reference)
    y = compute_column_stats(test)
    _, p_value = mmd(x, y, n_permutations=n_permutations)
    return bool(p_value < alpha)
