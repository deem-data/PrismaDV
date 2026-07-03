"""Evaluate stats novelty approach (partition summarization with MMD test) on ICDBench"""
import os
import warnings

import pandas as pd

from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning)

from workflow_prismadv.icd_bench_experiments import instantiate_evaluation_case


def index_of_peculiarity(series):
    """Compute Index of Peculiarity for a textual series"""
    counts = series.value_counts(dropna=True)
    if len(counts) == 0:
        return np.nan
    return 1 - (counts.max() / counts.sum())


def compute_column_stats(df):
    stats_list = []

    for col in df.columns:
        series = df[col]
        col_stats = []

        # Completeness
        completeness = series.notnull().mean()
        col_stats.append(completeness)

        # Number of distinctive values
        n_unique = series.nunique(dropna=True)
        col_stats.append(n_unique)

        # Ratio of the most frequent value
        if series.notnull().any():
            most_freq_ratio = series.value_counts(dropna=True).max() / len(series.dropna())
        else:
            most_freq_ratio = np.nan
        col_stats.append(most_freq_ratio)

        if np.issubdtype(series.dtype, np.number):
            # Numeric metrics
            col_stats.append(series.max())
            col_stats.append(series.mean())
            col_stats.append(series.min())
            col_stats.append(series.std())
            # Placeholder for textual peculiarity
            col_stats.append(np.nan)
        else:
            # Placeholder for numeric stats
            col_stats.extend([np.nan, np.nan, np.nan, np.nan])
            # Index of peculiarity for text
            col_stats.append(index_of_peculiarity(series))

        stats_list.append(col_stats)

    # Convert to NumPy array
    return np.array(stats_list, dtype=float)


import numpy as np


def gaussian_kernel(x, y, sigma=1.0):
    """Compute Gaussian (RBF) kernel between two sets of samples."""
    x_norm = np.sum(x ** 2, axis=1).reshape(-1, 1)
    y_norm = np.sum(y ** 2, axis=1).reshape(1, -1)
    dist = x_norm + y_norm - 2 * np.dot(x, y.T)
    return np.exp(-dist / (2 * sigma ** 2))


def median_heuristic_sigma(X, Y):
    """Choose kernel bandwidth sigma using the median heuristic."""
    Z = np.vstack([X, Y])
    dists = []
    for i in range(len(Z)):
        for j in range(i + 1, len(Z)):
            dists.append(np.linalg.norm(Z[i] - Z[j]))
    return np.median(dists)


def mmd(X, Y, n_permutations=1000):
    """Compute MMD statistic and p-value using a permutation test."""
    n, m = len(X), len(Y)

    sigma = median_heuristic_sigma(X, Y)

    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)

    mmd_stat = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

    # Permutation test
    Z = np.vstack([X, Y])
    n_total = len(Z)
    mmd_perms = []
    for _ in range(n_permutations):
        idx = np.random.permutation(n_total)
        Xp = Z[idx[:n]]
        Yp = Z[idx[n:]]
        Kxxp = gaussian_kernel(Xp, Xp, sigma).mean()
        Kyyp = gaussian_kernel(Yp, Yp, sigma).mean()
        Kxyp = gaussian_kernel(Xp, Yp, sigma).mean()
        mmd_perms.append(Kxxp + Kyyp - 2 * Kxyp)

    p_value = np.mean(np.array(mmd_perms) > mmd_stat)
    return mmd_stat, p_value


def should_be_rejected(data_sample, data_to_validate):
    x = compute_column_stats(data_sample)
    y = compute_column_stats(data_to_validate)
    mmd_value, p_val = mmd(x, y, n_permutations=1000)
    return p_val < 0.05  # Reject if p-value is less than 0.05


exact_matches = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench")
for case_dir in os.listdir(constraints_output_dir):
    evaluation_case = instantiate_evaluation_case(case_dir)

    data_sample = pd.DataFrame(evaluation_case.sample_data())

    data_to_reject = pd.DataFrame(evaluation_case.data_to_reject())
    rejected = should_be_rejected(data_sample, data_to_reject)

    data_to_pass = pd.DataFrame(evaluation_case.data_to_pass())
    passed = not should_be_rejected(data_sample, data_to_pass)

    print(f"\nEvaluating {evaluation_case.__class__.__module__}.{evaluation_case.__class__.__name__}...")

    print(f"\tRejects data to reject? {rejected}")
    if not rejected:
        false_positives += 1
    else:
        true_negatives += 1

    print(f"\tPasses data to pass? {passed}")
    if not passed:
        false_negatives += 1
    else:
        true_positives += 1

num_cases = true_positives + false_negatives
print(f"\n\nFinal evaluation results for {num_cases} cases:\n")

print(f"{exact_matches}/{num_cases} exact matches of the ground truth constraint.\n")

print(f"{true_positives}/{num_cases} correct data batches passed.")
print(f"{false_negatives}/{num_cases} false alarms for correct data batches.")

print(f"{true_negatives}/{num_cases} problematic data batches identified.")
print(f"{false_positives}/{num_cases} problematic data batches missed.\n")

if true_positives != 0:
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
else:
    f1 = 0.0

print("F1 Score:", f1)
