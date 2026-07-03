"""Evaluate Isolation Forest on ICDBench"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning)

from workflow_prismadv.icd_bench_experiments import instantiate_evaluation_case


def learn_novelty_detector(df):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    numeric_transformer = SimpleImputer(strategy="mean")

    # Categorical: fill NaNs with "missing" and then one-hot encode
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    isof = IsolationForest(contamination='auto', n_estimators=10)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("isof", isof)
    ])

    model = pipeline.fit(df)
    return model


def should_be_rejected(model, df):
    predictions = model.predict(df)
    return (predictions == -1).any()


exact_matches = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench")
for case_dir in os.listdir(constraints_output_dir):
    evaluation_case = instantiate_evaluation_case(case_dir)

    data_sample = pd.DataFrame(evaluation_case.sample_data())

    model = learn_novelty_detector(data_sample)

    data_to_reject = pd.DataFrame(evaluation_case.data_to_reject())
    rejected = should_be_rejected(model, data_to_reject)

    data_to_pass = pd.DataFrame(evaluation_case.data_to_pass())
    passed = not should_be_rejected(model, data_to_pass)

    print(f"\nEvaluating {evaluation_case.__class__.__module__}.{evaluation_case.__class__.__name__}...")

    print(f"\tGround truth constraint for '{evaluation_case.target_column()}':")
    print(f"\t\t{evaluation_case.ground_truth_constraint()}")

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
