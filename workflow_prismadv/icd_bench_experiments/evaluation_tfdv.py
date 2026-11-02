"""Evaluate TensorFlow Data Validation on ICDBench"""
import os
import tempfile
import time
import warnings

import pandas as pd
import tensorflow_data_validation as tfdv

from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning)

from workflow_prismadv.icd_bench_experiments import instantiate_evaluation_case
from workflow_prismadv.icd_bench_experiments.scoring import case_dirs, out_argument, write_outcomes

args = out_argument(__doc__, "tfdv--tfdv--r1.tsv")
rows = []
exact_matches = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

num_constraints = 0

for case_dir in case_dirs():
    evaluation_case = instantiate_evaluation_case(case_dir)
    started = time.perf_counter()

    data_sample = pd.DataFrame(evaluation_case.sample_data())
    assert type(evaluation_case.target_column()) == str
    selected_features = [evaluation_case.target_column()]
    data_sample_subset = data_sample[selected_features]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_train_csv_location = tmp_file.name
        data_sample_subset.to_csv(tmp_train_csv_location, index=False)
    train_stats = tfdv.generate_statistics_from_csv(
        data_location=(tmp_train_csv_location)
    )

    schema = tfdv.infer_schema(train_stats)
    # Required for the GithubFashionTrendsCategory case, since Spark cannot handle columns with nulls only
    data_to_reject = pd.DataFrame(evaluation_case.data_to_reject())
    data_to_reject_subset = data_to_reject[selected_features]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_new_csv_to_reject_location = tmp_file.name
        data_to_reject_subset.to_csv(tmp_new_csv_to_reject_location, index=False)
    eval_to_reject_stats = tfdv.generate_statistics_from_csv(
        data_location=(tmp_new_csv_to_reject_location)
    )
    anomalies_to_reject = tfdv.validate_statistics(eval_to_reject_stats, schema)
    rejected = len(anomalies_to_reject.anomaly_info) != 0

    data_to_pass = pd.DataFrame(evaluation_case.data_to_pass())
    data_to_pass_subset = data_to_pass[selected_features]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_new_csv_to_pass_location = tmp_file.name
        data_to_pass_subset.to_csv(tmp_new_csv_to_pass_location, index=False)
    eval_to_pass_stats = tfdv.generate_statistics_from_csv(
        data_location=(tmp_new_csv_to_pass_location)
    )
    anomalies_to_pass = tfdv.validate_statistics(eval_to_pass_stats, schema)
    passed = len(anomalies_to_pass.anomaly_info) == 0
    rows.append({"case": case_dir, "passes_pass_batch": passed, "rejects_reject_batch": rejected,
                 "wall_clock_seconds": round(time.perf_counter() - started, 3)})

    print(f"\nEvaluating {evaluation_case.__class__.__module__}.{evaluation_case.__class__.__name__}...")

    print(f"\tGround truth constraint for '{evaluation_case.target_column()}':")
    print(f"\t\t{evaluation_case.ground_truth_constraint()}")

    print(f"\tGenerated constraints for '{evaluation_case.target_column()}':")

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

    print(f"anomaly info to reject: {anomalies_to_reject.anomaly_info}")
    print(f"anomaly info to pass: {anomalies_to_pass.anomaly_info}")
write_outcomes(rows, args.out)
