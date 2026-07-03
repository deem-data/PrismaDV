"""Evaluate TensorFlow Data Validation on ICDBench"""
import os
import tempfile
import warnings

import pandas as pd
import tensorflow_data_validation as tfdv

from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning)

from workflow_prismadv.icd_bench_experiments import instantiate_evaluation_case

exact_matches = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

num_constraints = 0

constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench")
for case_dir in os.listdir(constraints_output_dir):
    evaluation_case = instantiate_evaluation_case(case_dir)

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
assert (false_positives + true_negatives) == (true_positives + false_negatives)
num_cases = true_positives + false_negatives
print(f"\n\nFinal evaluation results for {num_cases} cases:\n")

print(f"{exact_matches}/{num_cases} exact matches of the ground truth constraint.\n")

print(f"{true_positives}/{num_cases} correct data batches passed.")
print(f"{false_negatives}/{num_cases} false alarms for correct data batches.")

print(f"{true_negatives}/{num_cases} problematic data batches identified.")
print(f"{false_positives}/{num_cases} problematic data batches missed.\n")

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

print("F1 Score:", f1)
