"""Evaluate AutoTest (Chen et al., SIGMOD 2025) on ICDBench."""
import os
import warnings

import pandas as pd

from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning)

from workflow_prismadv.icd_bench_experiments import instantiate_evaluation_case
from workflow_prismadv.eid_bench_baselines.pipelines.run_autotest import autotest_detect_tables

SDC_NAME = "rt_train"


def should_be_rejected(detected_df):
    """AutoTest rejects a batch if it flags an outlier in any column."""
    return len(detected_df) > 0


constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench")
case_dirs = sorted(os.listdir(constraints_output_dir))

cases = {}
id_to_df = {}
for case_dir in case_dirs:
    evaluation_case = instantiate_evaluation_case(case_dir)
    cases[case_dir] = evaluation_case
    id_to_df[f"{case_dir}__reject"] = pd.DataFrame(evaluation_case.data_to_reject())
    id_to_df[f"{case_dir}__pass"] = pd.DataFrame(evaluation_case.data_to_pass())

print(f"Running AutoTest ({SDC_NAME} SDCs) on {len(id_to_df)} batches "
      f"across {len(cases)} ICDBench cases...")
detections = autotest_detect_tables(id_to_df, sdc_name=SDC_NAME, run_name="icd_eval")

exact_matches = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

for case_dir in case_dirs:
    evaluation_case = cases[case_dir]

    rejected = should_be_rejected(detections[f"{case_dir}__reject"])
    passed = not should_be_rejected(detections[f"{case_dir}__pass"])

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
