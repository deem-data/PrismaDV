import os
import signal
import sys
import warnings
from pathlib import Path

import oyaml as yaml
import pandas as pd

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning)


def find_pocketflow_constraints_output_path(evaluation_case):
    target_dir = evaluation_case.__class__.__module__.split('.')[-1] + "." + evaluation_case.__class__.__name__
    constraints_output_dir = (
                get_project_root() / "data_processed" / "icd_bench" / target_dir / "constraints")
    Path(constraints_output_dir).mkdir(parents=True, exist_ok=True)
    constraints_output_path = (constraints_output_dir / f"pocketflow_constraints.yaml")
    return constraints_output_path


def instantiate_evaluation_case(case_dir):
    import importlib
    case_package, case_class = case_dir.split('.')
    module = importlib.import_module(f"benchmarks.ICDBench.{case_package}")
    cls = getattr(module, case_class)
    evaluation_case = cls()
    return evaluation_case


# ANSI escape code for red text
RED = "\033[31m"
RESET = "\033[0m"

exact_matches = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench")
for case_dir in os.listdir(constraints_output_dir):
    evaluation_case = instantiate_evaluation_case(case_dir)
    constraints_output_path = find_pocketflow_constraints_output_path(evaluation_case)
    with open(constraints_output_path, "r") as f:
        res_dict = yaml.safe_load(f)

    try:
        generated_constraints = res_dict['constraints']
    except:
        generated_constraints = []

    # HACK: FILTER OUT DATATYPE constraints, since they are not generated correctly at the moment
    # generated_constraints = [c for c in generated_constraints if not c.startswith(".hasDataType")]

    generated_exact_ground_truth = evaluation_case.ground_truth_constraint() in generated_constraints
    if generated_exact_ground_truth:
        exact_matches += 1

    dq_manager = DeequDataQualityManager()

    # Required for the GithubFashionTrendsCategory case, since Spark cannot handle columns with nulls only
    data_to_reject = pd.DataFrame(evaluation_case.data_to_reject())
    if evaluation_case.__class__.__name__ == 'GithubFashionTrendsCategory':
        from pyspark.sql.types import StructType, StructField, StringType, FloatType

        schema = StructType([
            StructField("comments", StringType(), True),
            StructField("category", FloatType(), True),
        ])
        data_to_reject_df, spark = dq_manager.spark_df_from_pandas_df(data_to_reject, schema=schema)
    else:
        data_to_reject_df, spark = dq_manager.spark_df_from_pandas_df(data_to_reject)

    reject_constraint_results = dq_manager.validate_on_spark_df(spark, data_to_reject_df, generated_constraints)
    rejected = 'Failure' in reject_constraint_results

    data_to_pass = pd.DataFrame(evaluation_case.data_to_pass())
    data_to_pass_df, spark = dq_manager.spark_df_from_pandas_df(data_to_pass)
    pass_constraint_results = dq_manager.validate_on_spark_df(spark, data_to_pass_df, generated_constraints)
    passed = 'Failure' not in pass_constraint_results

    print(f"\nEvaluating {evaluation_case.__class__.__module__}.{evaluation_case.__class__.__name__}...")

    print(f"\tGround truth constraint for '{evaluation_case.target_column()}':")
    print(f"\t\t{evaluation_case.ground_truth_constraint()}")

    print(f"\tGenerated constraints for '{evaluation_case.target_column()}':")
    for constraint in generated_constraints:
        print(f"\t\t{constraint}")

    print(f"\tExact ground truth constraint? {generated_exact_ground_truth}")

    print(f"\tRejects data to reject? {rejected}")
    if not rejected:
        false_positives += 1
        for constraint in generated_constraints:
            print(f"\t\t{RED}Constraint passed: {constraint}{RESET}", file=sys.stderr)
    else:
        true_negatives += 1

    print(f"\tPasses data to pass? {passed}")
    if not passed:
        false_negatives += 1
        for constraint, result in zip(generated_constraints, pass_constraint_results):
            if result == 'Failure':
                print(f"\t\t{RED}Constraint failed: {constraint}{RESET}", file=sys.stderr)
    else:
        true_positives += 1

    # Release the Spark session each case; otherwise memory accumulates across
    # the 63 cases and the run gets OOM-killed before printing the F1.
    try:
        spark.sparkContext._gateway.shutdown_callback_server()
    except Exception:
        pass
    try:
        spark.stop()
    except Exception:
        pass

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

# Stop Spark session
spark.stop()
# For some reason this is necessary to make the program terminate...
os.kill(os.getpid(), signal.SIGTERM)
