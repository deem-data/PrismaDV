import argparse
import time
import warnings
from collections import defaultdict

import oyaml as yaml

from prismadv.data_models import ConstraintsWithSources, ValidationResults
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import FileLoader
from prismadv.project_manager import is_clean_label
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyspark")

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
skip_if_exist = True

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", choices=dataset_name_options, help="Run only this dataset")
parser.add_argument("--script_name", action="append", default=None,
                    help="Only these task scripts (repeatable). Default: every script.")
parser.add_argument("--label", action="append", default=None,
                    help="Only these processed-data labels (repeatable). Default: every label.")
parser.add_argument("--constraint_glob", action="append", default=None,
                    help="Only constraint artifacts that match (repeatable), e.g. 'prismadv--gemini-3.7-flash*.yaml'. "
                         "Default: '*prismadv*.yaml'.")
args = parser.parse_args()
constraint_globs = args.constraint_glob or ["*prismadv*.yaml"]

selected_datasets = [args.dataset_name] if args.dataset_name else dataset_name_options

for dataset_name in selected_datasets:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    dq_manager = DeequDataQualityManager()
    subtask_names = project_manager.get_available_subtasks()
    for subtask_name in subtask_names:
        processed_data_labels = project_manager.get_available_processed_data_labels_for_subtask(subtask_name)
        script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name)
        script_names = [script_path.stem for script_path in script_path_list]
        if args.script_name:
            script_names = [name for name in script_names if name in set(args.script_name)]
        for processed_data_label in processed_data_labels:
            if is_clean_label(processed_data_label):
                continue
            if args.label and str(processed_data_label) not in set(args.label):
                continue
            pending = []                            # (script_name, validation_jobs) still to validate on this label
            for script_name in script_names:
                constraints_path = project_manager.get_source_constraints_path(subtask_name, script_name)
                validation_results_dir = project_manager.get_constraints_validation_path(
                    subtask_name, processed_data_label, script_name
                )
                suggestion_file_list = sorted({p for pattern in constraint_globs for p in constraints_path.glob(pattern)})
                # Label 0 is skipped above, so only the error-injected side is
                # ever scored.
                clean = False

                validation_jobs = []
                all_exist = True
                for suggestion_file_path in suggestion_file_list:
                    validation_results_path = (
                        validation_results_dir
                        / f"validation_results_on_corrupted_test_data__{suggestion_file_path.stem}.yaml"
                    )
                    exists = validation_results_path.exists()
                    if not exists:
                        all_exist = False
                    validation_jobs.append((suggestion_file_path, validation_results_path, exists))

                if not validation_jobs:
                    continue

                if skip_if_exist and all_exist:
                    for suggestion_file_path, validation_results_path, _ in validation_jobs:
                        print(
                            f"\n{'='*80}"
                            f"\nDataset: {dataset_name}, Subtask: {subtask_name}, "
                            f"Label: {processed_data_label}, Script: {script_name}"
                            f"\nConstraint file: {suggestion_file_path.name}"
                            f"\nClean data: {clean}"
                            f"\n{'='*80}"
                        )
                        print(f"Validation results file {validation_results_path} already exists. Skipping...")
                    continue
                pending.append((script_name, validation_jobs))

            if not pending:
                continue
            # The corrupted batch depends on the label only: load it and start Spark once for every task and
            # constraint file of the label.
            test_data_path = project_manager.get_new_test_data_path(
                subtask_name, processed_data_label, clean=False
            )
            test_data = FileLoader.load_csv(test_data_path)
            spark_test_data, spark_test = dq_manager.spark_df_from_pandas_df(test_data)

            try:
                for script_name, validation_jobs in pending:
                    for suggestion_file_path, validation_results_path, exists in validation_jobs:
                        print(
                            f"\n{'='*80}"
                            f"\nDataset: {dataset_name}, Subtask: {subtask_name}, "
                            f"Label: {processed_data_label}, Script: {script_name}"
                            f"\nConstraint file: {suggestion_file_path.name}"
                            f"\nClean data: {clean}"
                            f"\n{'='*80}"
                        )
                        if exists and skip_if_exist == True:
                            print(f"Validation results file {validation_results_path} already exists. Skipping...")
                            continue
                        validation_results_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            with open(f"{suggestion_file_path}", "r") as f:
                                raw_constraint_dict = yaml.load(f, Loader=yaml.FullLoader)
                            constraints = ConstraintsWithSources.from_dict(raw_constraint_dict)
                        except Exception as e:
                            print(f"Error loading constraints from {suggestion_file_path}: {e}"
                                  f"\nSkipping this file.")
                            continue
                        valid_code_column_map = constraints.get_suggestions_code_column_map(valid_only=True)
                        code_list_for_constraints = list(valid_code_column_map.keys())
                        
                        if not code_list_for_constraints:
                            print("No valid constraints to validate. Skipping...")
                            continue
                        
                        print(f"Validating {len(code_list_for_constraints)} constraints...")
                        
                        # Use batched validation approach
                        time_start = time.perf_counter()
                        validation_results_list = dq_manager.validate_constraints_with_reasons(
                            spark_test, spark_test_data, code_list_for_constraints, isolated_check=True
                        )
                        time_end = time.perf_counter()
                        elapsed_time = time_end - time_start
                        
                        print(f"Time: {elapsed_time:.4f}s total ({elapsed_time / len(code_list_for_constraints):.4f}s per constraint)")
                        
                        # Build validation results
                        result_dict = defaultdict(lambda: {"code": []})
                        for code, (status, reason_if_failed) in zip(code_list_for_constraints, validation_results_list):
                            code_info = valid_code_column_map[code]
                            result_dict[code_info['column']]['code'].append({
                                "suggestion": code,
                                "status": status,
                                "reason_if_failed": reason_if_failed,
                                "level": code_info['level']
                            })
                        validation_results = ValidationResults.from_dict(result_dict)
                        validation_results.validation_summary = {
                            "elapsed_s": round(elapsed_time, 3),
                            "num_constraints": len(code_list_for_constraints),
                        }
                        validation_results.save_to_yaml(validation_results_path)
                        print(f"Results saved to: {validation_results_path}")

            finally:
                # Cleanup: Shutdown Spark session once after all tasks of the label are validated
                spark_test.sparkContext._gateway.shutdown_callback_server()
                spark_test.stop()
