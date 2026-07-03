import argparse
import time
import warnings
from collections import defaultdict

import oyaml as yaml

from prismadv.data_models import ValidationResults, Constraints
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import FileLoader
from prismadv.post_processing.individual_constraint import determine_validity
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyspark")

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", choices=dataset_name_options, help="Run only this dataset")
args = parser.parse_args()

selected_datasets = [args.dataset_name] if args.dataset_name else dataset_name_options

for dataset_name in selected_datasets:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    dq_manager = DeequDataQualityManager()
    subtask_names = project_manager.get_available_subtasks()
    for subtask_name in subtask_names:
        processed_data_labels = project_manager.get_available_processed_data_labels_for_subtask(subtask_name)
        script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name)
        script_names = [script_path.stem for script_path in script_path_list]
        for processed_data_label in processed_data_labels:
            if int(processed_data_label) == 0:
                continue
            for script_name in script_names:
                constraints_path = project_manager.get_constraints_path(
                    subtask_name, processed_data_label, script_name
                )
                validation_results_dir = project_manager.get_constraints_validation_path(
                    subtask_name, processed_data_label, script_name
                )
                suggestion_file_list = list(constraints_path.glob("few_shot*.yaml"))
                if int(processed_data_label) == 0:
                    clean = True
                else:
                    clean = False
                test_data_path = project_manager.get_new_test_data_path(
                    subtask_name, processed_data_label, clean=clean
                )
                test_data = FileLoader.load_csv(test_data_path)
                for suggestion_file_path in suggestion_file_list:
                    print(
                        f"Validating constraints for dataset: {dataset_name}, "
                        f"subtask: {subtask_name}, "
                        f"processed_data_label: {processed_data_label}, "
                        f"script: {script_name}, "
                        f"clean: {clean} ,"
                        f"Constraints path: {suggestion_file_path}"
                    )
                    if clean:
                        validation_results_path = validation_results_dir / f"validation_results_on_clean_test_data__{suggestion_file_path.stem}.yaml"
                    else:
                        validation_results_path = validation_results_dir / f"validation_results_on_corrupted_test_data__{suggestion_file_path.stem}.yaml"
                    validation_results_path.parent.mkdir(parents=True, exist_ok=True)
                    if validation_results_path.exists():
                        print(f"Validation results file {validation_results_path} already exists. Skipping...")
                        continue
                    try:
                        with open(f"{suggestion_file_path}", "r") as f:
                            raw_constraint_dict = yaml.load(f, Loader=yaml.FullLoader)
                        constraints = Constraints.from_dict(raw_constraint_dict)
                    except Exception as e:
                        print(f"Error loading constraints from {suggestion_file_path}: {e}"
                              f"\nSkipping this file.")
                        continue
                    valid_code_column_map = constraints.get_suggestions_code_column_map(valid_only=False)
                    code_list_for_constraints = [item for item in valid_code_column_map.keys()]
                    spark_test_data, spark_test = dq_manager.spark_df_from_pandas_df(test_data)
                    time_start = time.time()
                    status_on_test_data = dq_manager.validate_on_spark_df(
                        spark_test, spark_test_data, code_list_for_constraints)
                    time_end = time.time()
                    print("*" * 3, f"takes {time_end - time_start}")
                    result_dict = defaultdict(lambda: {"code": []})
                    for code, code_info in valid_code_column_map.items():
                        status, reason_if_failed = determine_validity(
                            code,
                            spark_test,
                            spark_test_data)
                        result_dict[code_info['column']]['code'].append({
                            "suggestion": code,
                            "status": status,
                            "reason_if_failed": reason_if_failed,
                            "level": code_info['level']
                        })
                    validation_results = ValidationResults.from_dict(result_dict)
                    validation_results.save_to_yaml(validation_results_path)

                    spark_test.sparkContext._gateway.shutdown_callback_server()
                    spark_test.stop()
