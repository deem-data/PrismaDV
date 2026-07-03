import oyaml as yaml
import tensorflow_data_validation as tfdv

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]


def count_anomalies(anomalies):
    anomaly_count = 0
    for key, value in anomalies.anomaly_info.items():
        if value.severity == 2:
            anomaly_count += 1
    return anomaly_count


for dataset_name in dataset_name_options:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    subtask_names = project_manager.get_available_subtasks()
    for subtask_name in subtask_names:
        processed_data_labels = project_manager.get_available_processed_data_labels_for_subtask(subtask_name)
        for processed_data_label in processed_data_labels:
            if int(processed_data_label) == 0:
                continue
            suggestion_file_path = project_manager.get_task_agnostic_constraint_path(
                subtask_name, processed_data_label,
            ) / "schema.pbtxt"
            validation_results_dir = project_manager.get_task_agnostic_constraints_validation_path(
                subtask_name, processed_data_label,
            )
            if int(processed_data_label) == 0:
                clean = True
            else:
                clean = False
            test_data_path = project_manager.get_new_test_data_path(
                subtask_name, processed_data_label, clean=clean
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
                schema_on_observed_data = tfdv.load_schema_text(
                    suggestion_file_path
                )
            except Exception as e:
                print(f"Error loading constraints from {suggestion_file_path}: {e}"
                      f"\nSkipping this file.")
                continue
            eval_stats = tfdv.generate_statistics_from_csv(str(test_data_path))
            anomalies = tfdv.validate_statistics(eval_stats, schema_on_observed_data)

            result = {
                "pass": (len(anomalies.anomaly_info) == 0),
                "anomalies": {k: v.description for k, v in anomalies.anomaly_info.items()}
            }

            with open(validation_results_path, "w") as f:
                yaml.dump(result, f, sort_keys=False)
