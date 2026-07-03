import warnings

from prismadv.data_models import ValidationResults
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyspark")

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]

selected_datasets = dataset_name_options

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

                suggestion_file_list = list(constraints_path.glob("*.yaml"))

                # --- Orphan validation cleanup (integrated) ---
                # Remove validation result files whose corresponding constraint YAML no longer exists
                constraint_stems = {p.stem for p in suggestion_file_list}
                if validation_results_dir.exists():
                    for vr in validation_results_dir.glob("validation_results_on_*__*.yaml"):
                        name = vr.stem  # e.g., validation_results_on_clean_test_data__foo_bar
                        try:
                            constraint_part = name.split("__", 1)[1]
                        except IndexError:
                            print(f"Deleting malformed validation file {vr}")
                            vr.unlink()
                            continue
                        if constraint_part not in constraint_stems:
                            print(f"Deleting orphan validation file {vr} (constraint removed)")
                            vr.unlink()
                # --- End orphan cleanup ---

                if int(processed_data_label) == 0:
                    clean = True
                else:
                    clean = False

                test_data_path = project_manager.get_new_test_data_path(
                    subtask_name, processed_data_label, clean=clean
                )
                test_data = FileLoader.load_csv(test_data_path)

                for suggestion_file_path in suggestion_file_list:
                    if clean:
                        validation_results_path = validation_results_dir / (
                            f"validation_results_on_clean_test_data__{suggestion_file_path.stem}.yaml"
                        )
                    else:
                        validation_results_path = validation_results_dir / (
                            f"validation_results_on_corrupted_test_data__{suggestion_file_path.stem}.yaml"
                        )

                    validation_results_path.parent.mkdir(parents=True, exist_ok=True)

                    if validation_results_path.exists():
                        validation_results = ValidationResults.from_yaml(validation_results_path)
                        flag_out_of_memory = False
                        for column_name, column_result in validation_results.results.items():
                            for entry in column_result.code:
                                if "java.lang.OutOfMemoryError" in entry.reason_if_failed:
                                    flag_out_of_memory = True
                                    break
                        if flag_out_of_memory:
                            print(f"{validation_results_path}")
                            # unlink them
                            validation_results_path.unlink()
