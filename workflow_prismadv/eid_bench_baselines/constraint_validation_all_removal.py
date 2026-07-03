import warnings

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyspark")

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]

removing_prefix_list = ["few_shot", "single_shot", "mini_swe_agent"]

for dataset_name in dataset_name_options:
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
                suggestion_file_list = []
                for prefix in removing_prefix_list:
                    suggestion_file_list.extend(list(constraints_path.glob(f"{prefix}*.yaml")))
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
                        f"Removing constraints for dataset: {dataset_name}, "
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
                        print(f"Removing validation results file {validation_results_path}.")
                        validation_results_path.unlink()
