import argparse

from prismadv.error_injection.managers.ml_inference import MLInferenceErrorInjectionManager
from prismadv.error_injection.managers.sql_query import GeneralErrorInjectionManager
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root


def error_injection(dataset_name, subtask_name, error_config_file_name):
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    raw_file_path = project_manager.files_root
    target_table_name = project_manager.raw_data_info["target_table_name"]
    processed_data_dir = project_manager.get_base_processed_data_path_for_subtask(subtask_name)

    if subtask_name in ["classification", "regression"]:
        target_table_name = target_table_name
        target_column_name = project_manager.get_subtask_info(subtask_name)['target_column_name']
        submission_default_value = project_manager.get_subtask_info(subtask_name)['submission_default_value']
        error_injection_manager = MLInferenceErrorInjectionManager(
            raw_file_path=raw_file_path,
            target_table_name=target_table_name,
            target_column_name=target_column_name,
            processed_data_dir=processed_data_dir,
            submission_default_value=submission_default_value
        )
    elif subtask_name in ["bi", "dev", "feature_engineering", "info", "general_task"]:
        error_injection_manager = GeneralErrorInjectionManager(
            raw_file_path=raw_file_path,
            target_table_name=target_table_name,
            processed_data_dir=processed_data_dir,
            sample_size=1.0
        )
    else:
        raise ValueError(f"Downstream task {subtask_name} is not supported.")

    corrupts = error_injection_manager.load_error_injection_config(
        error_injection_config_path=project_manager.errors_root / error_config_file_name)

    corrupts_existence = error_injection_manager.check_corrupts_existence(corrupts, processed_data_dir)
    if corrupts_existence:
        print(f"The error configuration in {error_config_file_name} has already been applied to {dataset_name} "
              f"for the downstream task {subtask_name}. Skipping error injection.")
    else:
        error_injection_manager.error_injection(corrupts)
        # Save the corrupted test data
        error_injection_manager.save_data()
        print(f"Error injection for {dataset_name} is done.\nThe corrupted data is saved in {processed_data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject errors into datasets")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name to process (e.g., 'imdb', 'students'). If not provided, processes all datasets."
    )
    args = parser.parse_args()

    all_datasets = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
    dataset_name_options = [args.dataset_name] if args.dataset_name else all_datasets

    for dataset_name in dataset_name_options:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        subtask_names = project_manager.get_available_subtasks()
        for subtask_name in subtask_names:
            error_config_selections = [error_config_file.name for error_config_file in
                                       project_manager.errors_root.iterdir()]
            error_config_selections.sort()
            for error_config_file_name in error_config_selections:
                if not error_config_file_name.startswith(subtask_name):
                    continue
                print(f"\tStarting error injection for dataset: {dataset_name}, downstream task: {subtask_name}, "
                      f"error config: {error_config_file_name}")
                error_injection(
                    dataset_name=dataset_name,
                    subtask_name=subtask_name,
                    error_config_file_name=error_config_file_name
                )
