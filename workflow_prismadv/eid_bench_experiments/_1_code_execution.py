import argparse

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow.e2e_evaluation.code_execution.run_general_python_code import run_general_python_code
from workflow.e2e_evaluation.code_execution.run_sql_code import run_general_sql_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run code execution on corrupted data")
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
        available_subtasks = project_manager.get_available_subtasks()
        for subtask in available_subtasks:
            processed_data_labels = project_manager.get_available_processed_data_labels_for_subtask(subtask)
            for processed_data_label in processed_data_labels:
                if subtask == "general_task" or subtask == "info" or subtask in ["classification", "regression"]:
                    run_general_python_code(
                        dataset_name=dataset_name,
                        subtask_name=subtask,
                        processed_data_label=f"{processed_data_label}",
                    )
                elif subtask == "bi" or subtask == "dev" or subtask == "feature_engineering":
                    run_general_sql_code(
                        dataset_name=dataset_name,
                        subtask_name=subtask,
                        processed_data_label=f"{processed_data_label}",
                    )
