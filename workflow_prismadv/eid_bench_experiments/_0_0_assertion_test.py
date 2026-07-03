from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow.e2e_evaluation.code_execution.run_general_python_code_for_assertion_selection import \
    run_general_python_code_for_assertion_selection

# test which assertions in the code are valid and which are not.
if __name__ == "__main__":
    dataset_selections = ["hr_analytics", "IPL_win_prediction", "sleep_health", "imdb", "students"]
    for dataset_name in dataset_selections:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        available_subtasks = project_manager.get_available_subtasks()
        for subtask in available_subtasks:
            processed_data_label = "0"
            if subtask == "general_task":
                run_general_python_code_for_assertion_selection(
                    dataset_name=dataset_name, subtask_name=subtask,
                    processed_data_label=f"{processed_data_label}"
                )
