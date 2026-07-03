from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_baselines.pipelines.run_autotest import run_autotest

if __name__ == "__main__":
    dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
    subtask_options = ["general_task"]
    processed_data_label = "0"

    for dataset_name in dataset_name_options:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        for subtask_name in subtask_options:
            try:
                task_description = project_manager.get_subtask_description(subtask_name=subtask_name)
            except ValueError as e:
                print(f"Skipping invalid subtask '{subtask_name}' for dataset '{dataset_name}': {e}")
                continue
            run_autotest(
                dataset_name,
                subtask_name,
                processed_data_label=processed_data_label,
            )
