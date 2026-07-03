from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_baselines.pipelines.run_few_shot import run_few_shot

dataset_name_options = ["imdb", "hr_analytics", "sleep_health", "students", "IPL_win_prediction"]
# dataset_name_options = ["sleep_health"]
subtask_options = ["general_task"]
model_name_options = ["gpt-4o", "gpt-4.1", "gpt-5-mini", 'gpt-5', 'gemini-2.5-flash', 'gemini-2.5-pro']
processed_data_label = "0"

for dataset_name in dataset_name_options:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for subtask_name in subtask_options:
        try:
            task_description = project_manager.get_subtask_description(subtask_name=subtask_name)
        except ValueError as e:
            print(f"Skipping invalid subtask '{subtask_name}' for dataset '{dataset_name}': {e}")
            continue
        for model_name in model_name_options:
            print(f"running for dataset: {dataset_name}, subtask: {subtask_name}, "
                  f"processed_data_label: {processed_data_label}, model_name: {model_name}")
            run_few_shot(
                dataset_name=dataset_name,
                subtask_name=subtask_name,
                processed_data_label=processed_data_label,
                model_name=model_name
            )
