from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
model_names = ['gpt-5']

import oyaml as yaml


def cleanup(
        dataset_name,
        subtask_name,
        processed_data_label,
        post_processed_file):
    with open(f"{post_processed_file}", "r") as f:
        post_processed_file_dict = yaml.load(f, Loader=yaml.FullLoader)
    if 'fixing_results' not in post_processed_file_dict:
        print(f"No fixing results found in {post_processed_file}",
              f"for dataset: {dataset_name}, subtask: {subtask_name}, "
              f"processed_data_label: {processed_data_label}. Removing the file.")
        # remove the file
        post_processed_file.unlink()


for dataset_name in dataset_name_options:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    subtask_names = project_manager.get_available_subtasks()
    for subtask_name in subtask_names:
        processed_data_labels = project_manager.get_available_processed_data_labels_for_subtask(subtask_name)
        script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name)
        script_names = [script_path.stem for script_path in script_path_list]
        processed_data_label = "0"
        for script_name in script_names:
            constraints_path = project_manager.get_constraints_path(
                subtask_name, processed_data_label, script_name)
            constraint_validation_results_dir = project_manager.get_constraints_validation_path(
                subtask_name, processed_data_label, script_name)
            post_processed_file_list = list(constraints_path.glob("post_processed_prismadv--*.yaml"))
            for post_processed_file in post_processed_file_list:
                model_name = post_processed_file.stem.split("--")[1]
                if model_name not in model_names:
                    continue
                cleanup(
                    dataset_name=dataset_name,
                    subtask_name=subtask_name,
                    processed_data_label=processed_data_label,
                    post_processed_file=post_processed_file,
                )
