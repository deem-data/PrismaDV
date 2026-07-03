import pandas as pd

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow.cad_evaluation.pipelines.run_prismadv_cad import run_prismadv_cad

dataset_selections = ["imdb", "IPL_win_prediction", "sleep_health", "hr_analytics", "students"]
df_list = []

for dataset_name in dataset_selections:
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
            constraint_file_list = list(constraints_path.glob("prismadv--*.yaml"))
            for constraint_file in constraint_file_list:
                print(
                    f"Constraints for dataset: {dataset_name},\n"
                    f"subtask: {subtask_name}, processed_data_label: {processed_data_label},\n"
                    f"script: {script_name}, constraint_file: {constraint_file.name}"
                )
                single_df = run_prismadv_cad(
                    dataset_name=dataset_name,
                    subtask_name=subtask_name,
                    constraint_file=constraint_file,
                )
                if single_df is not None:
                    df_list.append(single_df)
full_df = pd.concat(df_list, ignore_index=True)

# save to disk
full_df.to_csv("tables/cad_results.csv", index=False)
