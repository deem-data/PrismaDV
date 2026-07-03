from workflow_prismadv.eid_bench_ablation_studies.pipelines.run_prismadv_post_processing_wo_code_correction import (
    run_prismadv_post_processing_wo_code_correction
)

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
model_names = ["gpt-5"]

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
            constraint_file_list = list(constraints_path.glob("prismadv--*.yaml"))
            for constraint_file in constraint_file_list:
                model_name = constraint_file.stem.split("--")[1]
                if model_name not in model_names:
                    continue
                print(
                    f"Post-processing constraints for dataset: {dataset_name},\n"
                    f"subtask: {subtask_name}, processed_data_label: {processed_data_label},\n"
                    f"script: {script_name}, constraint_file: {constraint_file.name}"
                )
                run_prismadv_post_processing_wo_code_correction(
                    dataset_name=dataset_name,
                    subtask_name=subtask_name,
                    processed_data_label=processed_data_label,
                    constraint_file=constraint_file,
                )
