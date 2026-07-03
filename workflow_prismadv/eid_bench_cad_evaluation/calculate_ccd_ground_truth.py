import oyaml as yaml

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow.cad_evaluation.code_execution.run_ccd_detection import \
    run_ccd_detection

# test which assertions in the code are valid and which are not.
if __name__ == "__main__":
    output_dir = get_project_root() / "workflow/cad_evaluation/cad_ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_selections = ["imdb", "IPL_win_prediction", "sleep_health", "hr_analytics", "students"]
    dataset_results = {}
    for dataset_name in dataset_selections:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        available_subtasks = project_manager.get_available_subtasks()
        for subtask in available_subtasks:
            processed_data_label = "0"
            if subtask == "general_task":
                subtask_results = run_ccd_detection(
                    dataset_name=dataset_name, subtask_name=subtask,
                    processed_data_label=f"{processed_data_label}"
                )
                print(subtask_results)
                dataset_results[f"{dataset_name}"] = {f"{subtask}": subtask_results}
    for dataset_name in dataset_selections:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        for script_name in dataset_results[dataset_name]["general_task"]:
            annotation_file_path = project_manager.get_annotation_file_path_from_subtask(subtask_name="general_task",
                                                                                         script_name=script_name,
                                                                                         ok_if_not_exist=True)
            with open(annotation_file_path, "r") as f:
                annotation_dict = yaml.load(f, Loader=yaml.FullLoader)
                if annotation_dict['annotations']['correlated_columns'] is None:
                    annotation_dict['annotations']['correlated_columns'] = \
                        dataset_results[dataset_name]["general_task"][
                            script_name
                        ]
                    with open(annotation_file_path, "w") as fw:
                        yaml.dump(annotation_dict, fw)
                    print(f"Updated annotation file {annotation_file_path} with accessed columns: "
                          f"{dataset_results[dataset_name]['general_task'][script_name]}")
                else:
                    print(f"Annotation file {annotation_file_path} already has accessed columns: "
                          f"{annotation_dict['annotations']['correlated_columns']}. Skipping...")
