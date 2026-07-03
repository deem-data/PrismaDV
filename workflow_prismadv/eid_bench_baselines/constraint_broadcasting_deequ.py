from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
subtask_options = ["general_task"]

for dataset_name in dataset_name_options:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for subtask_name in subtask_options:
        try:
            task_description = project_manager.get_subtask_description(subtask_name=subtask_name)
        except ValueError as e:
            continue
        print(f"Processing dataset: {dataset_name}, subtask: {subtask_name}")

        processed_data_labels = project_manager.get_available_processed_data_labels_for_subtask(subtask_name)
        processed_data_label = "0"

        other_labels = [label for label in processed_data_labels if label != processed_data_label]
        existing_file = project_manager.get_task_agnostic_constraint_path(subtask_name,
                                                                          processed_data_label) / "deequ_constraints.yaml"
        if not existing_file.exists():
            continue
        for other_label in other_labels:
            other_constraints_path = project_manager.get_task_agnostic_constraint_path(subtask_name,
                                                                                       other_label)
            other_constraints_path.mkdir(parents=True, exist_ok=True)
            target_file = other_constraints_path / "deequ_constraints.yaml"
            if target_file.exists():
                print(f"File {target_file} already exists. Skipping...")
            else:
                print(f"Copying {existing_file} to {target_file}")
                target_file.write_text(existing_file.read_text())
