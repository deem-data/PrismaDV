from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
subtask_options = ["general_task"]
model_name_options = ["gpt-4o", "gpt-4.1", "gpt-5-mini", "gpt-5", "gemini-2.5-flash", "gemini-2.5-pro"]
processed_data_label_to_broadcast = "0"
registered_method_name = ["prismadv", "single_shot", "few_shot", "post_processed_prismadv", "mini_swe_agent"]

overwrite = False  # Set to False to keep original behavior


def _is_registered(fname: str) -> bool:
    return any(fname.startswith(method) for method in registered_method_name)


for dataset_name in dataset_name_options:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for subtask_name in subtask_options:
        try:
            task_description = project_manager.get_subtask_description(subtask_name=subtask_name)
        except ValueError:
            continue

        print(f"Processing dataset: {dataset_name}, subtask: {subtask_name}")
        processed_data_labels = project_manager.get_available_processed_data_labels_for_subtask(subtask_name)
        other_labels = [label for label in processed_data_labels if label != processed_data_label_to_broadcast]

        script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name=subtask_name)
        for script_path in script_path_list:
            # Base constraints dir for label "0"
            base_constraints_path = project_manager.get_constraints_path(
                subtask_name, processed_data_label_to_broadcast, script_path.stem
            )
            base_constraints_path.mkdir(parents=True, exist_ok=True)

            # Existing files in base
            base_files = [f for f in base_constraints_path.glob("*.yaml") if _is_registered(f.name)]
            base_names = {f.name for f in base_files}

            # Broadcast copies to others
            for other_label in other_labels:
                other_constraints_path = project_manager.get_constraints_path(
                    subtask_name, other_label, script_path.stem
                )
                other_constraints_path.mkdir(parents=True, exist_ok=True)

                # 1) Copy missing ones or overwrite
                for base_file in base_files:
                    target_file = other_constraints_path / base_file.name
                    if target_file.exists() and not overwrite:
                        print(f"File {target_file} already exists. Skipping...")
                    else:
                        action = "Overwriting" if target_file.exists() else "Copying"
                        print(f"{action} {base_file} to {target_file}")
                        target_file.write_text(base_file.read_text())

                # 2) Delete extras not in base
                other_files = [f for f in other_constraints_path.glob("*.yaml") if _is_registered(f.name)]
                for f in other_files:
                    if f.name not in base_names:
                        print(f"Deleting {f} as it does not exist in label {processed_data_label_to_broadcast}")
                        try:
                            f.unlink()
                        except Exception as e:
                            print(f"Failed to delete {f}: {e}")
