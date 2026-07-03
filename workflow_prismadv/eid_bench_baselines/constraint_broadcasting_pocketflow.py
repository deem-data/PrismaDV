"""Broadcast PocketFlow constraints from the clean label to all other processed-data labels."""
import shutil

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

dataset_name_options = ["imdb", "hr_analytics", "sleep_health", "students", "IPL_win_prediction"]
subtask_options = ["general_task"]
source_label = "0"

for dataset_name in dataset_name_options:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for subtask_name in subtask_options:
        try:
            project_manager.get_subtask_description(subtask_name=subtask_name)
        except ValueError:
            continue
        labels = project_manager.get_available_processed_data_labels_for_subtask(subtask_name)
        other_labels = [label for label in labels if label != source_label]
        script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name)
        for script_path in script_path_list:
            src_dir = project_manager.get_constraints_path(subtask_name, source_label, script_path.stem)
            src_files = list(src_dir.glob("pocketflow--gpt-5--*.yaml"))
            if not src_files:
                continue
            for other_label in other_labels:
                dst_dir = project_manager.get_constraints_path(subtask_name, other_label, script_path.stem)
                dst_dir.mkdir(parents=True, exist_ok=True)
                for src in src_files:
                    dst = dst_dir / src.name
                    if dst.exists():
                        print(f"exists, skip: {dst}")
                        continue
                    shutil.copy(src, dst)
                    print(f"{src} -> {dst}")
