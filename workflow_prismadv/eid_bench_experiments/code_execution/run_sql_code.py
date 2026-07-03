import shutil

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.runtime_environments import DuckDBExecutor
from prismadv.utils import get_task_instance, get_project_root


def run_general_sql_code(dataset_name, subtask_name, processed_data_label, single_script=""):
    executor = DuckDBExecutor()
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    original_data_path = project_manager.dataset_path
    for script_path in project_manager.get_available_script_path_list_for_subtask(subtask_name):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        task_instance = get_task_instance(script_path)
        for clean in [True, False]:
            input_path = project_manager.get_new_data_path(subtask_name, processed_data_label, clean=clean)
            output_path = project_manager.get_execution_output_path(
                subtask_name,
                processed_data_label,
                script_path.stem,
                clean=clean
            )
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            executor.run_script(
                project_name=original_data_path.name,
                script_context=task_instance.original_script,
                input_path=input_path,
                output_path=output_path
            )
