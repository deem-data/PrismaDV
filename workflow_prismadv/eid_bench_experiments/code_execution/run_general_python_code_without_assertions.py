import shutil
from pathlib import Path

from prismadv.data_models.code_container import CodeContainer
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.runtime_environments import PythonExecutor
from prismadv.utils import get_project_root

# Next to results_on_corrupted_new_data. It lies under output/, so `clear_derived` wipes it with the other runs.
WITHOUT_ASSERTIONS_DIR = "results_on_corrupted_new_data_without_assertions"


def get_without_assertions_output_path(project_manager: ProjectManager, subtask_name, processed_data_label,
                                       script_name) -> Path:
    return project_manager.get_execution_output_dir(subtask_name, processed_data_label, script_name) / \
        WITHOUT_ASSERTIONS_DIR


def run_general_python_code_without_assertions(dataset_name, subtask_name, processed_data_label,
                                               executor: PythonExecutor = None, single_script=""):
    """Run each task that failed on the corrupted data again, without its assertion blocks, on the same data.

    A task that then runs to the end gives a wrong result without an error (silent failure); a task that still
    raises crashes. Tasks that ran to the end on the corrupted data are skipped.
    """
    executor = executor or PythonExecutor()
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    input_path = project_manager.get_new_data_path(subtask_name, processed_data_label, clean=False)
    for script_path in project_manager.get_available_script_path_list_for_subtask(subtask_name):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        corrupted_output_path = project_manager.get_execution_output_path(
            subtask_name, processed_data_label, script_path.stem, clean=False
        )
        output_path = get_without_assertions_output_path(
            project_manager, subtask_name, processed_data_label, script_path.stem
        )
        if output_path.exists():
            shutil.rmtree(output_path)
        if not (corrupted_output_path / "error.txt").exists():
            continue
        code_without_assertions, _ = CodeContainer(FileLoader.load_py_file(script_path)).extract_assertions()
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run_script(
            project_name=project_manager.dataset_path.name,
            input_path=input_path,
            script_context=code_without_assertions,
            output_path=output_path,
            timeout=6000
        )
