from prismadv.data_models.code_container import CodeContainer
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.runtime_environments import PythonExecutor
from prismadv.utils import get_project_root


def run_general_python_code_for_assertion_selection(dataset_name, subtask_name, processed_data_label, single_script=""):
    executor = PythonExecutor()
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for script_path in project_manager.get_available_script_path_list_for_subtask(subtask_name):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        clean = True
        input_path = project_manager.get_new_data_path(subtask_name, processed_data_label, clean=clean)
        code: CodeContainer = FileLoader.load_py_file(script_path)
        print(script_path)
        code_without_assertions, assertions = code.extract_assertions()

        results = {}
        for assertion_idx in range(len(assertions)):
            code_with_inserted_assertions = code_without_assertions.insert_assertions(
                [assertions[assertion_idx]]
            )
            response = executor.run_script(
                project_name=dataset_name,
                input_path=input_path,
                script_context=code_with_inserted_assertions,
                output_path=None,
            )

            if response.startswith("Success"):
                results[assertion_idx] = {
                    "status": True,
                    "message": response
                }
            else:
                results[assertion_idx] = {
                    "status": False,
                    "message": response
                }
        print(f"results: {results}")
        num_success = sum(1 for r in results.values() if r["status"])
        num_failed = sum(1 for r in results.values() if not r["status"])
        print(f"results on {script_path.stem}:")
        print(f"num_success: {num_success}, num_failed: {num_failed}")
