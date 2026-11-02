import argparse
from concurrent.futures import ThreadPoolExecutor

from prismadv.project_manager import is_clean_label
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.runtime_environments import PythonExecutor
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_experiments.code_execution.run_general_python_code_without_assertions import \
    run_general_python_code_without_assertions

# Runs after _1_code_execution.py. Only general tasks have assertion blocks.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run failed tasks again without their assertion blocks")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name to process (e.g., 'imdb', 'students'). If not provided, processes all datasets."
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Only execute this processed-data label. Default: every corrupted label."
    )
    parser.add_argument("--workers", type=int, default=1, help="Labels executed in parallel.")
    args = parser.parse_args()

    all_datasets = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
    dataset_name_options = [args.dataset_name] if args.dataset_name else all_datasets

    executor = PythonExecutor()
    jobs = []
    for dataset_name in dataset_name_options:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        if "general_task" not in project_manager.get_available_subtasks():
            continue
        for processed_data_label in project_manager.get_available_processed_data_labels_for_subtask("general_task"):
            if is_clean_label(processed_data_label):
                continue
            if args.label is not None and str(processed_data_label) != args.label:
                continue
            jobs.append((dataset_name, f"{processed_data_label}"))

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda job: run_general_python_code_without_assertions(
            dataset_name=job[0], subtask_name="general_task", processed_data_label=job[1], executor=executor
        ), jobs))
