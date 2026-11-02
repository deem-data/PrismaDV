import argparse
import sys

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_baselines.pipelines.run_few_shot import run_few_shot
from workflow_prismadv.utils.replicates import add_replicate_argument, add_script_argument

dataset_name_options = ["imdb", "hr_analytics", "sleep_health", "students", "IPL_win_prediction"]
subtask_options = ["general_task"]
model_name_options = ["gpt-4o", "gpt-4.1", "gpt-5-mini", "gpt-5", "gpt-6-sol-medium", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.7-flash", "gemini-3.8-flash"]
processed_data_label = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", choices=dataset_name_options, help="Run only this dataset")
parser.add_argument("--model_name", choices=model_name_options, help="Run only this model")
add_replicate_argument(parser)
add_script_argument(parser)
args = parser.parse_args()

selected_datasets = [args.dataset_name] if args.dataset_name else dataset_name_options
selected_models = [args.model_name] if args.model_name else model_name_options
failed_runs = []

for dataset_name in selected_datasets:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for subtask_name in subtask_options:
        try:
            task_description = project_manager.get_subtask_description(subtask_name=subtask_name)
        except ValueError as e:
            print(f"Skipping invalid subtask '{subtask_name}' for dataset '{dataset_name}': {e}")
            continue
        for model_name in selected_models:
            print(f"running for dataset: {dataset_name}, subtask: {subtask_name}, "
                  f"processed_data_label: {processed_data_label}, model_name: {model_name}, "
                  f"replicate: {args.replicate}")
            failed = run_few_shot(
                dataset_name=dataset_name,
                subtask_name=subtask_name,
                processed_data_label=processed_data_label,
                model_name=model_name,
                replicate=args.replicate,
                script_stems=args.script_stems,
            )
            failed_runs += [f"{dataset_name}/{stem}/{model_name}" for stem in failed]

if failed_runs:
    # Nonzero so a driver can rerun; finished scripts are skipped on the rerun.
    sys.exit(f"{len(failed_runs)} script(s) failed: {', '.join(failed_runs)}")
