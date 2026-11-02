import argparse

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_baselines.pipelines.run_mini_swe_agent import run_mini_swe_agent
from workflow_prismadv.utils.replicates import add_replicate_argument, add_script_argument

dataset_name_options = ["imdb", "hr_analytics", "sleep_health", "students", "IPL_win_prediction"]
subtask_options = ["general_task"]
model_name_options = ['gpt-5']  # the paper's row; --model_name runs another
processed_data_label = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", choices=dataset_name_options, help="Run only this dataset")
add_replicate_argument(parser)
add_script_argument(parser)
parser.add_argument("--model_name", default=None, help="Run only this model (default: the paper's)")
args = parser.parse_args()

selected_datasets = [args.dataset_name] if args.dataset_name else dataset_name_options

for dataset_name in selected_datasets:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for subtask_name in subtask_options:
        try:
            task_description = project_manager.get_subtask_description(subtask_name=subtask_name)
        except ValueError as e:
            print(f"Skipping invalid subtask '{subtask_name}' for dataset '{dataset_name}': {e}")
            continue
        for model_name in ([args.model_name] if args.model_name else model_name_options):
            print(f"running mini swe agent for dataset: {dataset_name}, subtask: {subtask_name}, "
                  f"processed_data_label: {processed_data_label}, model_name: {model_name}, "
                  f"replicate: {args.replicate}")
            run_mini_swe_agent(
                dataset_name=dataset_name,
                subtask_name=subtask_name,
                processed_data_label=processed_data_label,
                model_name=model_name,
                replicate=args.replicate,
                script_stems=args.script_stems,
            )
