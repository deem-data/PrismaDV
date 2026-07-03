import argparse
import warnings

from prismadv.data_models.config import PrismaDVConfig
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow.e2e_evaluation.pipelines.run_prismadv import run_prismadv

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyspark")

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
subtask_options = ["general_task"]
model_name_options = ["gpt-4o", "gpt-4.1", "gpt-5-mini", 'gpt-5', 'gemini-2.5-flash', 'gemini-2.5-pro']
processed_data_label = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", choices=dataset_name_options, help="Run only this dataset")
args = parser.parse_args()

if args.dataset_name:
    selected_datasets = [args.dataset_name]
else:
    selected_datasets = dataset_name_options

for dataset_name in selected_datasets:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for subtask_name in subtask_options:
        try:
            task_description = project_manager.get_subtask_description(subtask_name=subtask_name)
        except ValueError as e:
            print(f"Skipping invalid subtask '{subtask_name}' for dataset '{dataset_name}': {e}")
            continue
        for model_name in model_name_options:
            prismadv_config = PrismaDVConfig.from_dict({
                "model": {
                    "use_async": True,
                    "use_dataflow": True,
                    "correlation_detection": True,
                    "downstream_task_description": task_description
                },
                "llm": {
                    "model_name": model_name,
                    "temperature": 0.6,
                    "max_tokens": None,
                    "seed": None
                },
                "io": {
                    "overwrite": False,
                }
            })
            print(f"running for dataset: {dataset_name}, subtask: {subtask_name}, "
                  f"processed_data_label: {processed_data_label}, model_name: {model_name}")
            run_prismadv(
                dataset_name=dataset_name,
                subtask_name=subtask_name,
                processed_data_label=processed_data_label,
                prismadv_config=prismadv_config
            )
