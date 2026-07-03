import argparse
import json
import warnings

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow.e2e_evaluation.metrics.calculator import MetricsCalculation

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyspark")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate performance metrics on execution results")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name to process (e.g., 'imdb', 'students'). If not provided, processes all datasets."
    )
    args = parser.parse_args()

    all_datasets = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
    dataset_name_options = [args.dataset_name] if args.dataset_name else all_datasets

    metric_calculator = MetricsCalculation()

    for dataset_name in dataset_name_options:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        dq_manager = DeequDataQualityManager()
        subtask_names = project_manager.get_available_subtasks()
        for subtask_name in subtask_names:
            processed_data_labels = project_manager.get_available_processed_data_labels_for_subtask(subtask_name)
            script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name)
            script_names = [script_path.stem for script_path in script_path_list]
            for processed_data_label in processed_data_labels:
                for script_name in script_names:
                    execution_output_dir = project_manager.get_execution_output_dir(
                        subtask_name, processed_data_label, script_name
                    )
                    execution_output_validation_path = project_manager.get_execution_output_validation_path(
                        subtask_name, processed_data_label, script_name
                    )
                    print(f"Dataset: {dataset_name}, Subtask: {subtask_name}, Script: {script_name}, "
                          f"Processed Data Label: {processed_data_label}")
                    result = metric_calculator.calculate(
                        sub_task=subtask_name, script_output_dir=execution_output_dir
                    )
                    output_file = execution_output_validation_path / f"basic_metrics_evaluation.json"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Saving evaluation result to {output_file}")
                    with output_file.open("w", encoding="utf-8") as f:
                        json.dump(result, f, indent=4, ensure_ascii=False)
