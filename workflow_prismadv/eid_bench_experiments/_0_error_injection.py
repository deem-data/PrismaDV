"""Inject the EIDBench-synth error configs into every dataset's processed labels.

`benchmarks/EIDBench-synth/<dataset>/errors/error-group-<k>/general_task_<n>.yaml`
becomes `data_processed/<dataset>/general/<k>-<n>/`: the clean and the corrupted
`new_data` plus the config it was made from. Every corruption step must set its
own `random_state`; the seed never comes from the label. Label 0 is the clean
baseline: it has no config and receives the uncorrupted split only.
"""
import argparse
import shutil

from prismadv.error_injection.managers.ml_inference import MLInferenceErrorInjectionManager
from prismadv.error_injection.managers.sql_query import GeneralErrorInjectionManager
from prismadv.project_manager import CLEAN_LABEL, config_labels, config_path
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

ALL_DATASETS = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]


def build_manager(project_manager, subtask_name):
    raw_file_path = project_manager.files_root
    target_table_name = project_manager.raw_data_info["target_table_name"]
    processed_data_dir = project_manager.get_base_processed_data_path_for_subtask(subtask_name)

    if subtask_name in ["classification", "regression"]:
        subtask_info = project_manager.get_subtask_info(subtask_name)
        return MLInferenceErrorInjectionManager(
            raw_file_path=raw_file_path,
            target_table_name=target_table_name,
            target_column_name=subtask_info['target_column_name'],
            processed_data_dir=processed_data_dir,
            submission_default_value=subtask_info['submission_default_value']
        )
    if subtask_name in ["bi", "dev", "feature_engineering", "info", "general_task"]:
        return GeneralErrorInjectionManager(
            raw_file_path=raw_file_path,
            target_table_name=target_table_name,
            processed_data_dir=processed_data_dir,
            sample_size=1.0
        )
    raise ValueError(f"Downstream task {subtask_name} is not supported.")


# Everything under a label that is derived from the injected data. Wiped by
# `--force` so a regenerated label carries no result computed on its old data.
# `constraints/` (LLM artifacts made from the clean split) is kept.
DERIVED_LABEL_ENTRIES = (
    "files_with_clean_new_data",
    "files_with_corrupted_new_data",
    "error_injection_config.yaml",
    "output",
    "output_validation",
    "constraints_validation",
)


def clear_derived(processed_data_path):
    for entry in DERIVED_LABEL_ENTRIES:
        target = processed_data_path / entry
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def clean_baseline(dataset_name, subtask_name, force=False):
    """Write label 0: the uncorrupted split, no config."""
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    manager = build_manager(project_manager, subtask_name)
    processed_data_path = project_manager.get_base_processed_data_path_for_subtask(subtask_name) / CLEAN_LABEL

    if force and processed_data_path.exists():
        clear_derived(processed_data_path)
    if (processed_data_path / "files_with_clean_new_data" / "new_data.csv").exists():
        print(f"Clean baseline for {dataset_name}/{subtask_name} already exists at {processed_data_path}.")
        return
    if not isinstance(manager, GeneralErrorInjectionManager):
        raise NotImplementedError("clean baseline export is only implemented for the general manager")
    manager.save_clean_data(CLEAN_LABEL)
    print(f"Clean baseline for {dataset_name}/{subtask_name} written to {processed_data_path}.")


def error_injection(dataset_name, subtask_name, label, force=False):
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    manager = build_manager(project_manager, subtask_name)
    error_config_path = config_path(project_manager.errors_root, subtask_name, label)
    processed_data_path = project_manager.get_base_processed_data_path_for_subtask(subtask_name) / label

    if force and processed_data_path.exists():
        clear_derived(processed_data_path)

    corrupts = manager.load_error_injection_config(error_injection_config_path=error_config_path)
    unseeded = [type(corrupt).__name__ for corrupt in corrupts if getattr(corrupt, "random_state", None) is None]
    if unseeded:
        raise ValueError(f"{error_config_path}: every corruption step needs a random_state; missing in {unseeded}")

    if manager.check_corrupts_existence(corrupts, processed_data_path):
        print(f"The error configuration in {error_config_path.name} has already been applied to {dataset_name} "
              f"for the downstream task {subtask_name} at label {label}. Skipping error injection.")
        return

    manager.error_injection(corrupts)
    manager.save_data(label)
    print(f"Error injection for {dataset_name} is done.\nThe corrupted data is saved in {processed_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject errors into datasets")
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
        help="Only this processed-data label: 0 or <group>-<n>, e.g. 12-3 for "
             "errors/error-group-12/general_task_3.yaml. Default: label 0 and every config."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-inject even if the label already holds this config, wiping the label's derived data, "
             "outputs, ground truth and validation results first."
    )
    args = parser.parse_args()

    dataset_name_options = [args.dataset_name] if args.dataset_name else ALL_DATASETS

    for dataset_name in dataset_name_options:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        for subtask_name in project_manager.get_available_subtasks():
            labels = [CLEAN_LABEL] + config_labels(project_manager.errors_root, subtask_name)
            if args.label is not None:
                if args.label not in labels:
                    raise SystemExit(f"{dataset_name}/{subtask_name}: no label {args.label}")
                labels = [args.label]
            for label in labels:
                print(f"\tStarting error injection for dataset: {dataset_name}, downstream task: {subtask_name}, "
                      f"label: {label}")
                if label == CLEAN_LABEL:
                    clean_baseline(dataset_name, subtask_name, force=args.force)
                else:
                    error_injection(dataset_name, subtask_name, label, force=args.force)
