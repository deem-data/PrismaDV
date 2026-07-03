#!/usr/bin/env python3
"""Isolation Forest task-agnostic baseline for EIDBench-synth."""
from __future__ import annotations

import argparse

from prismadv.data_models import Constraints, ValidationResults
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.task_agnostic_baselines.novelty_models import (
    learn_isolation_forest,
    should_be_rejected,
)

DATASET_NAME_OPTIONS = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
CLEAN_LABEL = "0"
CONSTRAINT_FILENAME = "isolation_forest_constraints.yaml"
NOVELTY_COLUMN = "__novelty__"
SUGGESTION = "isolation_forest.detects_no_anomaly"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=None,
                        help="Restrict to one or more dataset ids (default: all).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validation_result(rejected: bool) -> ValidationResults:
    reason = (
        "isolation-forest flagged at least one row as a novelty/anomaly"
        if rejected
        else ""
    )
    return ValidationResults.from_dict({
        NOVELTY_COLUMN: {
            "code": [{
                "suggestion": SUGGESTION,
                "status": not rejected,
                "reason_if_failed": reason,
                "level": "error",
            }]
        }
    })


def run_dataset(dataset_name: str, *, overwrite: bool = False) -> None:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for subtask_name in project_manager.get_available_subtasks():
        try:
            project_manager.get_subtask_description(subtask_name=subtask_name)
        except ValueError as exc:
            print(f"Skipping invalid subtask '{subtask_name}' for '{dataset_name}': {exc}")
            continue

        train_data = FileLoader.load_csv(
            project_manager.get_new_test_data_path(subtask_name, CLEAN_LABEL, clean=True)
        )
        detector = learn_isolation_forest(train_data)

        for processed_data_label in project_manager.get_available_processed_data_labels_for_subtask(subtask_name):
            if int(processed_data_label) == 0:
                continue

            constraint_path = project_manager.get_task_agnostic_constraint_path(
                subtask_name, processed_data_label
            ) / CONSTRAINT_FILENAME
            validation_dir = project_manager.get_task_agnostic_constraints_validation_path(
                subtask_name, processed_data_label
            )
            validation_path = (
                validation_dir
                / f"validation_results_on_corrupted_test_data__{constraint_path.stem}.yaml"
            )
            if validation_path.exists() and not overwrite:
                print(f"  [{dataset_name}/{subtask_name}/{processed_data_label}] exists, skipping.")
                continue

            test_data = FileLoader.load_csv(
                project_manager.get_new_test_data_path(subtask_name, processed_data_label, clean=False)
            )
            rejected = should_be_rejected(detector, test_data)

            constraint_path.parent.mkdir(parents=True, exist_ok=True)
            Constraints().save_to_yaml(constraint_path)
            validation_dir.mkdir(parents=True, exist_ok=True)
            _validation_result(rejected).save_to_yaml(validation_path)
            print(
                f"  [{dataset_name}/{subtask_name}/{processed_data_label}] "
                f"rejected={rejected} -> {validation_path}"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = args.dataset if args.dataset else DATASET_NAME_OPTIONS
    for dataset_name in datasets:
        print(f"=== {dataset_name} ===")
        run_dataset(dataset_name, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
