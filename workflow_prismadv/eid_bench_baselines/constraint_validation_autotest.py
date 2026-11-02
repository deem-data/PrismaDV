"""Validate corrupted EIDBench batches with AutoTest (task-agnostic baseline)."""
from __future__ import annotations

import argparse

import oyaml as yaml

from prismadv.project_manager import is_clean_label
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_baselines.pipelines.run_autotest import (
    autotest_detect_csvs,
    autotest_table_timings,
)

DATASET_NAME_OPTIONS = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
SDC_NAME = "rt_train"
RESULT_STEM = "autotest_constraints"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=None,
                        help="Restrict to one or more dataset ids (default: all).")
    parser.add_argument("--label", action="append", default=None,
                        help="Restrict to one or more labels <group>-<n> (default: all).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-name", default=None,
                        help="Staging directory name under AUTOTEST_STAGING. Jobs that run at the same "
                             "time need different names (default: derived from --dataset).")
    return parser.parse_args(argv)


def anomalies_from_detection(detected_df):
    """Turn AutoTest's detected-outlier table into a {column: description} dict."""
    anomalies = {}
    for _, row in detected_df.iterrows():
        header = str(row["header"])
        desc = f"AutoTest flagged outlier(s) {row['outlier']} (confidence {float(row['conf']):.4f})"
        anomalies[header] = desc
    return anomalies


def collect_jobs(datasets, labels=None, *, overwrite=False):
    jobs = []
    id_to_csv = {}
    for dataset_name in datasets:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        for subtask_name in project_manager.get_available_subtasks():
            for processed_data_label in project_manager.get_available_processed_data_labels_for_subtask(subtask_name):
                if is_clean_label(processed_data_label):
                    continue
                if labels and processed_data_label not in labels:
                    continue
                validation_results_dir = project_manager.get_task_agnostic_constraints_validation_path(
                    subtask_name, processed_data_label,
                )
                validation_results_path = (
                    validation_results_dir / f"validation_results_on_corrupted_test_data__{RESULT_STEM}.yaml"
                )
                validation_results_path.parent.mkdir(parents=True, exist_ok=True)
                if validation_results_path.exists() and not overwrite:
                    print(f"Validation results file {validation_results_path} already exists. Skipping...")
                    continue
                test_data_path = project_manager.get_new_test_data_path(
                    subtask_name, processed_data_label, clean=False
                )
                if not test_data_path.exists():
                    print(f"Test data {test_data_path} missing. Skipping...")
                    continue
                job_id = f"{dataset_name}__{subtask_name}__{processed_data_label}"
                jobs.append((job_id, validation_results_path))
                id_to_csv[job_id] = test_data_path
    return jobs, id_to_csv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = args.dataset if args.dataset else DATASET_NAME_OPTIONS
    run_name = args.run_name or ("eid_validation" if not args.dataset else "eid_validation__" + "_".join(datasets))
    jobs, id_to_csv = collect_jobs(datasets, args.label, overwrite=args.overwrite)
    if not id_to_csv:
        print("Nothing to validate.")
        return 0

    print(f"Running AutoTest on {len(id_to_csv)} corrupted batches...")
    detections = autotest_detect_csvs(id_to_csv, sdc_name=SDC_NAME, run_name=run_name, skip_missing=True)
    timings = autotest_table_timings(id_to_csv, run_name)

    for job_id, validation_results_path in jobs:
        if job_id not in detections:
            print(f"{job_id}: AutoTest failed on this table (see <staging>/outputs/{job_id}.error); no result written")
            continue
        detected_df = detections[job_id]
        anomalies = anomalies_from_detection(detected_df)
        result = {
            "pass": (len(anomalies) == 0),
            "anomalies": anomalies,
        }
        if job_id in timings:
            result["validation_summary"] = timings[job_id]
        with open(validation_results_path, "w") as f:
            yaml.dump(result, f, sort_keys=False)
        print(f"{job_id}: pass={result['pass']} ({len(anomalies)} columns flagged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
