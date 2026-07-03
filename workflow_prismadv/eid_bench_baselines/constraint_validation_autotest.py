"""Validate corrupted EIDBench batches with AutoTest (task-agnostic baseline)."""
import oyaml as yaml

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_baselines.pipelines.run_autotest import autotest_detect_csvs

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
SDC_NAME = "rt_train"
RESULT_STEM = "autotest_constraints"


def anomalies_from_detection(detected_df):
    """Turn AutoTest's detected-outlier table into a {column: description} dict."""
    anomalies = {}
    for _, row in detected_df.iterrows():
        header = str(row["header"])
        desc = f"AutoTest flagged outlier(s) {row['outlier']} (confidence {float(row['conf']):.4f})"
        anomalies[header] = desc
    return anomalies


jobs = []
id_to_csv = {}
for dataset_name in dataset_name_options:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for subtask_name in project_manager.get_available_subtasks():
        for processed_data_label in project_manager.get_available_processed_data_labels_for_subtask(subtask_name):
            if int(processed_data_label) == 0:
                continue
            validation_results_dir = project_manager.get_task_agnostic_constraints_validation_path(
                subtask_name, processed_data_label,
            )
            validation_results_path = (
                validation_results_dir / f"validation_results_on_corrupted_test_data__{RESULT_STEM}.yaml"
            )
            validation_results_path.parent.mkdir(parents=True, exist_ok=True)
            if validation_results_path.exists():
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

if id_to_csv:
    print(f"Running AutoTest on {len(id_to_csv)} corrupted batches...")
    detections = autotest_detect_csvs(id_to_csv, sdc_name=SDC_NAME, run_name="eid_validation")

    for job_id, validation_results_path in jobs:
        detected_df = detections[job_id]
        anomalies = anomalies_from_detection(detected_df)
        result = {
            "pass": (len(anomalies) == 0),
            "anomalies": anomalies,
        }
        with open(validation_results_path, "w") as f:
            yaml.dump(result, f, sort_keys=False)
        print(f"{job_id}: pass={result['pass']} ({len(anomalies)} columns flagged)")
else:
    print("Nothing to validate.")
