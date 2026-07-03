"""Score AutoTest on EIDBench (precision/recall/F1)."""
import numpy as np
import oyaml as yaml
import pandas as pd

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.utils.results_analysis import format_results_latex, build_confusion_matrices

dataset_name_options = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
model_order = ['']
CONSTRAINT_STEM = "autotest_constraints"

results_df = pd.DataFrame()
for dataset_name in dataset_name_options:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    for subtask_name in project_manager.get_available_subtasks():
        processed_data_labels = project_manager.get_available_processed_data_labels_for_subtask(subtask_name)
        script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name)
        script_names = [p.stem for p in script_path_list]
        for processed_data_label in processed_data_labels:
            if int(processed_data_label) == 0:
                continue
            for script_name in script_names:
                cv_dir = project_manager.get_task_agnostic_constraints_validation_path(
                    subtask_name, processed_data_label)
                for is_clean in [True, False]:
                    tag = "clean" if is_clean else "corrupted"
                    vpath = cv_dir / f"validation_results_on_{tag}_test_data__{CONSTRAINT_STEM}.yaml"
                    try:
                        with open(vpath) as f:
                            validation_results = yaml.safe_load(f)
                    except FileNotFoundError:
                        continue
                    num_failed_error = len(validation_results['anomalies'])
                    predicted_as_safe = (num_failed_error == 0)

                    exec_path = project_manager.get_execution_output_validation_path(
                        subtask_name, processed_data_label, script_name) / "basic_metrics_evaluation.json"
                    try:
                        with open(exec_path) as f:
                            execution_results = yaml.load(f, Loader=yaml.FullLoader)
                    except FileNotFoundError:
                        continue
                    is_safe = execution_results['clean_data_is_safe'] if is_clean \
                        else execution_results['corrupted_data_is_safe']

                    results_df = pd.concat([results_df, pd.DataFrame([{
                        "llm_name": "", "llm_temperature": np.nan,
                        "dataset_name": dataset_name, "subtask_name": subtask_name,
                        "processed_data_label": processed_data_label, "script_name": script_name,
                        "is_clean": is_clean,
                        "num_passed_warning": np.nan, "num_failed_warning": np.nan,
                        "num_failed_error": num_failed_error, "num_passed_error": np.nan,
                        "num_non_compilable": 0, "total_constraints": 0,
                        "predicted_as_safe": predicted_as_safe, "is_safe": is_safe,
                    }])], ignore_index=True)

print(f"Collected {len(results_df)} (label x script) evaluation units.\n")

cm = build_confusion_matrices(results_df)


def agg(df):
    tp = df["safe,predicted_as_safe"].sum()
    fn = df["safe,predicted_as_unsafe"].sum()
    fp = df["unsafe,predicted_as_safe"].sum()
    tn = df["unsafe,predicted_as_unsafe"].sum()
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p and r == r and (p + r)) else float("nan")
    return tp, fn, fp, tn, p, r, f1

num = cm.copy()
for c in ["safe,predicted_as_safe", "safe,predicted_as_unsafe",
          "unsafe,predicted_as_safe", "unsafe,predicted_as_unsafe"]:
    num[c] = pd.to_numeric(num[c], errors="coerce").fillna(0)

print(f"{'dataset':22s} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4}  {'prec':>6} {'rec':>6} {'f1':>6}")
print("-" * 64)
for ds in dataset_name_options:
    sub = num[num["dataset_name"] == ds]
    if sub.empty:
        continue
    tp, fn, fp, tn, p, r, f1 = agg(sub)
    print(f"{ds:22s} {tp:4d} {fn:4d} {fp:4d} {tn:4d}  {p:6.3f} {r:6.3f} {f1:6.3f}")
print("-" * 64)
tp, fn, fp, tn, p, r, f1 = agg(num)
print(f"{'OVERALL':22s} {tp:4d} {fn:4d} {fp:4d} {tn:4d}  {p:6.3f} {r:6.3f} {f1:6.3f}")

print("\n===== LaTeX (paper table) =====")
print(format_results_latex(
    confusion_matrex_df=cm,
    dataset_name_options=dataset_name_options,
    model_order=model_order,
    decimals=1,
    model_label_prefix="autotest",
    include_overall=True,
))

out_csv = get_project_root() / "data_processed" / "autotest_eidbench_results_df.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved per-unit results to {out_csv}")
