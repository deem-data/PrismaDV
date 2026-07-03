"""Task-agnostic previous-data novelty-detection baseline (Redyuk et al., EDBT 2021).

This baseline is evaluated ONLY on the *New Data Batches* scenario
(``test_1_cross_new_data``), because it models the data distribution of
previously-observed batches and is blind to tasks. It is N/A for the *New Tasks*
and *New Data Batches + New Tasks* scenarios.

Protocol (mirrors the SIFTA / ``test_1_cross_new_data`` split):
  * Data batches (= ``processed_data_label``) are split 1:1 into D_train (observed)
    and D_new (held-out), reusing a SIFTA reference run's splits when available.
  * Train: fit a one-class kNN novelty detector on the per-batch descriptive-statistics
    feature vectors of the D_train batches (all treated as "acceptable",
    contamination = 1%). Labels of D_train batches are NOT used.
  * Test: for each batch in D_new, emit a single pass/reject decision.
  * The decision is task-agnostic, so it is broadcast across all tasks in
    (T_train u T_val) = ``train_eval_script_name_list`` to form the same
    (task, batch) cells SIFTA is scored on.
  * Metric: F1 with positive class = "unsafe" (reject), identical to the other
    baselines (see ``workflow_sifta/baselines/evaluate_f1.py``).

Usage:
    poetry install --with baseline
    poetry run python workflow_sifta/baselines/redyuk/run.py \
        --dataset_name hr_analytics --split_strategy_from run_20260111_231103

    # Or generate splits from scratch with the SIFTA default seed/ratios:
    poetry run python workflow_sifta/baselines/redyuk/run.py --dataset_name hr_analytics
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_sifta.baselines.redyuk.data_profiler import (
    KNNNoveltyDetector,
    SupervisedKNNClassifier,
    compute_profile,
    infer_column_kinds,
)

# SIFTA default split configuration (see workflow_sifta/optimization.py).
ALL_PROCESSED_DATA_LABELS = [str(i) for i in range(1, 26)]
DEFAULT_LABEL_SPLIT_RATIO = 0.5
DEFAULT_SCRIPT_SPLIT_RATIO = 0.6
DEFAULT_TRAIN_EVAL_SCRIPT_SPLIT_RATIO = 0.5
DEFAULT_RANDOM_SEED = 1


def load_splits_from_reference_run(reference_run_name: str, project_root: Path) -> dict:
    """Load D_train / D_new label lists and the (T_train u T_val) task list."""
    summary_path = project_root / "optimization_runs" / reference_run_name / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Reference run summary not found: {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)
    return {
        "d_train_labels": summary["train_eval_label_list"],
        "d_new_labels": summary["cross_new_data_test_label_list"],
        "train_tasks": summary["train_eval_script_name_list"],
        "new_tasks": summary.get("new_script_name_list", []),
    }


def generate_splits(pm: ProjectManager, subtask: str, seed: int, label_split_ratio: float) -> dict:
    """Generate splits from scratch, matching ``workflow_sifta/optimization.py``."""
    random.seed(seed)

    shuffled_labels = ALL_PROCESSED_DATA_LABELS.copy()
    random.shuffle(shuffled_labels)
    n_train_eval = int(len(shuffled_labels) * label_split_ratio)
    d_train_labels = shuffled_labels[:n_train_eval]
    d_new_labels = shuffled_labels[n_train_eval:]

    all_scripts = sorted(p.stem for p in pm.get_available_script_path_list_for_subtask(subtask))
    shuffled_scripts = all_scripts.copy()
    random.shuffle(shuffled_scripts)
    n_train_eval_scripts = int(len(shuffled_scripts) * DEFAULT_SCRIPT_SPLIT_RATIO)

    return {
        "d_train_labels": d_train_labels,
        "d_new_labels": d_new_labels,
        "train_tasks": shuffled_scripts[:n_train_eval_scripts],   # T_train u T_val
        "new_tasks": shuffled_scripts[n_train_eval_scripts:],     # T_test (held-out)
    }


def read_batch(pm: ProjectManager, subtask: str, label: str, clean: bool) -> pd.DataFrame:
    path = pm.get_new_test_data_path(subtask, label, clean=clean)
    return pd.read_csv(path)


def read_ground_truth_is_safe(pm: ProjectManager, subtask: str, label: str, script: str):
    """Return ``corrupted_data_is_safe`` for a (script, batch) cell, or None if missing."""
    exec_path = (
        pm.get_execution_output_validation_path(subtask, label, script)
        / "basic_metrics_evaluation.json"
    )
    if not exec_path.exists():
        return None
    with open(exec_path) as f:
        return json.load(f).get("corrupted_data_is_safe")


def label_historical_batches(pm, subtask, labels, task_scripts):
    """Label each historical batch good/bad from observed train-task outcomes.

    A batch is BAD if at least one train task failed on it (``corrupted_data_is_safe``
    is False); otherwise GOOD. Cells with missing ground truth are ignored.
    """
    good, bad = [], []
    for lbl in labels:
        failed = any(
            read_ground_truth_is_safe(pm, subtask, lbl, t) is False for t in task_scripts
        )
        (bad if failed else good).append(lbl)
    return good, bad


def compute_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return f1, precision, recall


def _json_num(x):
    return float(x) if isinstance(x, float) and math.isfinite(x) else (None if isinstance(x, float) else x)


def run_dataset(
    dataset_name: str,
    subtask: str,
    splits: dict,
    contamination: float,
    n_neighbors: int,
    project_root: Path,
    fit_mode: str = "good_only",
    init_label: str = "0",
    init_data: str = "clean",
    detector_type: str = "one_class",
    scenario: str = "new_data",
) -> dict:
    pm = ProjectManager(project_root=project_root, dataset_name=dataset_name, downstream_task_type="general")

    d_train_labels = sorted(splits["d_train_labels"], key=int)
    d_new_labels = sorted(splits["d_new_labels"], key=int)
    # Historical batches are ALWAYS labeled from the train tasks (the only task
    # outcomes ever observed). The scoring/broadcast tasks depend on the scenario:
    #   new_data           -> train+eval tasks (test_1_cross_new_data)
    #   new_data_new_task   -> held-out new tasks (test_3_new_script_new_data)
    train_tasks = splits["train_tasks"]
    eval_tasks = train_tasks if scenario == "new_data" else splits["new_tasks"]

    print(f"\n=== {dataset_name}/{subtask} [scenario={scenario}] ===")
    print(f"  D_train (observed) batches: {d_train_labels}")
    print(f"  D_new (held-out)   batches: {d_new_labels}")
    print(f"  Labeling tasks (train+eval): {len(train_tasks)}  |  Scoring tasks: {len(eval_tasks)}")

    if not eval_tasks:
        raise ValueError(f"No scoring tasks for scenario '{scenario}' (new_tasks empty in splits).")

    # Fix the per-column metric schema from the clean init batch so every batch
    # yields a constant-length feature vector.
    column_kinds = infer_column_kinds(read_batch(pm, subtask, init_label, clean=True))
    print(f"  Feature schema: {len(column_kinds)} columns "
          f"({sum(1 for k in column_kinds.values() if k == 'numeric')} numeric)")

    def profile_of(label, clean):
        return compute_profile(read_batch(pm, subtask, label, clean=clean), column_kinds)

    good_labels, bad_labels = label_historical_batches(pm, subtask, d_train_labels, train_tasks)
    print(f"  Historical labels: {len(good_labels)} good, {len(bad_labels)} bad "
          f"(good={good_labels})")

    def skip_result(reason: str, fit_size: int) -> dict:
        print(f"  [SKIP] cannot fit detector: {reason}. F1 undefined for this split.")
        return {
            "dataset_name": dataset_name, "subtask": subtask, "status": "insufficient_fit_points",
            "detector_type": detector_type, "fit_mode": fit_mode, "fit_set_size": fit_size,
            "n_neighbors_used": 0, "good_historical_labels": good_labels,
            "bad_historical_labels": bad_labels, "init_label": init_label, "init_data": init_data,
            "d_train_labels": d_train_labels, "d_new_labels": d_new_labels,
            "scenario": scenario, "train_tasks": train_tasks, "eval_tasks": eval_tasks,
            "batch_decisions": {},
            "f1": None, "precision": None, "recall": None,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "num_cells_scored": 0, "num_cells_skipped": 0, "per_cell_results": [],
        }

    if detector_type == "supervised_knn":
        # Plain (uniform) supervised kNN over labeled historical batches.
        # init + good -> class 0 (pass), bad -> class 1 (reject). Uses BOTH classes.
        X = [profile_of(init_label, clean=(init_data == "clean"))] + \
            [profile_of(lbl, clean=False) for lbl in good_labels] + \
            [profile_of(lbl, clean=False) for lbl in bad_labels]
        y = [0] * (1 + len(good_labels)) + [1] * len(bad_labels)
        fit_desc = (f"init(label {init_label}, {init_data}) + {len(good_labels)} good (cls 0) "
                    f"+ {len(bad_labels)} bad (cls 1)")
        print(f"  Fit set: {len(X)} labeled points [{fit_desc}]")
        if len(set(y)) < 2:
            return skip_result("only one class present in labeled history", len(X))
        eff_k = max(1, min(n_neighbors, len(X)))
        if eff_k != n_neighbors:
            print(f"  [warn] capping n_neighbors {n_neighbors} -> {eff_k} ({len(X)} fit points)")
        model = SupervisedKNNClassifier(n_neighbors=eff_k).fit(X, y)
        fit_set_size = len(X)
    else:  # one_class
        if fit_mode == "good_only":
            # init batch (good by default) + historical batches with NO failing
            # train task. Safety labels ARE used here; D_new labels are never seen.
            fit_profiles = [profile_of(init_label, clean=(init_data == "clean"))] + \
                [profile_of(lbl, clean=False) for lbl in good_labels]
            fit_desc = f"init(label {init_label}, {init_data}) + {len(good_labels)} good D_train"
        elif fit_mode == "all":
            # Original Redyuk behaviour: treat every observed batch as acceptable.
            fit_profiles = [profile_of(lbl, clean=False) for lbl in d_train_labels]
            fit_desc = f"all {len(d_train_labels)} D_train (no labels used)"
        else:
            raise ValueError(f"Unknown fit_mode: {fit_mode}")
        print(f"  Fit set: {len(fit_profiles)} points [{fit_desc}]")
        # One-class kNN needs >= 2 fit points (n_neighbors < n_samples).
        if len(fit_profiles) < 2:
            return skip_result(f"only {len(fit_profiles)} good/init point(s)", len(fit_profiles))
        eff_k = max(1, min(n_neighbors, len(fit_profiles) - 1))
        if eff_k != n_neighbors:
            print(f"  [warn] capping n_neighbors {n_neighbors} -> {eff_k} "
                  f"(only {len(fit_profiles)} fit points)")
        model = KNNNoveltyDetector(contamination=contamination, n_neighbors=eff_k).fit(fit_profiles)
        fit_set_size = len(fit_profiles)

    # One decision per D_new batch (0 = pass/GOOD/safe, 1 = reject/BAD/unsafe).
    batch_decisions: dict[str, int] = {}
    for lbl in d_new_labels:
        batch_decisions[lbl] = int(model.predict([profile_of(lbl, clean=False)])[0])
    n_reject = sum(batch_decisions.values())
    print(f"  Batch decisions: {n_reject}/{len(d_new_labels)} rejected")

    # Broadcast each batch decision across all tasks and score against ground truth.
    tp = fp = tn = fn = 0
    cells = []
    skipped = 0
    for script in eval_tasks:
        for lbl in d_new_labels:
            is_safe = read_ground_truth_is_safe(pm, subtask, lbl, script)
            if is_safe is None:
                skipped += 1
                continue
            predicted_unsafe = batch_decisions[lbl] == 1
            if not is_safe and predicted_unsafe:
                tp += 1
            elif not is_safe and not predicted_unsafe:
                fn += 1
            elif is_safe and predicted_unsafe:
                fp += 1
            else:
                tn += 1
            cells.append({
                "script_name": script,
                "processed_data_label": lbl,
                "predicted_unsafe": predicted_unsafe,
                "is_safe": is_safe,
            })

    f1, precision, recall = compute_f1(tp, fp, fn)
    print(f"  Confusion: TP={tp} FP={fp} TN={tn} FN={fn} (skipped {skipped} cells w/o GT)")
    print(f"  F1={f1:.4f} precision={precision:.4f} recall={recall:.4f}"
          if math.isfinite(f1) else f"  F1=NaN (TP+FP={tp+fp}, TP+FN={tp+fn})")

    return {
        "dataset_name": dataset_name,
        "subtask": subtask,
        "status": "ok",
        "detector_type": detector_type,
        "fit_mode": fit_mode,
        "fit_set_size": fit_set_size,
        "n_neighbors_used": eff_k,
        "good_historical_labels": good_labels,
        "bad_historical_labels": bad_labels,
        "init_label": init_label,
        "init_data": init_data,
        "d_train_labels": d_train_labels,
        "d_new_labels": d_new_labels,
        "scenario": scenario,
        "train_tasks": train_tasks,
        "eval_tasks": eval_tasks,
        "batch_decisions": batch_decisions,
        "f1": _json_num(f1),
        "precision": _json_num(precision),
        "recall": _json_num(recall),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "num_cells_scored": len(cells),
        "num_cells_skipped": skipped,
        "per_cell_results": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_name", type=str, default="hr_analytics")
    parser.add_argument("--subtask", type=str, default="general_task")
    parser.add_argument("--split_strategy_from", type=str, default=None,
                        help="Reference run dir (e.g. run_20260111_231103) to copy splits from.")
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help="Seed for split generation (only when --split_strategy_from is unset).")
    parser.add_argument("--label_split_ratio", type=float, default=DEFAULT_LABEL_SPLIT_RATIO)
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--detector", choices=["one_class", "supervised_knn"], default="one_class",
                        help="one_class: Redyuk novelty detector (uses only good/normal data). "
                             "supervised_knn: plain labeled kNN classifier using BOTH good and bad "
                             "historical batches (NOT the Redyuk method).")
    parser.add_argument("--fit_mode", choices=["good_only", "all"], default="good_only",
                        help="(one_class only) good_only: fit on init + historical batches with no "
                             "failing train task. all: treat every observed batch as good.")
    parser.add_argument("--init_label", type=str, default="0",
                        help="Processed-data label of the init batch (good by default).")
    parser.add_argument("--init_data", choices=["clean", "corrupted"], default="clean",
                        help="Which new_data variant of the init batch to use as the good reference.")
    parser.add_argument("--scenario", choices=["new_data", "new_data_new_task"], default="new_data",
                        help="new_data: score per-batch decisions on train+eval tasks "
                             "(test_1_cross_new_data). new_data_new_task: score them on the held-out "
                             "new tasks (test_3_new_script_new_data). Historical labels always come "
                             "from the train tasks.")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    project_root = get_project_root()

    if args.split_strategy_from:
        splits = load_splits_from_reference_run(args.split_strategy_from, project_root)
        print(f"Loaded splits from reference run: {args.split_strategy_from}")
    else:
        pm = ProjectManager(project_root=project_root, dataset_name=args.dataset_name,
                            downstream_task_type="general")
        splits = generate_splits(pm, args.subtask, args.random_seed, args.label_split_ratio)
        print(f"Generated splits (seed={args.random_seed}, label_split_ratio={args.label_split_ratio})")

    result = run_dataset(
        dataset_name=args.dataset_name,
        subtask=args.subtask,
        splits=splits,
        contamination=args.contamination,
        n_neighbors=args.n_neighbors,
        project_root=project_root,
        fit_mode=args.fit_mode,
        init_label=args.init_label,
        init_data=args.init_data,
        detector_type=args.detector,
        scenario=args.scenario,
    )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        project_root / "optimization_runs" / f"baseline_redyuk_run_{run_timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_timestamp": run_timestamp,
        "baseline_method": ("Redyuk-EDBT2021-KNN-novelty" if args.detector == "one_class"
                            else "supervised-kNN (plain, uses good+bad labels)"),
        "reference": "Redyuk, Kaoudi, Markl, Schelter. Automating Data Quality Validation "
                     "for Dynamic Data Ingestion. EDBT 2021.",
        "split_strategy_from": args.split_strategy_from,
        "dataset_name": args.dataset_name,
        "subtask": args.subtask,
        "detector": args.detector,
        "contamination": args.contamination,
        "n_neighbors": args.n_neighbors,
        "n_neighbors_used": result["n_neighbors_used"],
        "fit_mode": args.fit_mode,
        "fit_set_size": result["fit_set_size"],
        "init_label": args.init_label,
        "init_data": args.init_data,
        "scenario": args.scenario,
        "result": result,
    }

    # Populate the matching scenario block; the detector is task-agnostic, so
    # New Tasks (test_2) is never meaningful (it would reuse train-data labels).
    scenario_block = {
        "description": ("Train+eval scripts + new data labels (broadcast batch decisions)"
                        if args.scenario == "new_data"
                        else "New scripts + new data labels (broadcast batch decisions)"),
        "f1": result["f1"], "precision": result["precision"], "recall": result["recall"],
        "tp": result["tp"], "fp": result["fp"], "tn": result["tn"], "fn": result["fn"],
        "num_cells_scored": result["num_cells_scored"],
        "num_cells_skipped": result["num_cells_skipped"],
    }
    na = "N/A (task-agnostic detector cannot condition on new tasks)"
    summary["test_1_cross_new_data"] = scenario_block if args.scenario == "new_data" else na
    summary["test_2_new_script_train_data"] = na
    summary["test_3_new_script_new_data"] = (
        scenario_block if args.scenario == "new_data_new_task" else na)

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to: {summary_file}")


if __name__ == "__main__":
    main()
