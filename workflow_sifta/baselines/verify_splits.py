"""
Verify that dataset splits match exactly between runs.

Usage:
    python workflow_sifta/baselines/verify_splits.py \
        --reference_run run_20260110_234647 \
        --baseline_run baseline_gepa_run_20260112_XXXXXX
"""

import argparse
import json
from pathlib import Path

from prismadv.utils import get_project_root


def load_run_summary(run_name: str) -> dict:
    """Load summary.json from a run directory."""
    summary_path = Path(get_project_root()) / "optimization_runs" / run_name / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    with open(summary_path, 'r') as f:
        return json.load(f)


def compare_splits(reference_summary: dict, baseline_summary: dict) -> dict:
    """Compare splits between two runs."""
    results = {
        "train_eval_labels_match": False,
        "cross_new_data_labels_match": False,
        "train_eval_scripts_match": False,
        "train_scripts_match": False,
        "eval_scripts_match": False,
        "new_scripts_match": False,
        "all_match": False,
    }

    # Compare label splits
    ref_train_eval_labels = reference_summary.get("train_eval_label_list", [])
    base_train_eval_labels = baseline_summary.get("train_eval_label_list", [])
    results["train_eval_labels_match"] = ref_train_eval_labels == base_train_eval_labels

    ref_cross_labels = reference_summary.get("cross_new_data_test_label_list", [])
    base_cross_labels = baseline_summary.get("cross_new_data_test_label_list", [])
    results["cross_new_data_labels_match"] = ref_cross_labels == base_cross_labels

    # Compare script splits
    ref_train_eval_scripts = reference_summary.get("train_eval_script_name_list", [])
    base_train_eval_scripts = baseline_summary.get("train_eval_script_name_list", [])
    results["train_eval_scripts_match"] = ref_train_eval_scripts == base_train_eval_scripts

    ref_train_scripts = reference_summary.get("train_script_name_list", [])
    base_train_scripts = baseline_summary.get("train_script_name_list", [])
    results["train_scripts_match"] = ref_train_scripts == base_train_scripts

    ref_eval_scripts = reference_summary.get("eval_script_name_list", [])
    base_eval_scripts = baseline_summary.get("eval_script_name_list", [])
    results["eval_scripts_match"] = ref_eval_scripts == base_eval_scripts

    ref_new_scripts = reference_summary.get("new_script_name_list", [])
    base_new_scripts = baseline_summary.get("new_script_name_list", [])
    results["new_scripts_match"] = ref_new_scripts == base_new_scripts

    # Overall check
    results["all_match"] = all([
        results["train_eval_labels_match"],
        results["cross_new_data_labels_match"],
        results["train_eval_scripts_match"],
        results["train_scripts_match"],
        results["eval_scripts_match"],
        results["new_scripts_match"],
    ])

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify dataset splits match between runs")
    parser.add_argument(
        "--reference_run",
        type=str,
        required=True,
        help="Reference run directory name (e.g., run_20260110_234647)"
    )
    parser.add_argument(
        "--baseline_run",
        type=str,
        required=True,
        help="Baseline run directory name (e.g., baseline_gepa_run_20260112_XXXXXX)"
    )

    args = parser.parse_args()

    print(f"{'=' * 80}")
    print(f"Verifying Dataset Splits")
    print(f"{'=' * 80}")
    print(f"Reference run: {args.reference_run}")
    print(f"Baseline run:  {args.baseline_run}")
    print(f"{'=' * 80}\n")

    # Load summaries
    try:
        reference_summary = load_run_summary(args.reference_run)
        baseline_summary = load_run_summary(args.baseline_run)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Check if baseline used split_strategy_from
    split_strategy_from = baseline_summary.get("split_strategy_from")
    if split_strategy_from:
        print(f"Baseline used split_strategy_from: {split_strategy_from}")
        if split_strategy_from != args.reference_run:
            print(f"WARNING: Baseline copied splits from '{split_strategy_from}', not '{args.reference_run}'")
    else:
        print(f"Baseline generated new splits (did not use split_strategy_from)")

    print()

    # Compare splits
    results = compare_splits(reference_summary, baseline_summary)

    # Print results
    print("Split Comparison Results:")
    print(f"  Train/Eval Labels:     {'MATCH' if results['train_eval_labels_match'] else 'DIFFER'}")
    print(f"  Cross-New-Data Labels: {'MATCH' if results['cross_new_data_labels_match'] else 'DIFFER'}")
    print(f"  Train/Eval Scripts:    {'MATCH' if results['train_eval_scripts_match'] else 'DIFFER'}")
    print(f"  Train Scripts:         {'MATCH' if results['train_scripts_match'] else 'DIFFER'}")
    print(f"  Eval Scripts:          {'MATCH' if results['eval_scripts_match'] else 'DIFFER'}")
    print(f"  New Scripts:           {'MATCH' if results['new_scripts_match'] else 'DIFFER'}")
    print()

    if results["all_match"]:
        print("SUCCESS: All splits match exactly!")
        return 0
    else:
        print("FAILURE: Splits do not match!")
        print("\nTo fix this, ensure the baseline uses SPLIT_STRATEGY_FROM pointing to the reference run.")
        return 1


if __name__ == "__main__":
    exit(main())
