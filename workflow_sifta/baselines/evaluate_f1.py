from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import dspy

from prismadv.utils import (
    get_project_root,
    load_dotenv,
    suppress_py4j_logging,
)
from workflow_sifta.baselines.gepa.dataset_preparation import prepare_script_level_dataset
from workflow_sifta.baselines.gepa.script_level_module import ScriptLevelConstraintGenerationModule
from workflow_sifta.optimization_fns import OptimizationWorkflow

suppress_py4j_logging()
load_dotenv()

# Configure DSPy LLM
llm_name = "gpt-4.1-mini"
lm = dspy.LM(f"openai/{llm_name}", temperature=1, max_tokens=32000)
dspy.configure(lm=lm)


def load_instructions_from_log(log_dir: Path, model_name: str = "final") -> dict:
    if model_name in ("initial", "round_-1_initial", "round_0_initial"):
        # Standard location for initial instructions: round_-1_initial.json
        initial_file = log_dir / "round_-1_initial.json"

        # Fallback for old naming convention
        if not initial_file.exists():
            initial_file = log_dir / "round_0_initial.json"

        # Fallback for GEPA/SIFTA old convention (initial in round_1.json)
        if not initial_file.exists():
            initial_file = log_dir / "round_1.json"

        if initial_file.exists():
            with open(initial_file, "r") as f:
                data = json.load(f)
            return data.get("instructions", {})
        else:
            raise FileNotFoundError(
                f"Could not find initial program file in {log_dir}"
            )

    elif model_name in ("final", "final_program"):
        file_path = log_dir / "final_program.json"
        with open(file_path, "r") as f:
            data = json.load(f)
        return data.get("instructions", {})

    elif model_name.startswith("round_"):
        round_num = model_name.replace("round_", "")
        file_path = log_dir / f"round_{round_num}.json"
        with open(file_path, "r") as f:
            data = json.load(f)
        # Use optimized_instructions if available, otherwise instructions
        return data.get("optimized_instructions", data.get("instructions", {}))

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def apply_instructions_to_module(
        module: ScriptLevelConstraintGenerationModule, instructions: dict
):
    for pred_name, pred in module.named_predictors():
        if pred_name in instructions:
            pred.signature = pred.signature.with_instructions(instructions[pred_name])


def _compute_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) > 0:
        return 2 * precision * recall / (precision + recall)
    return float("nan")


def evaluate_f1_on_test_set(
        workflow: OptimizationWorkflow,
        module: ScriptLevelConstraintGenerationModule,
        test_set: list,
        model_label: str = "model",
) -> dict:
    print(f"\nEvaluating {model_label} on test set ({len(test_set)} examples)...")

    total_tp = total_fp = total_tn = total_fn = 0
    per_script_f1_scores = []

    for i, example in enumerate(test_set):
        print(f"  Processing {i + 1}/{len(test_set)}: {example['script_name']}...", end="")

        try:
            # Run the module to get predictions
            pred = module(
                code_script=example["code_script"],
                columns_desc=example["columns_desc"],
                downstream_task_description=example["downstream_task_description"],
            )

            dataset_name = example["dataset_name"]
            subtask_name = example["subtask_name"]
            new_data_safety = example.get("new_data_safety", {})

            if not new_data_safety:
                print(" skipped (no safety labels)")
                continue

            # Validate on training data first
            constraints_with_sources = workflow.validate_constraints_on_training_data(
                dataset_name=dataset_name,
                subtask_name=subtask_name,
                constraints_with_sources=pred,
            )

            # Compute F1 for this script
            script_tp = script_fp = script_tn = script_fn = 0
            for processed_data_label, is_safe in new_data_safety.items():
                validation_results = workflow.validate_constraints_on_test_data(
                    dataset_name=dataset_name,
                    subtask_name=subtask_name,
                    constraints_with_sources=constraints_with_sources,
                    processed_data_label=processed_data_label,
                    clean=False,
                )
                num_failed_error = validation_results.check_result()[2]
                predicted_safe = num_failed_error == 0

                # Positive class = "unsafe/error" (standard for anomaly detection)
                # TP = correctly detected error, TN = correctly accepted safe data
                # FP = false alarm, FN = missed error (BAD!)
                if not is_safe and not predicted_safe:
                    script_tp += 1  # TP: error detected correctly
                elif not is_safe and predicted_safe:
                    script_fn += 1  # FN: error missed (BAD!)
                elif is_safe and not predicted_safe:
                    script_fp += 1  # FP: false alarm
                else:  # is_safe and predicted_safe
                    script_tn += 1  # TN: safe data accepted correctly

            total_tp += script_tp
            total_fp += script_fp
            total_tn += script_tn
            total_fn += script_fn

            script_f1 = _compute_f1(script_tp, script_fp, script_fn)
            per_script_f1_scores.append(script_f1)
            print(f" F1={script_f1:.4f}" if math.isfinite(script_f1) else " F1=NaN")

        except Exception as e:
            print(f" error: {e}")
            continue

    # Compute overall metrics
    overall_f1 = _compute_f1(total_tp, total_fp, total_fn)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float("nan")

    # Compute average per-script F1 (excluding NaN)
    valid_f1_scores = [f for f in per_script_f1_scores if math.isfinite(f)]
    avg_per_script_f1 = sum(valid_f1_scores) / len(valid_f1_scores) if valid_f1_scores else float("nan")

    results = {
        "f1": overall_f1,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "tn": total_tn,
        "fn": total_fn,
        "avg_per_script_f1": avg_per_script_f1,
        "num_scripts_evaluated": len(per_script_f1_scores),
        "per_script_f1_scores": per_script_f1_scores,
    }

    print(f"\n{model_label} F1 Results:")
    print(f"  Overall F1: {overall_f1:.4f}" if math.isfinite(overall_f1) else "  Overall F1: NaN")
    print(f"  Precision: {precision:.4f}" if math.isfinite(precision) else "  Precision: NaN")
    print(f"  Recall: {recall:.4f}" if math.isfinite(recall) else "  Recall: NaN")
    print(f"  Confusion Matrix: TP={total_tp}, FP={total_fp}, TN={total_tn}, FN={total_fn}")
    print(f"  Avg Per-Script F1: {avg_per_script_f1:.4f}" if math.isfinite(
        avg_per_script_f1) else "  Avg Per-Script F1: NaN")
    print(f"  Scripts Evaluated: {len(per_script_f1_scores)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline optimized programs using F1 score"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to optimization run directory (e.g., optimization_runs/baseline_run_20231229)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="final",
        help="Which model to evaluate: 'initial', 'final', or 'round_N' (default: final)",
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Also evaluate the initial baseline model for comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: same as run_dir)",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = get_project_root()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading from: {run_dir}")
    print(f"Saving results to: {output_dir}")
    print(f"Using LLM: {llm_name}")

    summary = {}
    # Load summary to get configuration
    summary_file = run_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        print(f"Run timestamp: {summary.get('run_timestamp', 'unknown')}")
        print(f"Original LLM: {summary.get('llm_name', 'unknown')}")
        print(f"Train set size: {summary.get('train_set_size', 'unknown')}")
        print(f"Val set size: {summary.get('val_set_size', 'unknown')}")

    # Get configuration from summary
    dataset_subtasks = summary.get("dataset_subtasks")
    test_script_name_list = summary.get("test_script_name_list")
    processed_data_label_list = summary.get("processed_data_label_list")

    if not dataset_subtasks or not test_script_name_list or not processed_data_label_list:
        raise ValueError(
            "Summary file missing required fields: dataset_subtasks, "
            "test_script_name_list, or processed_data_label_list"
        )

    print(f"\nTest scripts: {test_script_name_list}")
    print(f"Processed data labels: {processed_data_label_list}")

    # Initialize workflow
    print("\nInitializing workflow...")
    workflow = OptimizationWorkflow(
        train_dataset_subtasks=dataset_subtasks,
        train_processed_data_label_list=processed_data_label_list,
        test_dataset_subtasks=dataset_subtasks,
        test_processed_data_label_list=processed_data_label_list,
        project_root=project_root,
    )

    # Prepare test dataset
    print("\nPreparing test dataset...")
    test_set = prepare_script_level_dataset(
        dataset_subtasks=dataset_subtasks,
        script_name_list=test_script_name_list,
        processed_data_label="0",
        new_processed_data_label_list=processed_data_label_list,
        project_root=project_root,
    )
    print(f"Test set size: {len(test_set)}")

    results_data = {
        "run_dir": str(run_dir),
        "evaluation_timestamp": datetime.now().isoformat(),
        "test_scripts": test_script_name_list,
        "processed_data_labels": processed_data_label_list,
        "llm_name": llm_name,
    }

    # Evaluate baseline if requested
    if args.compare_baseline:
        print(f"\n{'=' * 60}")
        print("=== Baseline Model Evaluation ===")
        print(f"{'=' * 60}")

        baseline_instructions = load_instructions_from_log(run_dir, "initial")
        baseline_module = ScriptLevelConstraintGenerationModule()
        apply_instructions_to_module(baseline_module, baseline_instructions)

        baseline_results = evaluate_f1_on_test_set(
            workflow=workflow,
            module=baseline_module,
            test_set=test_set,
            model_label="Initial (baseline)",
        )
        results_data["baseline"] = {
            "f1": float(baseline_results["f1"]) if math.isfinite(baseline_results["f1"]) else None,
            "precision": float(baseline_results["precision"]) if math.isfinite(baseline_results["precision"]) else None,
            "recall": float(baseline_results["recall"]) if math.isfinite(baseline_results["recall"]) else None,
            "tp": baseline_results["tp"],
            "fp": baseline_results["fp"],
            "tn": baseline_results["tn"],
            "fn": baseline_results["fn"],
            "avg_per_script_f1": float(baseline_results["avg_per_script_f1"]) if math.isfinite(
                baseline_results["avg_per_script_f1"]) else None,
            "num_scripts_evaluated": baseline_results["num_scripts_evaluated"],
        }

    # Evaluate the specified model
    print(f"\n{'=' * 60}")
    print(f"=== {args.model.title()} Model Evaluation ===")
    print(f"{'=' * 60}")

    model_instructions = load_instructions_from_log(run_dir, args.model)
    print(f"\nLoaded instructions for predictors: {list(model_instructions.keys())}")

    model_module = ScriptLevelConstraintGenerationModule()
    apply_instructions_to_module(model_module, model_instructions)

    model_results = evaluate_f1_on_test_set(
        workflow=workflow,
        module=model_module,
        test_set=test_set,
        model_label=args.model.title(),
    )
    results_data[args.model] = {
        "f1": float(model_results["f1"]) if math.isfinite(model_results["f1"]) else None,
        "precision": float(model_results["precision"]) if math.isfinite(model_results["precision"]) else None,
        "recall": float(model_results["recall"]) if math.isfinite(model_results["recall"]) else None,
        "tp": model_results["tp"],
        "fp": model_results["fp"],
        "tn": model_results["tn"],
        "fn": model_results["fn"],
        "avg_per_script_f1": float(model_results["avg_per_script_f1"]) if math.isfinite(
            model_results["avg_per_script_f1"]) else None,
        "num_scripts_evaluated": model_results["num_scripts_evaluated"],
    }

    # Calculate improvement if we have baseline
    if args.compare_baseline and "baseline" in results_data:
        baseline_f1 = results_data["baseline"]["f1"]
        model_f1 = results_data[args.model]["f1"]

        if baseline_f1 is not None and model_f1 is not None:
            improvement = model_f1 - baseline_f1
            results_data["f1_improvement"] = improvement
            print(f"\n{'=' * 60}")
            print(f"F1 Improvement: {improvement:+.4f} ({baseline_f1:.4f} -> {model_f1:.4f})")
            print(f"{'=' * 60}")

    # Save results
    results_file = (
            output_dir
            / f"f1_evaluation_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved evaluation results to: {results_file}")

    # Cleanup
    workflow.cleanup_spark_sessions()
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
