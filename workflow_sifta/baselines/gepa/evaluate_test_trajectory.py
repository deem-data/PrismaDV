"""
Evaluate GEPA optimization trajectory on test sets.

This script evaluates the "current best" prompt at each iteration of the GEPA optimization
on test configurations. This allows plotting test performance alongside the optimization
trajectory to analyze generalization.

Usage:
    # Single test
    python workflow_sifta/baselines/gepa/evaluate_test_trajectory.py \
        --run_dir optimization_runs/baseline_gepa_run_20260112_120000 \
        --test_type test_1_cross_new_data

    # All three tests
    python workflow_sifta/baselines/gepa/evaluate_test_trajectory.py \
        --run_dir optimization_runs/baseline_gepa_run_20260112_120000 \
        --test_type all
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import dspy
import numpy as np
from tqdm import tqdm

from prismadv.utils import get_project_root, load_dotenv
from workflow_sifta.baselines.gepa.script_level_module import ScriptLevelConstraintGenerationModule
from workflow_sifta.baselines.evaluate_f1 import evaluate_f1_on_test_set
from workflow_sifta.optimization_fns import OptimizationWorkflow

load_dotenv()


def load_round_data(run_dir: Path) -> list[dict]:
    """Load all round data from a run directory."""
    # Load initial program first (if exists)
    rounds = []
    initial_file = run_dir / "round_-1_initial.json"
    if initial_file.exists():
        with open(initial_file, 'r') as f:
            rounds.append(json.load(f))

    # Load optimization rounds (skip round_-1 since we already loaded it)
    round_files = sorted([f for f in run_dir.glob("round_*.json") if f.name != "round_-1_initial.json"])
    for round_file in round_files:
        with open(round_file, 'r') as f:
            rounds.append(json.load(f))

    # Load final program if it exists
    final_file = run_dir / "final_program.json"
    if final_file.exists():
        with open(final_file, 'r') as f:
            rounds.append(json.load(f))

    return rounds


def load_summary(run_dir: Path) -> dict:
    """Load summary.json from run directory."""
    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    with open(summary_file, 'r') as f:
        return json.load(f)


def get_test_configuration(summary: dict, test_type: str) -> dict:
    """
    Extract test configuration from summary.

    Returns dict with:
        - script_list: list of script names
        - label_list: list of data labels
        - description: human-readable description
    """
    valid_types = ["test_1_cross_new_data", "test_2_new_script_train_data", "test_3_new_script_new_data"]

    if test_type not in valid_types:
        raise ValueError(f"Invalid test_type: {test_type}. Must be one of {valid_types}")

    # Extract configuration from the test set configurations in summary
    test_config = summary.get(test_type)
    if test_config:
        return {
            "script_list": test_config.get("script_list", []),
            "label_list": test_config.get("label_list", []),
            "description": test_config.get("description", ""),
            "dataset_name": summary.get("train_dataset_name"),
            "dataset_subtasks": summary.get("dataset_subtasks", {}),
        }
    else:
        raise ValueError(f"Test configuration '{test_type}' not found in summary")


def extract_best_model_changes(rounds: list[dict]) -> list[dict]:
    """
    Extract only the iterations where the best model changed across all rounds.

    Returns list of dicts with:
        - round: round number
        - global_iteration: global iteration number (continuous across rounds)
        - local_iteration: iteration within the round
        - instructions: prompt instructions
        - eval_score: validation score (from optimization)
        - is_improvement: whether this is an improvement over previous best
    """
    changes = []
    global_iteration_offset = 0
    global_best_score = -float('inf')

    for round_data in rounds:
        round_num = round_data.get("round", -1)

        # Handle initial model first (round -1) - no GEPA details expected
        if round_num == -1:
            # Initial program before optimization
            changes.append({
                "round": round_num,
                "global_iteration": -1,
                "local_iteration": -1,
                "instructions": round_data.get("instructions", {}),
                "eval_score": round_data.get("score", 0.0),
                "is_improvement": True,  # First model
            })
            global_best_score = round_data.get("score", 0.0)
            continue  # Skip rest of processing for initial round

        # Skip final round here - will be added at the end
        if round_num == "final":
            continue

        # For optimization rounds, we need GEPA details
        gepa_details = round_data.get("gepa_details", {})

        if not gepa_details or "history" not in gepa_details:
            print(f"Warning: No GEPA details for round {round_num}, skipping")
            continue

        history = gepa_details["history"]

        # Track through iterations in this round
        current_best_score = round_data.get("score", 0.0)

        for local_iteration, entry in enumerate(history):
            score = entry.get("score")
            if score is None:
                continue

            # Check if this is a new best
            if score > current_best_score:
                is_global_improvement = score > global_best_score

                changes.append({
                    "round": round_num,
                    "global_iteration": local_iteration + global_iteration_offset,
                    "local_iteration": local_iteration,
                    "instructions": entry.get("instructions", {}),
                    "eval_score": score,
                    "is_improvement": is_global_improvement,
                })

                current_best_score = score
                if is_global_improvement:
                    global_best_score = score

        # Update offset for next round
        global_iteration_offset += len(history) + 2

    # Add final program if it exists (round == "final")
    final_round = next((r for r in rounds if r.get("round") == "final"), None)
    if final_round:
        final_score = final_round.get("score", 0.0)
        is_global_improvement = final_score > global_best_score

        # Use optimized_instructions if available, else regular instructions
        instructions = final_round.get("optimized_instructions") or final_round.get("instructions", {})

        changes.append({
            "round": "final",
            "global_iteration": global_iteration_offset,
            "local_iteration": -1,  # Not applicable for final program
            "instructions": instructions,
            "eval_score": final_score,
            "is_improvement": is_global_improvement,
        })

    return changes


def apply_instructions_to_module(module: ScriptLevelConstraintGenerationModule, instructions: dict):
    """Apply instructions to module's predictors.

    Note: Uses with_instructions() to avoid modifying class-level signature.
    """
    for pred_name, pred in module.named_predictors():
        if pred_name in instructions:
            pred.signature = pred.signature.with_instructions(instructions[pred_name])


def prepare_test_set(
    script_list: list[str],
    label_list: list[str],
    dataset_name: str,
    dataset_subtasks: dict,
    project_root: Path,
) -> list[dict]:
    """Prepare test set for evaluation."""
    from workflow_sifta.baselines.gepa.dataset_preparation import prepare_script_level_dataset

    return prepare_script_level_dataset(
        dataset_subtasks={dataset_name: list(dataset_subtasks.get(dataset_name, []))},
        script_name_list=script_list,
        processed_data_label="0",
        new_processed_data_label_list=label_list,
        project_root=project_root,
    )


def evaluate_single_test_type(
    run_dir: Path,
    test_type: str,
    llm_name: str,
    sample_interval: int,
    force: bool,
    project_root: Path,
) -> dict:
    """Evaluate a single test type and return results."""

    # Check for existing results
    existing_pattern = f"test_trajectory_{test_type}_*.json"
    existing_files = list(run_dir.glob(existing_pattern))
    if existing_files and not force:
        existing_file = sorted(existing_files)[-1]
        print(f"  Found existing results: {existing_file.name}")
        print(f"  Skipping (use --force to re-evaluate)")
        with open(existing_file, 'r') as f:
            return json.load(f)

    print(f"\n{'=' * 80}")
    print(f"Evaluating: {test_type}")
    print(f"{'=' * 80}")

    # Load summary and test configuration
    summary = load_summary(run_dir)
    test_config = get_test_configuration(summary, test_type)

    print(f"Test Configuration: {test_config['description']}")
    print(f"  Scripts ({len(test_config['script_list'])}): {test_config['script_list']}")
    print(f"  Labels ({len(test_config['label_list'])}): {test_config['label_list']}")

    # Initialize workflow with test configuration
    dataset_name = test_config['dataset_name']
    dataset_subtasks = test_config['dataset_subtasks']

    workflow = OptimizationWorkflow(
        train_dataset_subtasks=dataset_subtasks,
        train_processed_data_label_list=test_config['label_list'],
        project_root=project_root,
    )

    # Prepare test set
    print("Preparing test set...")
    test_set = prepare_test_set(
        script_list=test_config['script_list'],
        label_list=test_config['label_list'],
        dataset_name=dataset_name,
        dataset_subtasks=dataset_subtasks,
        project_root=project_root,
    )
    print(f"Test set size: {len(test_set)} examples")

    # Load optimization trajectory
    rounds = load_round_data(run_dir)
    trajectory = extract_best_model_changes(rounds)

    if not trajectory:
        print("Warning: No trajectory points found from GEPA details!")
        print("This may happen if GEPA details weren't logged properly.")
        print("Falling back to initial/final evaluation only (like MIPROv2)...")

        # Fallback: evaluate initial and final programs only
        results = []

        # Find initial program
        initial_round = next((r for r in rounds if r.get("round") == -1), None)
        if initial_round:
            print("Evaluating initial program...")
            initial_module = ScriptLevelConstraintGenerationModule()
            apply_instructions_to_module(initial_module, initial_round.get("instructions", {}))

            initial_test_results = evaluate_f1_on_test_set(
                workflow=workflow,
                module=initial_module,
                test_set=test_set,
                model_label="initial",
            )

            results.append({
                "stage": "initial",
                "round": -1,
                "global_iteration": -1,
                "eval_score": initial_round.get("score", 0.0),
                "test_f1": initial_test_results["f1"],
                "test_precision": initial_test_results["precision"],
                "test_recall": initial_test_results["recall"],
                "test_tp": initial_test_results["tp"],
                "test_fp": initial_test_results["fp"],
                "test_tn": initial_test_results["tn"],
                "test_fn": initial_test_results["fn"],
                "test_avg_per_script_f1": initial_test_results["avg_per_script_f1"],
                "test_num_scripts_evaluated": initial_test_results["num_scripts_evaluated"],
            })

        # Find final/optimized program (prioritize "final" round, then max round number)
        final_round = None
        for r in rounds:
            if r.get("round") == "final":
                final_round = r
                break
        if not final_round:
            # Fall back to highest round number
            numeric_rounds = [r for r in rounds if isinstance(r.get("round"), int) and r.get("round", -1) >= 0]
            if numeric_rounds:
                final_round = max(numeric_rounds, key=lambda r: r.get("round", -1))

        if final_round:
            print("Evaluating final program...")
            final_module = ScriptLevelConstraintGenerationModule()
            # Use optimized instructions if available, else regular instructions
            instructions = final_round.get("optimized_instructions") or final_round.get("instructions", {})
            apply_instructions_to_module(final_module, instructions)

            final_test_results = evaluate_f1_on_test_set(
                workflow=workflow,
                module=final_module,
                test_set=test_set,
                model_label="final",
            )

            results.append({
                "stage": "final",
                "round": final_round.get("round", 0),
                "global_iteration": 0,  # No trajectory info available
                "eval_score": final_round.get("score", 0.0),
                "test_f1": final_test_results["f1"],
                "test_precision": final_test_results["precision"],
                "test_recall": final_test_results["recall"],
                "test_tp": final_test_results["tp"],
                "test_fp": final_test_results["fp"],
                "test_tn": final_test_results["tn"],
                "test_fn": final_test_results["fn"],
                "test_avg_per_script_f1": final_test_results["avg_per_script_f1"],
                "test_num_scripts_evaluated": final_test_results["num_scripts_evaluated"],
            })

        # Save fallback results
        output_data = {
            "run_dir": str(run_dir),
            "test_type": test_type,
            "test_config": test_config,
            "evaluation_timestamp": datetime.now().isoformat(),
            "llm_name": llm_name,
            "sample_interval": sample_interval,
            "trajectory": results,
            "warning": "No GEPA trajectory details found - evaluated initial/final only",
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = run_dir / f"test_trajectory_{test_type}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {output_file.name}")
        if len(results) == 2:
            print(f"Test F1 Summary for {test_type}:")
            print(f"  Initial: {results[0]['test_f1']:.4f}")
            print(f"  Final:   {results[1]['test_f1']:.4f}")
            if not np.isnan(results[0]['test_f1']) and not np.isnan(results[1]['test_f1']):
                improvement = results[1]['test_f1'] - results[0]['test_f1']
                print(f"  Improvement: {improvement:+.4f}")

        workflow.cleanup_spark_sessions()
        return output_data

    # Filter by sample interval if specified
    if sample_interval > 1:
        trajectory = trajectory[::sample_interval]
        print(f"Sampled points: {len(trajectory)} (every {sample_interval} improvements)")

    # Evaluate each point in trajectory
    print(f"Evaluating {len(trajectory)} prompts on test set...")

    results = []
    for i, point in enumerate(tqdm(trajectory, desc=f"Evaluating {test_type}")):
        # Create module and apply instructions
        module = ScriptLevelConstraintGenerationModule()
        apply_instructions_to_module(module, point["instructions"])

        # Evaluate on test set
        test_results = evaluate_f1_on_test_set(
            workflow=workflow,
            module=module,
            test_set=test_set,
            model_label=f"iter_{point['global_iteration']}",
        )

        # Combine with trajectory info
        result_point = {
            # Trajectory metadata
            "round": point["round"],
            "global_iteration": point["global_iteration"],
            "local_iteration": point["local_iteration"],
            "is_improvement": point.get("is_improvement", False),

            # Eval score (from optimization)
            "eval_score": point["eval_score"],

            # Test F1 metrics
            "test_f1": test_results["f1"],
            "test_precision": test_results["precision"],
            "test_recall": test_results["recall"],

            # Full confusion matrix
            "test_tp": test_results["tp"],
            "test_fp": test_results["fp"],
            "test_tn": test_results["tn"],
            "test_fn": test_results["fn"],

            # Additional info
            "test_avg_per_script_f1": test_results["avg_per_script_f1"],
            "test_num_scripts_evaluated": test_results["num_scripts_evaluated"],
        }

        results.append(result_point)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = run_dir / f"test_trajectory_{test_type}_{timestamp}.json"

    output_data = {
        "run_dir": str(run_dir),
        "test_type": test_type,
        "test_config": test_config,
        "evaluation_timestamp": datetime.now().isoformat(),
        "llm_name": llm_name,
        "sample_interval": sample_interval,
        "trajectory": results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file.name}")

    # Print summary statistics
    f1_scores = [r["test_f1"] for r in results if r["test_f1"] is not None and not np.isnan(r["test_f1"])]

    if f1_scores:
        print(f"Test F1 Summary for {test_type}:")
        print(f"  First: {f1_scores[0]:.4f}")
        print(f"  Last:  {f1_scores[-1]:.4f}")
        print(f"  Improvement: {f1_scores[-1] - f1_scores[0]:+.4f}")

    # Cleanup
    workflow.cleanup_spark_sessions()

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate GEPA optimization trajectory on test sets")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to optimization run directory"
    )
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=["test_1_cross_new_data", "test_2_new_script_train_data", "test_3_new_script_new_data", "all"],
        help="Which test configuration to use ('all' runs all three tests)"
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="gpt-4.1-mini",
        help="LLM to use for constraint generation (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Evaluate every N model improvements (default: 1 = all improvements)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if results already exist"
    )

    args = parser.parse_args()

    # Setup paths
    project_root = get_project_root()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"{'=' * 80}")
    print(f"GEPA Test Trajectory Evaluation")
    print(f"{'=' * 80}")
    print(f"Run directory: {run_dir}")
    print(f"Test type: {args.test_type}")
    print(f"LLM: {args.llm_name}")
    print(f"Sample interval: every {args.sample_interval} iteration(s)")
    print(f"{'=' * 80}\n")

    # Configure DSPy
    lm = dspy.LM(f"openai/{args.llm_name}", temperature=1, max_tokens=32000)
    dspy.configure(lm=lm)

    # Determine which test types to run
    if args.test_type == "all":
        test_types = ["test_1_cross_new_data", "test_2_new_script_train_data", "test_3_new_script_new_data"]
        print("Running all three test configurations...\n")
    else:
        test_types = [args.test_type]

    # Evaluate each test type
    all_results = {}
    for test_type in test_types:
        result_data = evaluate_single_test_type(
            run_dir=run_dir,
            test_type=test_type,
            llm_name=args.llm_name,
            sample_interval=args.sample_interval,
            force=args.force,
            project_root=project_root,
        )
        all_results[test_type] = result_data

    # Print final summary
    print(f"\n{'=' * 80}")
    print(f"All Evaluations Complete!")
    print(f"{'=' * 80}")

    if len(test_types) > 1:
        print(f"\nSummary across all test types:")
        for test_type in test_types:
            results = all_results[test_type]["trajectory"]
            f1_scores = [r["test_f1"] for r in results if r["test_f1"] is not None and not np.isnan(r["test_f1"])]
            if f1_scores:
                print(f"\n{test_type}:")
                print(f"  Initial F1: {f1_scores[0]:.4f}")
                print(f"  Final F1:   {f1_scores[-1]:.4f}")
                print(f"  Improvement: {f1_scores[-1] - f1_scores[0]:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
