"""
Evaluate optimization trajectory on test sets.

This script evaluates the "current best" prompt at each iteration of the optimization
on one of the three test configurations. This allows plotting test performance alongside
the optimization trajectory to analyze generalization.

The three test types are:
1. test_1_cross_new_data: Train scripts + new data labels
2. test_2_new_script_train_data: New scripts + train data labels
3. test_3_new_script_new_data: New scripts + new data labels

Caching:
- If a test trajectory file for the same test_type already exists, evaluation is skipped
- Use --force to re-evaluate even if results exist

Usage:
    # Evaluate all three test types (default)
    poetry run python workflow_sifta/evaluate_test_trajectory.py \
        --run_dir optimization_runs/run_20260108_120000

    # Evaluate specific test type
    poetry run python workflow_sifta/evaluate_test_trajectory.py \
        --run_dir optimization_runs/run_20260108_120000 \
        --test_type test_1

    # Force re-evaluation
    poetry run python workflow_sifta/evaluate_test_trajectory.py \
        --run_dir optimization_runs/run_20260108_120000 \
        --force

Test type shortcuts:
    test_1 -> test_1_cross_new_data
    test_2 -> test_2_new_script_train_data
    test_3 -> test_3_new_script_new_data
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import dspy
import numpy as np
from tqdm import tqdm

from prismadv.llm.dspy.models.column_wise_module import HumanDesignedConstraintGenerationModule
from prismadv.utils import get_project_root, load_dotenv
from workflow_sifta.optimization_fns import OptimizationWorkflow

load_dotenv()

# Mapping from short test names to full test type names
TEST_TYPE_MAP = {
    "test_1": "test_1_cross_new_data",
    "test_2": "test_2_new_script_train_data",
    "test_3": "test_3_new_script_new_data",
}

ALL_TEST_TYPES = list(TEST_TYPE_MAP.values())


def find_existing_test_trajectory(run_dir: Path, test_type: str) -> Path | None:
    """
    Find existing test trajectory file for the given test type.

    Returns:
        Path to existing file if found, None otherwise
    """
    # Look for files matching pattern: test_trajectory_{test_type}_*.json
    pattern = f"test_trajectory_{test_type}_*.json"
    matching_files = list(run_dir.glob(pattern))

    if matching_files:
        # Return the most recent one (by filename timestamp)
        return sorted(matching_files)[-1]

    return None


def load_round_data(run_dir: Path) -> list[dict]:
    """Load all round data from a run directory."""
    round_files = sorted(run_dir.glob("round_*.json"))
    rounds = []
    for round_file in round_files:
        with open(round_file, 'r') as f:
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

    if test_type not in summary:
        raise ValueError(f"Test configuration '{test_type}' not found in summary. Available: {list(summary.keys())}")

    return summary[test_type]


def extract_best_model_changes(rounds: list[dict]) -> list[dict]:
    """
    Extract only the iterations where the best model changed across all rounds.

    This is more efficient than evaluating at every iteration, since we only
    evaluate when the best prompt actually improves.

    Returns list of dicts with:
        - round: round number
        - global_iteration: global iteration number (continuous across rounds)
        - local_iteration: iteration within the round
        - candidate_idx: index of current best candidate
        - instructions: prompt instructions
        - eval_score: validation score (from optimization)
        - is_improvement: whether this is an improvement over previous best
    """
    changes = []
    global_iteration_offset = 0
    global_best_candidate_idx = None
    global_best_score = -float('inf')

    for round_data in rounds:
        round_num = round_data.get("round", -1)
        sifta_details = round_data.get("sifta_details", {})

        if not sifta_details:
            print(f"Warning: No SIFTA details for round {round_num}, skipping")
            continue

        num_iterations = sifta_details["num_iterations"]
        candidate_to_iteration = {int(k): int(v) for k, v in sifta_details["candidate_to_iteration"].items()}
        iteration_to_candidate = {int(k): int(v) for k, v in sifta_details["iteration_to_candidate"].items()}
        all_candidate_instructions = sifta_details["all_candidate_instructions"]
        val_aggregate_scores = sifta_details["val_aggregate_scores"]

        # Add base program at start of each round (iteration -1)
        base_candidate_idx = 0
        base_score = val_aggregate_scores[base_candidate_idx]

        if global_best_candidate_idx is None:
            # First round, first point
            changes.append({
                "round": round_num,
                "global_iteration": -1 + global_iteration_offset,
                "local_iteration": -1,
                "candidate_idx": base_candidate_idx,
                "instructions": all_candidate_instructions[base_candidate_idx],
                "eval_score": base_score,
                "is_improvement": True,  # First model
            })
            global_best_candidate_idx = base_candidate_idx
            global_best_score = base_score

        # Track through iterations in this round
        current_best_candidate_idx = base_candidate_idx
        current_best_score = base_score

        for local_iteration in range(num_iterations + 1):
            if local_iteration in iteration_to_candidate:
                new_candidate_idx = iteration_to_candidate[local_iteration]
                new_score = val_aggregate_scores[new_candidate_idx]

                # Check if this is a new best
                if new_score > current_best_score:
                    is_global_improvement = new_score > global_best_score

                    changes.append({
                        "round": round_num,
                        "global_iteration": local_iteration + global_iteration_offset,
                        "local_iteration": local_iteration,
                        "candidate_idx": new_candidate_idx,
                        "instructions": all_candidate_instructions[new_candidate_idx],
                        "eval_score": new_score,
                        "is_improvement": is_global_improvement,
                    })

                    current_best_candidate_idx = new_candidate_idx
                    current_best_score = new_score

                    if is_global_improvement:
                        global_best_candidate_idx = new_candidate_idx
                        global_best_score = new_score

        # Update offset for next round
        global_iteration_offset += num_iterations + 2

    return changes


def apply_instructions_to_module(module: HumanDesignedConstraintGenerationModule, instructions: dict):
    """Apply instructions to module's predictors.

    Note: We must create a copy of the signature before modifying it,
    because DSPy signatures are class-level attributes. Modifying them
    directly would affect all module instances.
    """
    for pred_name, pred in module.named_predictors():
        if pred_name in instructions:
            # Create a copy of the signature to avoid modifying the class-level attribute
            pred.signature = pred.signature.with_instructions(instructions[pred_name])


def evaluate_on_test_set(
        workflow: OptimizationWorkflow,
        module: HumanDesignedConstraintGenerationModule,
        test_script_list: list[str],
        llm_name: str,
        num_threads: int = 8,
) -> dict:
    """
    Evaluate module on test set using both F1 and fail precision.

    Returns dict with complete evaluation metrics including:
    - F1, precision, recall
    - Full confusion matrix (TP, FP, TN, FN)
    - Fail precision score
    - Per-script results
    """
    # Evaluate F1 on test set
    f1_results = workflow.evaluate_f1_on_test_set(
        module=module,
        test_script_name_list=test_script_list,
        num_threads=num_threads,
    )

    # Compute fail precision on test set
    from workflow_sifta.metrics import create_metric
    fail_precision_metric = create_metric(
        workflow=workflow,
        llm_name=llm_name,
        convert_nan_to_zero=True,
    )

    # Prepare test dataset for fail precision calculation
    test_examples = workflow.prepare_single_column_training_dataset(
        script_name_list=test_script_list,
        processed_data_label="0",
        new_processed_data_label_list=workflow.test_processed_data_label_list,
    )

    # Calculate fail precision scores
    fail_precision_scores = []
    for example in test_examples:
        try:
            prediction = module(**example.inputs())
            score = fail_precision_metric(example, prediction, None)
            if not np.isnan(score):
                fail_precision_scores.append(float(score))
        except Exception:
            # Skip examples that fail
            continue

    avg_fail_precision = np.mean(fail_precision_scores) if fail_precision_scores else None

    return {
        # F1 metrics
        "f1": float(f1_results['f1']) if not np.isnan(f1_results['f1']) else None,
        "precision": float(f1_results['precision']) if not np.isnan(f1_results['precision']) else None,
        "recall": float(f1_results['recall']) if not np.isnan(f1_results['recall']) else None,

        # Full confusion matrix
        "tp": f1_results['tp'],
        "fp": f1_results['fp'],
        "tn": f1_results['tn'],
        "fn": f1_results['fn'],

        # Additional metrics
        "fail_precision": float(avg_fail_precision) if avg_fail_precision is not None else None,
        "total_constraints": f1_results['total_constraints'],
        "total_non_compilable": f1_results['total_non_compilable'],

        # Per-script details
        "per_script_results": f1_results.get('per_script_results', []),
    }


def find_existing_baseline(run_dir: Path, test_type: str) -> Path | None:
    """Find existing baseline file for the given test type."""
    pattern = f"baseline_{test_type}_*.json"
    matching_files = list(run_dir.glob(pattern))
    if matching_files:
        return sorted(matching_files)[-1]
    return None


def evaluate_baseline(
        run_dir: Path,
        test_type: str,
        llm_name: str,
        num_threads: int,
        force: bool,
        project_root: Path,
) -> dict | None:
    """
    Evaluate the handwritten HumanDesignedConstraintGenerationModule (no optimization) as baseline.
    This is the original human-designed prompt before any optimization.
    Returns the baseline results dict, or None if skipped.
    """
    existing_file = find_existing_baseline(run_dir, test_type)
    if existing_file and not force:
        print(f"  [Baseline] Found existing, loading from {existing_file.name}")
        with open(existing_file, 'r') as f:
            return json.load(f)

    print(f"  [Baseline] Evaluating handwritten HumanDesignedConstraintGenerationModule...")

    # Configure DSPy
    lm = dspy.LM(f"openai/{llm_name}", temperature=1, max_tokens=32000)
    dspy.configure(lm=lm)

    # Load summary and test configuration
    summary = load_summary(run_dir)
    test_config = get_test_configuration(summary, test_type)

    # Initialize workflow
    dataset_name = summary.get("train_dataset_name")
    dataset_subtasks = {dataset_name: summary.get("dataset_subtasks", {}).get(dataset_name, [])}

    workflow = OptimizationWorkflow(
        train_dataset_subtasks=dataset_subtasks,
        train_processed_data_label_list=test_config['label_list'],
        project_root=project_root,
    )

    # Create baseline module with original handwritten instructions (no modifications)
    module = HumanDesignedConstraintGenerationModule()

    # Evaluate on test set
    test_results = evaluate_on_test_set(
        workflow=workflow,
        module=module,
        test_script_list=test_config['script_list'],
        llm_name=llm_name,
        num_threads=num_threads,
    )

    baseline_data = {
        "test_type": test_type,
        "test_config": test_config,
        "evaluation_timestamp": datetime.now().isoformat(),
        "llm_name": llm_name,
        "description": "Initial HumanDesignedConstraintGenerationModule without optimization",
        "test_f1": test_results["f1"],
        "test_precision": test_results["precision"],
        "test_recall": test_results["recall"],
        "test_tp": test_results["tp"],
        "test_fp": test_results["fp"],
        "test_tn": test_results["tn"],
        "test_fn": test_results["fn"],
        "test_fail_precision": test_results["fail_precision"],
        "test_total_constraints": test_results["total_constraints"],
        "test_total_non_compilable": test_results["total_non_compilable"],
    }

    # Save baseline results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = run_dir / f"baseline_{test_type}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(baseline_data, f, indent=2)

    print(f"  [Baseline] F1: {baseline_data['test_f1']:.4f}, Fail Prec: {baseline_data['test_fail_precision']:.4f}")
    print(f"  [Baseline] Saved to: {output_file.name}")

    workflow.cleanup_spark_sessions()
    return baseline_data


def evaluate_single_test_type(
        run_dir: Path,
        test_type: str,
        llm_name: str,
        num_threads: int,
        sample_interval: int,
        force: bool,
        project_root: Path,
) -> bool:
    """
    Evaluate a single test type. Returns True if evaluation was performed, False if skipped.
    """
    # First evaluate baseline
    evaluate_baseline(
        run_dir=run_dir,
        test_type=test_type,
        llm_name=llm_name,
        num_threads=num_threads,
        force=force,
        project_root=project_root,
    )

    # Check for existing results
    existing_file = find_existing_test_trajectory(run_dir, test_type)
    if existing_file and not force:
        print(f"\n[{test_type}] Found existing results, skipping (use --force to re-evaluate)")
        with open(existing_file, 'r') as f:
            existing_data = json.load(f)
        f1_scores = [r["test_f1"] for r in existing_data["trajectory"] if r["test_f1"] is not None]
        if f1_scores:
            print(f"  Points: {len(existing_data['trajectory'])}, Final F1: {f1_scores[-1]:.4f}")
        return False

    if existing_file and force:
        print(f"\n[{test_type}] Found existing results but --force specified, re-evaluating")

    print(f"\n{'=' * 80}")
    print(f"Evaluating: {test_type}")
    print(f"{'=' * 80}")

    # Configure DSPy
    lm = dspy.LM(f"openai/{llm_name}", temperature=1, max_tokens=32000)
    dspy.configure(lm=lm)

    # Load summary and test configuration
    summary = load_summary(run_dir)
    test_config = get_test_configuration(summary, test_type)

    print(f"Config: {test_config['description']}")
    print(f"Scripts: {test_config['script_list']}")

    # Initialize workflow with test configuration
    dataset_name = summary.get("train_dataset_name")
    dataset_subtasks = {dataset_name: summary.get("dataset_subtasks", {}).get(dataset_name, [])}

    workflow = OptimizationWorkflow(
        train_dataset_subtasks=dataset_subtasks,
        train_processed_data_label_list=test_config['label_list'],
        project_root=project_root,
    )

    # Load optimization trajectory
    rounds = load_round_data(run_dir)
    trajectory = extract_best_model_changes(rounds)

    print(f"Model changes to evaluate: {len(trajectory)}")

    # Filter by sample interval if specified
    if sample_interval > 1:
        trajectory = trajectory[::sample_interval]
        print(f"Sampled points: {len(trajectory)}")

    # Evaluate each point in trajectory
    results = []
    for point in tqdm(trajectory, desc=f"Evaluating {test_type}"):
        module = HumanDesignedConstraintGenerationModule()
        apply_instructions_to_module(module, point["instructions"])

        test_results = evaluate_on_test_set(
            workflow=workflow,
            module=module,
            test_script_list=test_config['script_list'],
            llm_name=llm_name,
            num_threads=num_threads,
        )

        result_point = {
            "round": point["round"],
            "global_iteration": point["global_iteration"],
            "local_iteration": point["local_iteration"],
            "candidate_idx": point["candidate_idx"],
            "is_improvement": point.get("is_improvement", False),
            "eval_score": point["eval_score"],
            "test_f1": test_results["f1"],
            "test_precision": test_results["precision"],
            "test_recall": test_results["recall"],
            "test_tp": test_results["tp"],
            "test_fp": test_results["fp"],
            "test_tn": test_results["tn"],
            "test_fn": test_results["fn"],
            "test_fail_precision": test_results["fail_precision"],
            "test_total_constraints": test_results["total_constraints"],
            "test_total_non_compilable": test_results["total_non_compilable"],
            "test_per_script_results": test_results["per_script_results"],
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
        "num_threads": num_threads,
        "sample_interval": sample_interval,
        "trajectory": results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    f1_scores = [r["test_f1"] for r in results if r["test_f1"] is not None]
    fp_scores = [r["test_fail_precision"] for r in results if r["test_fail_precision"] is not None]

    print(f"\nResults saved to: {output_file}")
    if f1_scores:
        print(f"Test F1: {f1_scores[0]:.4f} -> {f1_scores[-1]:.4f} ({f1_scores[-1] - f1_scores[0]:+.4f})")
    if fp_scores:
        print(f"Fail Prec: {fp_scores[0]:.4f} -> {fp_scores[-1]:.4f} ({fp_scores[-1] - fp_scores[0]:+.4f})")

    workflow.cleanup_spark_sessions()
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate optimization trajectory on test sets")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to optimization run directory"
    )
    parser.add_argument(
        "--test_type",
        type=str,
        default=None,
        help="Test type: test_1, test_2, test_3 (or full name). If not specified, evaluates all three."
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of threads for evaluation (default: 8)"
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

    # Determine which test types to evaluate
    if args.test_type is None:
        test_types = ALL_TEST_TYPES
        print(f"No test type specified, evaluating all three: {list(TEST_TYPE_MAP.keys())}")
    else:
        # Map short name to full name if needed
        test_type = TEST_TYPE_MAP.get(args.test_type, args.test_type)
        if test_type not in ALL_TEST_TYPES:
            raise ValueError(f"Invalid test type: {args.test_type}. Use test_1, test_2, test_3 or full names.")
        test_types = [test_type]

    print(f"{'=' * 80}")
    print(f"Test Trajectory Evaluation")
    print(f"{'=' * 80}")
    print(f"Run directory: {run_dir}")
    print(f"Test types: {test_types}")
    print(f"LLM: {args.llm_name}")
    print(f"Threads: {args.num_threads}")

    # Evaluate each test type
    evaluated_count = 0
    for test_type in test_types:
        was_evaluated = evaluate_single_test_type(
            run_dir=run_dir,
            test_type=test_type,
            llm_name=args.llm_name,
            num_threads=args.num_threads,
            sample_interval=args.sample_interval,
            force=args.force,
            project_root=project_root,
        )
        if was_evaluated:
            evaluated_count += 1

    print(f"\n{'=' * 80}")
    print(f"Done! Evaluated {evaluated_count}/{len(test_types)} test types.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
