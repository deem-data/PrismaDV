"""
Evaluate column-level GEPA optimization trajectory on test sets using script-level F1.

This script uses the SAME F1 calculation as the original evaluate_test_trajectory.py:
- Aggregates all TP/FP/TN/FN across all scripts/labels FIRST
- Then calculates F1 from the aggregated confusion matrix (micro-average)

This is different from the original gepa_column/evaluate_test_trajectory.py which
calculates F1 per column and then averages (macro-average).

Usage:
    # Evaluate all three test types (default)
    poetry run python workflow_sifta/baselines/gepa_column/evaluate_test_trajectory_with_script_f1.py \
        --run_dir optimization_runs/gepa_column_run_20260114_211801

    # Evaluate specific test type
    poetry run python workflow_sifta/baselines/gepa_column/evaluate_test_trajectory_with_script_f1.py \
        --run_dir optimization_runs/gepa_column_run_20260114_211801 \
        --test_type test_1

    # Force re-evaluation
    poetry run python workflow_sifta/baselines/gepa_column/evaluate_test_trajectory_with_script_f1.py \
        --run_dir optimization_runs/gepa_column_run_20260114_211801 \
        --force
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import dspy
import numpy as np
from tqdm import tqdm

from prismadv.llm.dspy.models.column_wise_module import ConstraintGenerationModule
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
    """Find existing test trajectory file for the given test type."""
    pattern = f"test_trajectory_{test_type}_*_script_f1.json"
    matching_files = list(run_dir.glob(pattern))
    if matching_files:
        return sorted(matching_files)[-1]
    return None


def load_round_data(run_dir: Path) -> list[dict]:
    """Load all round data from a run directory, including round_-1_initial.json and final_program.json."""
    rounds = []

    # Load initial round
    initial_file = run_dir / "round_-1_initial.json"
    if initial_file.exists():
        with open(initial_file, 'r') as f:
            rounds.append(json.load(f))

    # Load regular rounds
    round_files = sorted(run_dir.glob("round_[0-9]*.json"))
    for round_file in round_files:
        with open(round_file, 'r') as f:
            rounds.append(json.load(f))

    # Load final program if exists
    final_program_file = run_dir / "final_program.json"
    if final_program_file.exists():
        with open(final_program_file, 'r') as f:
            final_data = json.load(f)
            # Mark it as final
            final_data["round"] = "final"
            rounds.append(final_data)

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


def extract_programs_to_evaluate(rounds: list[dict]) -> list[dict]:
    """
    Extract programs to evaluate from round data.

    For GEPA runs, we typically only have:
    - round_-1_initial.json: the initial (baseline) program
    - final_program.json: the final optimized program (if optimization completed)

    Returns list of dicts with:
        - round: round identifier (-1 for initial, "final" for final program)
        - instructions: prompt instructions
        - score: evaluation score from training (if available)
    """
    programs = []

    for round_data in rounds:
        round_id = round_data.get("round")
        instructions = round_data.get("instructions", {})
        score = round_data.get("score", None)

        if round_id == -1:
            # Initial program
            programs.append({
                "round": -1,
                "name": "initial",
                "instructions": instructions,
                "score": score,
            })
        elif round_id == "final":
            # Final program
            programs.append({
                "round": "final",
                "name": "final",
                "instructions": instructions,
                "score": score,
            })

    return programs


def apply_instructions_to_module(module: ConstraintGenerationModule, instructions: dict):
    """Apply instructions to module's predictors."""
    for pred_name, pred in module.named_predictors():
        if pred_name in instructions:
            pred.signature = pred.signature.with_instructions(instructions[pred_name])


def evaluate_on_test_set(
        workflow: OptimizationWorkflow,
        module: ConstraintGenerationModule,
        test_script_list: list[str],
        num_threads: int = 8,
) -> dict:
    """
    Evaluate module on test set using script-level F1 score (same as original).

    This uses workflow.evaluate_f1_on_test_set() which:
    1. Aggregates ALL TP/FP/TN/FN across all scripts and labels
    2. Calculates F1 from aggregated confusion matrix

    Returns dict with complete evaluation metrics.
    """
    # Use the same evaluation method as the original evaluate_test_trajectory.py
    f1_results = workflow.evaluate_f1_on_test_set(
        module=module,
        test_script_name_list=test_script_list,
        num_threads=num_threads,
    )

    # Also compute fail precision if needed (matching original behavior)
    from workflow_sifta.metrics import create_metric
    fail_precision_metric = create_metric(
        workflow=workflow,
        llm_name=workflow.llm_name if hasattr(workflow, 'llm_name') else "gpt-4.1-mini",
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


def evaluate_single_test_type(
        run_dir: Path,
        test_type: str,
        llm_name: str,
        num_threads: int,
        force: bool,
        project_root: Path,
) -> bool:
    """
    Evaluate a single test type. Returns True if evaluation was performed, False if skipped.
    """
    # Check for existing results
    existing_file = find_existing_test_trajectory(run_dir, test_type)
    if existing_file and not force:
        print(f"\n[{test_type}] Found existing results, skipping (use --force to re-evaluate)")
        with open(existing_file, 'r') as f:
            existing_data = json.load(f)
        programs = existing_data.get("programs", [])
        if programs:
            print(f"  Programs: {len(programs)}")
            for prog in programs:
                f1 = prog.get("test_f1")
                if f1 is not None:
                    print(f"    - {prog['name']}: F1={f1:.4f}")
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

    # Load programs to evaluate
    rounds = load_round_data(run_dir)
    programs = extract_programs_to_evaluate(rounds)

    print(f"Programs to evaluate: {len(programs)}")
    for prog in programs:
        print(f"  - {prog['name']} (training score: {prog['score']})")

    # Evaluate each program
    results = []
    for prog in tqdm(programs, desc=f"Evaluating {test_type}"):
        module = ConstraintGenerationModule()
        apply_instructions_to_module(module, prog["instructions"])

        test_results = evaluate_on_test_set(
            workflow=workflow,
            module=module,
            test_script_list=test_config['script_list'],
            num_threads=num_threads,
        )

        result_point = {
            "round": prog["round"],
            "name": prog["name"],
            "training_score": prog["score"],
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
        results.append(result_point)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = run_dir / f"test_trajectory_{test_type}_{timestamp}_script_f1.json"

    output_data = {
        "test_type": test_type,
        "test_config": test_config,
        "evaluation_timestamp": datetime.now().isoformat(),
        "llm_name": llm_name,
        "evaluation_method": "script_level_f1_micro_average",
        "description": "Uses workflow.evaluate_f1_on_test_set() - aggregates TP/FP/TN/FN first, then calculates F1",
        "programs": results,
        "summary": {
            "num_programs": len(results),
            "initial_f1": results[0]["test_f1"] if results else None,
            "final_f1": results[-1]["test_f1"] if len(results) > 1 else None,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(results)} program evaluations to: {output_file}")

    # Print results
    for result in results:
        print(f"\n{result['name']} (round {result['round']}):")
        print(f"  Training score: {result['training_score']:.2f}")
        if result["test_f1"] is not None:
            print(f"  Test F1: {result['test_f1']:.4f}")
            print(f"  Test Precision: {result['test_precision']:.4f}")
            print(f"  Test Recall: {result['test_recall']:.4f}")
            print(f"  Confusion: TP={result['test_tp']}, FP={result['test_fp']}, TN={result['test_tn']}, FN={result['test_fn']}")
        if result["test_fail_precision"] is not None:
            print(f"  Fail Precision: {result['test_fail_precision']:.4f}")

    workflow.cleanup_spark_sessions()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GEPA column-level optimization using script-level F1 (micro-average)"
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Path to optimization run directory")
    parser.add_argument(
        "--test_type",
        type=str,
        default=None,
        help="Test type to evaluate (test_1, test_2, test_3, or full name). If not specified, evaluates all.",
    )
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini", help="LLM to use for evaluation")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of parallel threads")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if results exist")

    args = parser.parse_args()

    project_root = get_project_root()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"Run directory: {run_dir}")
    print(f"LLM: {args.llm_name}")
    print(f"Threads: {args.num_threads}")
    print(f"Evaluation method: Script-level F1 (micro-average)")

    # Determine which test types to evaluate
    if args.test_type:
        # Map short names to full names
        test_type = TEST_TYPE_MAP.get(args.test_type, args.test_type)
        if test_type not in ALL_TEST_TYPES:
            raise ValueError(f"Invalid test_type: {args.test_type}. Must be one of {list(TEST_TYPE_MAP.keys())}")
        test_types_to_evaluate = [test_type]
    else:
        test_types_to_evaluate = ALL_TEST_TYPES

    print(f"Test types to evaluate: {test_types_to_evaluate}\n")

    # Evaluate each test type
    evaluated_count = 0
    for test_type in test_types_to_evaluate:
        was_evaluated = evaluate_single_test_type(
            run_dir=run_dir,
            test_type=test_type,
            llm_name=args.llm_name,
            num_threads=args.num_threads,
            force=args.force,
            project_root=project_root,
        )
        if was_evaluated:
            evaluated_count += 1

    print(f"\n{'=' * 80}")
    print(f"Done! Evaluated {evaluated_count}/{len(test_types_to_evaluate)} test types.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
