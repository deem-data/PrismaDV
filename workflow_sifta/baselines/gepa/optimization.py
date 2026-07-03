"""Script-level GEPA baseline optimization for PrismaDV."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import dspy
from dspy import GEPA
from dspy.dsp.utils.settings import settings

from prismadv.utils import get_project_root, load_dotenv
from workflow_sifta.baselines.gepa.dataset_preparation import prepare_script_level_dataset
from workflow_sifta.baselines.gepa.metrics import (
    create_script_level_metric,
    create_script_level_metric_with_feedback,
)
from workflow_sifta.baselines.gepa.script_level_module import ScriptLevelConstraintGenerationModule
from workflow_sifta.optimization_fns import OptimizationWorkflow

settings.max_errors = 10
load_dotenv()

# =============================================================================
# Parse Command Line Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="GEPA baseline optimization for PrismaDV")
parser.add_argument(
    "--split_strategy_from",
    type=str,
    default=None,
    help="Reference run to copy splits from (e.g., run_20260108_233004). If not provided, generates new splits."
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help="Dataset name to use. If not provided and split_strategy_from is set, reads from reference run."
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=1,
    help="Random seed for split generation (only used if split_strategy_from is None)"
)
parser.add_argument(
    "--label_split_ratio",
    type=float,
    default=0.5,
    help="Fraction of labels for train/eval (only used if split_strategy_from is None)"
)
parser.add_argument(
    "--script_split_ratio",
    type=float,
    default=0.6,
    help="Fraction of scripts for train/eval (only used if split_strategy_from is None)"
)
parser.add_argument(
    "--train_eval_script_split_ratio",
    type=float,
    default=0.5,
    help="Fraction of train_eval scripts for train (only used if split_strategy_from is None)"
)
parser.add_argument(
    "--eval_sample_size",
    type=int,
    default=20,
    help="Number of examples to sample for eval set"
)
parser.add_argument(
    "--max_rounds",
    type=int,
    default=1,
    help="Maximum number of optimization rounds"
)

args = parser.parse_args()

# =============================================================================
# Configuration from Arguments
# =============================================================================
SPLIT_STRATEGY_FROM = args.split_strategy_from
RANDOM_SEED = args.random_seed
LABEL_SPLIT_RATIO = args.label_split_ratio
SCRIPT_SPLIT_RATIO = args.script_split_ratio
TRAIN_EVAL_SCRIPT_SPLIT_RATIO = args.train_eval_script_split_ratio
EVAL_SAMPLE_SIZE = args.eval_sample_size

llm_name = "gpt-4.1-mini"
lm = dspy.LM(f"openai/{llm_name}", temperature=1, max_tokens=32000)
dspy.configure(lm=lm)

# =============================================================================
# Dataset Configuration
# =============================================================================
# If dataset_name not provided, try to read from reference run
if args.dataset_name:
    train_dataset_name = args.dataset_name
elif SPLIT_STRATEGY_FROM:
    # Read dataset name from reference run
    reference_summary_path = Path(get_project_root()) / "optimization_runs" / SPLIT_STRATEGY_FROM / "summary.json"
    if reference_summary_path.exists():
        with open(reference_summary_path, 'r') as f:
            reference_summary = json.load(f)
        train_dataset_name = reference_summary.get("train_dataset_name", "hr_analytics")
        print(f"Read dataset name from reference run: {train_dataset_name}")
    else:
        print(f"Warning: Reference run summary not found at {reference_summary_path}")
        print(f"Using default dataset: hr_analytics")
        train_dataset_name = "hr_analytics"
else:
    # Default dataset
    train_dataset_name = "hr_analytics"
    print(f"Using default dataset: {train_dataset_name}")
dataset_subtasks = {
    train_dataset_name: [
        "general_task",
    ]
}

# All available error labels (1-25)
all_processed_data_labels = [str(i) for i in range(1, 26)]

# =============================================================================
# Load splits from reference run or generate new splits
# =============================================================================
def load_splits_from_reference_run(reference_run_name: str):
    """Load dataset splits from a reference optimization run."""
    reference_summary_path = Path(get_project_root()) / "optimization_runs" / reference_run_name / "summary.json"

    if not reference_summary_path.exists():
        raise FileNotFoundError(f"Reference run summary not found: {reference_summary_path}")

    with open(reference_summary_path, 'r') as f:
        reference_summary = json.load(f)

    print(f"\n{'='*60}")
    print(f"Loading splits from reference run: {reference_run_name}")
    print(f"{'='*60}")

    return {
        "train_eval_label_list": reference_summary["train_eval_label_list"],
        "cross_new_data_test_label_list": reference_summary["cross_new_data_test_label_list"],
        "train_eval_script_name_list": reference_summary["train_eval_script_name_list"],
        "train_script_name_list": reference_summary["train_script_name_list"],
        "eval_script_name_list": reference_summary["eval_script_name_list"],
        "new_script_name_list": reference_summary.get("new_script_name_list", []),
    }

if SPLIT_STRATEGY_FROM is not None:
    # Load splits from reference run
    splits = load_splits_from_reference_run(SPLIT_STRATEGY_FROM)
    train_eval_label_list = splits["train_eval_label_list"]
    cross_new_data_test_label_list = splits["cross_new_data_test_label_list"]
    # Script splits will be loaded after we get available scripts
    reference_script_splits = splits
else:
    # Generate new splits using random seed
    print(f"\n{'='*60}")
    print(f"Generating new splits with RANDOM_SEED={RANDOM_SEED}")
    print(f"{'='*60}")

    random.seed(RANDOM_SEED)
    shuffled_labels = all_processed_data_labels.copy()
    random.shuffle(shuffled_labels)

    train_eval_label_count = int(len(shuffled_labels) * LABEL_SPLIT_RATIO)
    train_eval_label_list = shuffled_labels[:train_eval_label_count]
    cross_new_data_test_label_list = shuffled_labels[train_eval_label_count:]
    reference_script_splits = None

print(f"Label split: train_eval={len(train_eval_label_list)}, cross_new_data_test={len(cross_new_data_test_label_list)}")
print(f"  Train/Eval labels: {train_eval_label_list}")
print(f"  Cross-new-data test labels: {cross_new_data_test_label_list}")

# Initialize workflow (scripts will be discovered dynamically)
workflow = OptimizationWorkflow(
    train_dataset_subtasks=dataset_subtasks,
    train_processed_data_label_list=train_eval_label_list,  # Only train/eval labels for optimization
    project_root=get_project_root(),
)

# Get all available scripts for the subtask
dataset_name = list(dataset_subtasks.keys())[0]
subtask_name = dataset_subtasks[dataset_name][0]
all_script_names = workflow.get_available_script_names_for_subtask(dataset_name, subtask_name)

print(f"Found {len(all_script_names)} scripts for subtask '{subtask_name}'")

# =============================================================================
# Script Splitting for Evaluation Levels
# =============================================================================
if reference_script_splits is not None:
    # Use splits from reference run
    train_eval_script_name_list = reference_script_splits["train_eval_script_name_list"]
    train_script_name_list = reference_script_splits["train_script_name_list"]
    eval_script_name_list = reference_script_splits["eval_script_name_list"]
    new_script_name_list = reference_script_splits["new_script_name_list"]

    # Verify all scripts exist in current dataset
    missing_scripts = []
    for script in train_eval_script_name_list + new_script_name_list:
        if script not in all_script_names:
            missing_scripts.append(script)

    if missing_scripts:
        raise ValueError(f"Scripts from reference run not found in dataset: {missing_scripts}")

    print(f"\nUsing script splits from reference run")
else:
    # Generate new splits using random seed
    random.seed(RANDOM_SEED)
    shuffled_scripts = all_script_names.copy()
    random.shuffle(shuffled_scripts)

    train_eval_script_count = int(len(shuffled_scripts) * SCRIPT_SPLIT_RATIO)
    train_eval_script_name_list = shuffled_scripts[:train_eval_script_count]
    new_script_name_list = shuffled_scripts[train_eval_script_count:]

    # Further split train_eval scripts into separate train and eval
    train_script_count = int(len(train_eval_script_name_list) * TRAIN_EVAL_SCRIPT_SPLIT_RATIO)
    train_script_name_list = train_eval_script_name_list[:train_script_count]
    eval_script_name_list = train_eval_script_name_list[train_script_count:]

print(f"Script split: train_eval={len(train_eval_script_name_list)}, new_script={len(new_script_name_list)}")
print(f"  Train scripts ({len(train_script_name_list)}): {train_script_name_list}")
print(f"  Eval scripts ({len(eval_script_name_list)}): {eval_script_name_list}")
print(f"  New script test scripts ({len(new_script_name_list)}): {new_script_name_list}")

# =============================================================================
# Test Set Configuration (matching main method's 3-test setup)
# =============================================================================
print(f"\n=== Test Set Configuration ===")
print(
    f"Test 1 - Cross new data: train+eval scripts ({len(train_eval_script_name_list)}) + new data labels ({len(cross_new_data_test_label_list)})")
print(
    f"Test 2 - New script + train data: new scripts ({len(new_script_name_list)}) + train data labels ({len(train_eval_label_list)})")
print(
    f"Test 3 - New script + new data: new scripts ({len(new_script_name_list)}) + new data labels ({len(cross_new_data_test_label_list)})")
print(f"  New script list: {new_script_name_list}")

print("\nPreparing datasets...")
print("Preparing training dataset...")
train_set = prepare_script_level_dataset(
    dataset_subtasks=dataset_subtasks,
    script_name_list=train_script_name_list,
    processed_data_label="0",
    new_processed_data_label_list=train_eval_label_list,  # Only train/eval labels
    project_root=get_project_root(),
)
print(f"Train set size: {len(train_set)}")

print("Preparing validation dataset (will sample)...")
eval_set_all = prepare_script_level_dataset(
    dataset_subtasks=dataset_subtasks,
    script_name_list=eval_script_name_list,
    processed_data_label="0",
    new_processed_data_label_list=train_eval_label_list,  # Only train/eval labels
    project_root=get_project_root(),
)

# Sample eval set to fixed size (matching main method)
if len(eval_set_all) > EVAL_SAMPLE_SIZE:
    eval_set = random.sample(eval_set_all, EVAL_SAMPLE_SIZE)
else:
    eval_set = eval_set_all

print(f"Eval set size: {len(eval_set)} (sampled from {len(eval_set_all)} total)")

initial_program = ScriptLevelConstraintGenerationModule()

metric = create_script_level_metric(
    workflow=workflow,
    convert_nan_to_zero=True,
)
metric_with_feedback = create_script_level_metric_with_feedback(
    workflow=workflow,
    feedback_sample_size=5,
    feedback_seed=0,
    convert_nan_to_zero=True,
)

evaluate = dspy.Evaluate(
    devset=eval_set,
    metric=metric,
    num_threads=8,
    display_table=False,
    display_progress=True,
    provide_traceback=False,
)


def extract_program_instructions(module: ScriptLevelConstraintGenerationModule) -> dict:
    """Extract all predictor instructions from a module."""
    instructions = {}
    for pred_name, pred in module.named_predictors():
        instructions[pred_name] = pred.signature.instructions
    return instructions


# Set up logging directory for optimization history
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path(get_project_root()) / "optimization_runs" / f"baseline_gepa_run_{run_timestamp}"
log_dir.mkdir(parents=True, exist_ok=True)
print(f"Logging optimization history to: {log_dir}")

optimization_history = []
latest_program_list = [initial_program]

# Evaluate initial program (matching main method)
print("\nEvaluating initial program...")
initial_evaluation = evaluate(initial_program)
initial_score = initial_evaluation['score']
print(f"Initial program eval score: {initial_score:.4f}")

# Log initial program (matching main method naming: round_-1_initial.json)
initial_program_data = {
    "round": -1,  # -1 indicates initial program before any optimization
    "score": float(initial_score),
    "instructions": extract_program_instructions(initial_program),
    "timestamp": datetime.now().isoformat(),
}

initial_file = log_dir / "round_-1_initial.json"
with open(initial_file, 'w') as f:
    json.dump(initial_program_data, f, indent=2)
print(f"Saved initial program to: {initial_file}")

round_num = 0
max_rounds = args.max_rounds


def write_summary(final_program: dict | None = None):
    summary = {
        "run_timestamp": run_timestamp,
        "max_rounds": max_rounds,
        "llm_name": llm_name,
        "baseline_method": "GEPA",
        "split_strategy_from": SPLIT_STRATEGY_FROM,  # Track where splits came from
        # Dataset configuration
        "train_dataset_name": train_dataset_name,
        "dataset_subtasks": dataset_subtasks,
        # Label configuration
        "all_processed_data_labels": all_processed_data_labels,
        "train_eval_label_list": train_eval_label_list,
        "cross_new_data_test_label_list": cross_new_data_test_label_list,
        # Script configuration
        "train_eval_script_name_list": train_eval_script_name_list,
        "train_script_name_list": train_script_name_list,
        "eval_script_name_list": eval_script_name_list,
        "new_script_name_list": new_script_name_list,
        # Test set configurations (matching main method's 3-test setup)
        "test_1_cross_new_data": {
            "description": "Train+eval scripts + new data labels",
            "script_list": train_eval_script_name_list,
            "label_list": cross_new_data_test_label_list,
        },
        "test_2_new_script_train_data": {
            "description": "New scripts + train data labels",
            "script_list": new_script_name_list,
            "label_list": train_eval_label_list,
        },
        "test_3_new_script_new_data": {
            "description": "New scripts + new data labels",
            "script_list": new_script_name_list,
            "label_list": cross_new_data_test_label_list,
        },
        # Dataset sizes
        "train_set_size": len(train_set),
        "eval_set_size": len(eval_set),
        "eval_set_all_size": len(eval_set_all),
        "feedback_sample_size": 5,
        # Initial program evaluation
        "initial_program_evaluation": initial_program_data if 'initial_program_data' in globals() else None,
        # Optimization history
        "optimization_history": optimization_history,
    }
    if final_program is not None:
        summary["final_program"] = final_program
    summary_file = log_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved optimization summary to: {summary_file}")


# Persist run metadata early for recovery on partial runs.
write_summary()

while round_num < max_rounds:
    print(f"\n{'=' * 60}")
    print(f"=== Round {round_num + 1}/{max_rounds} ===")
    print(f"{'=' * 60}")

    current_program = latest_program_list[-1]

    evaluation_results = evaluate(current_program)
    current_score = evaluation_results["score"]
    print(f"Current program score: {current_score}")

    current_instructions = extract_program_instructions(current_program)
    round_data = {
        "round": round_num,
        "score": float(current_score),
        "instructions": current_instructions,
        "timestamp": datetime.now().isoformat(),
    }

    optimizer = GEPA(
        metric=metric_with_feedback,
        max_full_evals=15,
        num_threads=8,
        track_stats=True,
        use_merge=False,
        reflection_minibatch_size=3,
        skip_perfect_score=True,
        reflection_lm=dspy.LM(
            model="openai/gpt-5",
            temperature=1.0,
            max_tokens=32000,
        ),
    )

    optimized_program = optimizer.compile(
        current_program,
        trainset=train_set,
        valset=eval_set,
    )
    latest_program_list.append(optimized_program)
    round_num += 1

    optimized_instructions = extract_program_instructions(optimized_program)
    round_data["optimized_instructions"] = optimized_instructions

    # Log GEPA trajectory details (similar to SIFTA)
    gepa_details = {}
    if hasattr(optimizer, 'history') and optimizer.history:
        # GEPA optimizer exposes history with trajectory information
        gepa_details["num_iterations"] = len(optimizer.history)
        gepa_details["history"] = []

        for i, entry in enumerate(optimizer.history):
            history_entry = {
                "iteration": i,
                "score": float(entry.get("score", 0.0)) if entry.get("score") is not None else None,
                "type": entry.get("type", "unknown"),  # e.g., "proposal", "accepted", "rejected"
            }
            # Add instructions if available
            if "program" in entry:
                history_entry["instructions"] = extract_program_instructions(entry["program"])
            gepa_details["history"].append(history_entry)

        # Extract validation scores
        val_scores = [e.get("score", 0.0) for e in optimizer.history if e.get("score") is not None]
        if val_scores:
            gepa_details["val_aggregate_scores"] = [float(s) for s in val_scores]
            gepa_details["best_score"] = float(max(val_scores))
            gepa_details["best_iteration"] = int(val_scores.index(max(val_scores)))

    round_data["gepa_details"] = gepa_details

    round_file = log_dir / f"round_{round_num}.json"
    with open(round_file, "w") as f:
        json.dump(round_data, f, indent=2)

    optimization_history.append(round_data)

    print(f"Round {round_num} complete. Total programs: {len(latest_program_list)}")
    print(f"Saved round {round_num} data to: {round_file}")

    write_summary()

print(f"\n{'=' * 60}")
print("=== Final Evaluation ===")
print(f"{'=' * 60}")
final_program = latest_program_list[-1]
final_evaluation = evaluate(final_program)
final_score = final_evaluation["score"]
print(f"Final program score: {final_score}")

final_instructions = extract_program_instructions(final_program)
final_data = {
    "round": "final",
    "score": float(final_score),
    "instructions": final_instructions,
    "timestamp": datetime.now().isoformat(),
}

final_file = log_dir / "final_program.json"
with open(final_file, "w") as f:
    json.dump(final_data, f, indent=2)
print(f"Saved final program to: {final_file}")

write_summary(final_program=final_data)

print("\nOptimization complete")

workflow.cleanup_spark_sessions()
