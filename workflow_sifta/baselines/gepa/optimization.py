"""Script-level GEPA baseline optimization for PrismaDV."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import dspy
from dspy import GEPA
from dspy.dsp.utils.settings import settings

from prismadv.llm_backend.model_routes import litellm_route
from prismadv.project_manager import is_clean_label
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root, load_dotenv
from workflow_sifta.baselines.gepa.dataset_preparation import prepare_script_level_dataset
from workflow_sifta.baselines.gepa.metrics import (
    create_script_level_metric,
    create_script_level_metric_with_feedback,
)
from workflow_sifta.baselines.gepa.script_level_module import ScriptLevelConstraintGenerationModule
from workflow_sifta.runtime import budget as opt_budget, gepa_state
from workflow_sifta._0_splits import load_split
from workflow_sifta.optimization_fns import OptimizationWorkflow
from workflow_sifta.runtime.usage import REFLECTION_LM, TASK_LM, reset_usage, set_rollout_id, tracked_lm, usage_summary

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
parser.add_argument(
    "--llm_name",
    type=str,
    default="gpt-4.1-mini",
    help="Task model for constraint generation. Must match the arm being compared "
         "against (default: gpt-4.1-mini)"
)
parser.add_argument(
    "--reflection_llm_name",
    type=str,
    default="gpt-5.2",
    help="Reflection/proposal model. Must match the arm being compared against "
         "(default: gpt-5.2)"
)
parser.add_argument(
    "--max_full_evals",
    type=int,
    default=15,
    help="Max full evaluations per round (default: 15)"
)
parser.add_argument(
    "--max_metric_calls",
    type=int,
    default=None,
    help="Absolute metric-call budget. Overrides --max_full_evals. Use with "
         "--match_budget_from to equalize total cost against a SIFTA run"
)
parser.add_argument(
    "--match_budget_from",
    type=str,
    default=None,
    help="Path to a SIFTA run dir (e.g. optimization_runs/run_2026...). Reads its true "
         "total_metric_computations from summary.json and uses it as --max_metric_calls"
)

parser.add_argument(
    "--validation_passes",
    type=int,
    default=None,
    help="Budget: K validation passes, i.e. K x |eval set| program evaluations in total (workflow_sifta/runtime/budget.py); "
         "replaces the metric-call budget when set"
)
parser.add_argument(
    "--split_seed",
    type=int,
    default=None,
    help="Replicate N: read benchmarks/EIDBench-synth/<dataset>/splits/split<N>.yaml and use N as the GEPA seed "
         "too (overrides --random_seed; needs --dataset_name)"
)

args = parser.parse_args()
if args.split_seed is not None:
    if args.split_strategy_from or not args.dataset_name:
        parser.error("--split_seed needs --dataset_name and excludes --split_strategy_from")
    args.random_seed = args.split_seed
GEPA_SEED = args.split_seed if args.split_seed is not None else 0  # 0 is dspy.GEPA's default seed

if args.match_budget_from:
    _summary = Path(args.match_budget_from) / "summary.json"
    if not _summary.exists():
        raise FileNotFoundError(f"No summary.json in {args.match_budget_from}")
    _total = json.loads(_summary.read_text()).get("total_metric_computations")
    if not _total:
        raise ValueError(
            f"{_summary} has no total_metric_computations. It predates budget tracking; "
            f"rerun the SIFTA arm or pass --max_metric_calls explicitly."
        )
    args.max_metric_calls = _total
    print(f"Matching budget from {args.match_budget_from}: max_metric_calls={_total}")

# =============================================================================
# Configuration from Arguments
# =============================================================================
SPLIT_STRATEGY_FROM = args.split_strategy_from
RANDOM_SEED = args.random_seed
LABEL_SPLIT_RATIO = args.label_split_ratio
SCRIPT_SPLIT_RATIO = args.script_split_ratio
TRAIN_EVAL_SCRIPT_SPLIT_RATIO = args.train_eval_script_split_ratio
EVAL_SAMPLE_SIZE = args.eval_sample_size

llm_name = args.llm_name
reflection_llm_name = args.reflection_llm_name
reset_usage()
set_rollout_id(args.split_seed)  # replicate N: its own cached answers
run_started = time.time()
lm = tracked_lm(litellm_route(llm_name), temperature=0.6, max_tokens=32000, usage_role=TASK_LM)
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

all_processed_data_labels = [
    label for label in ProjectManager(project_root=get_project_root(), dataset_name=train_dataset_name)
    .get_available_processed_data_labels_for_subtask("general_task")
    if not is_clean_label(label)
]

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

SPLIT_SEED = args.split_seed
if SPLIT_SEED is not None:
    split = load_split(train_dataset_name, SPLIT_SEED)
    train_eval_label_list = split["train_eval_labels"]
    cross_new_data_test_label_list = split["test_labels"]
    reference_script_splits = {
        "train_eval_script_name_list": split["train_scripts"] + split["eval_scripts"],
        "train_script_name_list": split["train_scripts"],
        "eval_script_name_list": split["eval_scripts"],
        "new_script_name_list": split["test_scripts"],
    }
    random.seed(SPLIT_SEED)  # draws the eval sample below
elif SPLIT_STRATEGY_FROM is not None:
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

    print(f"\nUsing script splits from {'split file' if SPLIT_SEED is not None else 'reference run'}")
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

metric = create_script_level_metric(workflow=workflow)
metric_with_feedback = create_script_level_metric_with_feedback(
    workflow=workflow,
    feedback_sample_size=5,
    feedback_seed=0,
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
while True:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(get_project_root()) / "optimization_runs" / f"baseline_gepa_run_{run_timestamp}"
    try:
        log_dir.mkdir(parents=True, exist_ok=False)
        break
    except FileExistsError:
        time.sleep(1)
print(f"Logging optimization history to: {log_dir}")

optimization_history = []
latest_program_list = [initial_program]

# Evaluate initial program (matching main method)
print("\nEvaluating initial program...")
# Every evaluation from here on counts toward the budget, the initial evaluation included.
budget = opt_budget.install(args.validation_passes, len(eval_set))
gepa_state.apply()  # the logged val subscores of the start program stay its own
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
        "reflection_llm_name": reflection_llm_name,
        "gepa_max_full_evals": args.max_full_evals,
        "gepa_max_metric_calls": args.max_metric_calls,
        "budget_matched_from": args.match_budget_from,
        "baseline_method": "GEPA",
        "split_strategy_from": SPLIT_STRATEGY_FROM,  # Track where splits came from
        "split_seed": SPLIT_SEED,
        "gepa_seed": GEPA_SEED,
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
        # Tokens, cost, calls and API latency, split by task and reflection LM.
        "llm_usage": usage_summary(),
        **opt_budget.summary_fields(),
        "wall_clock_s": round(time.time() - run_started, 3),
    }
    if final_program is not None:
        summary["final_program"] = final_program
    summary_file = log_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved optimization summary to: {summary_file}")


# Persist run metadata early for recovery on partial runs.
write_summary()

while round_num < max_rounds and not budget.exhausted:
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

    # GEPA takes exactly one of max_metric_calls / max_full_evals / auto.
    budget_kwargs = (
        {"max_metric_calls": args.max_metric_calls} if args.max_metric_calls is not None
        else {"max_full_evals": args.max_full_evals}
    )
    budget_kwargs = opt_budget.engine_budget_kwargs(args.validation_passes, budget_kwargs)
    optimizer = GEPA(
        metric=metric_with_feedback,
        **budget_kwargs,
        num_threads=8,
        track_stats=True,
        use_merge=False,
        reflection_minibatch_size=3,
        skip_perfect_score=True,
        seed=GEPA_SEED,
        reflection_lm=tracked_lm(
            model=litellm_route(reflection_llm_name),
            usage_role=REFLECTION_LM,
            temperature=0.6,
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

from workflow_sifta import hard_exit  # noqa: E402
hard_exit(0)
