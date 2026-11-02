import argparse
import json
import logging
import random
import time
import re
from datetime import datetime
from pathlib import Path
import dspy
from dspy import GEPA
from dspy.dsp.utils.settings import settings

from prismadv.llm.dspy.models.column_wise_module import ConstraintGenerationModule
from prismadv.llm_backend.model_routes import litellm_route
from prismadv.project_manager import is_clean_label
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import (
    get_project_root,
    load_dotenv,
)
from workflow_sifta.baselines.gepa_column.metrics import (
    create_column_level_f1_metric,
    create_column_level_f1_metric_with_feedback,
)
from workflow_sifta.runtime import budget as opt_budget, gepa_state
from workflow_sifta._0_splits import load_split
from workflow_sifta.metrics.optimization_metrics import METRIC_REV
from workflow_sifta.optimization_fns import OptimizationWorkflow
from workflow_sifta.runtime.usage import REFLECTION_LM, TASK_LM, reset_usage, set_rollout_id, tracked_lm, usage_summary

settings.max_errors = 10
load_dotenv()

# =============================================================================
# Parse Command Line Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="Column-level GEPA baseline optimization for PrismaDV")
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
    "--early_stop_patience",
    type=int,
    default=0,
    help="Early stopping: stop if no improvement for N consecutive GEPA iterations (0 = disabled). "
         "Detects iterations by intercepting GEPA log messages (reliable method). "
         "Recommended: 5-10 for quick testing, 15-30 for production runs."
)
parser.add_argument(
    "--early_stop_min_delta",
    type=float,
    default=0.0001,
    help="Minimum improvement to consider as progress (default: 0.0001)"
)
parser.add_argument(
    "--metric",
    type=str,
    choices=["f1", "fail_precision"],
    default="f1",
    help="Optimization objective: f1 = macro F1 of the column's constraints over the script's labels. Use "
         "fail_precision to hold the objective constant against SIFTA and isolate the optimizer (default: f1)"
)
parser.add_argument(
    "--llm_name",
    type=str,
    default="gpt-4.1-mini",
    help="Task model for constraint generation, bare name (default: gpt-4.1-mini). "
         "Must match the SIFTA arm being compared against"
)
parser.add_argument(
    "--reflection_llm_name",
    type=str,
    default="gpt-5.2",
    help="Reflection/proposal model. Must match the SIFTA arm being compared against "
         "(default: gpt-5.2)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="GEPA's own seed: minibatch and candidate sampling, and the feedback sampler. "
         "Vary it for replicates; the split is fixed by --random_seed / --split_strategy_from"
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
    "--split_seed",
    type=int,
    default=None,
    help="Replicate N: read benchmarks/EIDBench-synth/<dataset>/splits/split<N>.yaml and use N as the GEPA seed "
         "too (overrides --random_seed and --seed; needs --dataset_name)"
)
parser.add_argument(
    "--validation_passes",
    type=int,
    default=None,
    help="Budget: K validation passes, i.e. K x |eval set| program evaluations in total (workflow_sifta/runtime/budget.py); "
         "replaces the metric-call budget when set"
)

args = parser.parse_args()
if args.split_seed is not None:
    if args.split_strategy_from or not args.dataset_name:
        parser.error("--split_seed needs --dataset_name and excludes --split_strategy_from")
    args.random_seed = args.seed = args.split_seed

# Budget matching: SIFTA's real cost includes its condense passes and the fail
# precision sampler's precompute, neither of which sits inside max_metric_calls.
# Reading the recorded total keeps the two arms on equal total compute.
if args.match_budget_from:
    import json as _json
    from pathlib import Path as _Path

    _summary = _Path(args.match_budget_from) / "summary.json"
    if not _summary.exists():
        raise FileNotFoundError(f"No summary.json in {args.match_budget_from}")
    _total = _json.loads(_summary.read_text()).get("total_metric_computations")
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
MAX_ROUNDS = args.max_rounds

# --- Early Stopping Configuration ---
EARLY_STOP_PATIENCE = args.early_stop_patience
EARLY_STOP_MIN_DELTA = args.early_stop_min_delta

# --- LLM Configuration ---
LLM_NAME = args.llm_name  # Model for constraint generation
LLM_TEMPERATURE = 0.6  # Sampling temperature
LLM_MAX_TOKENS = 32000  # Max output tokens

REFLECTION_LLM_NAME = args.reflection_llm_name  # Model for GEPA reflection/proposal
REFLECTION_LLM_TEMPERATURE = 0.6
REFLECTION_LLM_MAX_TOKENS = 32000

# --- Replicate seed ---
GEPA_SEED = args.seed

# --- Initial Program Configuration ---
INITIAL_PROGRAM_TYPE = "base"

# --- GEPA Optimizer Configuration ---
GEPA_MAX_FULL_EVALS = args.max_full_evals  # Max full evaluations per round
GEPA_MAX_METRIC_CALLS = args.max_metric_calls  # Absolute budget; overrides max_full_evals
METRIC_TYPE = args.metric  # "f1" or "fail_precision"
GEPA_NUM_THREADS = 8  # Parallel threads for evaluation
GEPA_REFLECTION_MINIBATCH_SIZE = 3  # Minibatch size for reflection
GEPA_SKIP_PERFECT_SCORE = True  # Skip examples with perfect score
GEPA_USE_MERGE = False  # Don't merge candidates

# Configure DSPy LM
reset_usage()
set_rollout_id(args.split_seed)  # replicate N: its own cached answers
run_started = time.time()
lm = tracked_lm(litellm_route(LLM_NAME), temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS,
                usage_role=TASK_LM)
dspy.configure(lm=lm)
print(f"llm={LLM_NAME} reflection_llm={REFLECTION_LLM_NAME} seed={GEPA_SEED} metric={args.metric}")

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
        "random_seed": reference_summary.get("random_seed"),
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
    # The reference run drew its splits and then its eval sample from one seeded
    # stream. Replaying the same draws puts this run's eval sample on the same
    # examples as the SIFTA arm, instead of leaving it to an unseeded generator.
    if splits["random_seed"] is not None:
        RANDOM_SEED = splits["random_seed"]
    random.seed(RANDOM_SEED)
    random.shuffle(all_processed_data_labels.copy())
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

    if SPLIT_SEED is None:
        # Second draw of the reference run's stream (see the label shuffle above), so
        # the eval sample below comes out of the same generator state.
        random.shuffle(all_script_names.copy())
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

# =============================================================================
# DATASET PREPARATION
# =============================================================================

print("\nPreparing datasets...")
print("Preparing train dataset (NO condensing for GEPA baseline)...")

# Train set - use full dataset without condensing
train_set = workflow.prepare_single_column_training_dataset(
    script_name_list=train_script_name_list,
    processed_data_label="0",
    new_processed_data_label_list=train_eval_label_list,
)
print(f"Train set size: {len(train_set)}")

print("Preparing eval dataset (sampling)...")

# Eval set: prepare all examples from eval scripts, then sample
eval_set_all = workflow.prepare_single_column_training_dataset(
    script_name_list=eval_script_name_list,
    processed_data_label="0",
    new_processed_data_label_list=train_eval_label_list,
)

# Sample eval set to fixed size
if len(eval_set_all) > EVAL_SAMPLE_SIZE:
    eval_set = random.sample(eval_set_all, EVAL_SAMPLE_SIZE)
else:
    eval_set = eval_set_all

print(f"Eval set size: {len(eval_set)} (sampled from {len(eval_set_all)} total)")

# =============================================================================
# METRIC & EVALUATOR SETUP
# =============================================================================

initial_program = ConstraintGenerationModule()
print(f"Using ConstraintGenerationModule (base) as initial program")

# Create metrics. fail_precision reuses SIFTA's own metric so the two arms optimize
# the identical objective and only the optimizer differs.
if METRIC_TYPE == "fail_precision":
    from workflow_sifta.metrics import create_metric as create_fp_metric
    from workflow_sifta.metrics import create_metric_with_feedback as create_fp_metric_with_feedback

    metric = create_fp_metric(
        workflow=workflow,
        llm_name=LLM_NAME,
        convert_nan_to_zero=True,
    )
    metric_with_feedback = create_fp_metric_with_feedback(
        workflow=workflow,
        llm_name=LLM_NAME,
        convert_nan_to_zero=True,
    )
else:
    metric = create_column_level_f1_metric(workflow=workflow)
    metric_with_feedback = create_column_level_f1_metric_with_feedback(
        workflow=workflow,
        feedback_seed=GEPA_SEED,
    )
print(f"Optimization objective: {METRIC_TYPE}")

# Eval evaluator - created once with fixed eval set
evaluate = dspy.Evaluate(
    devset=eval_set,
    metric=metric,
    num_threads=GEPA_NUM_THREADS,
    display_table=False,
    display_progress=True,
    provide_traceback=False,
)


# =============================================================================
# EARLY STOPPING
# =============================================================================


class EarlyStoppingTracker:
    """Track optimization progress and trigger early stopping if no improvement.

    Works with GEPALoggingHandler which intercepts GEPA's log messages to reliably
    detect full validation runs ("Iteration X: ... full valset score: Y").
    """

    def __init__(self, patience: int, min_delta: float = 0.0001):
        """
        Args:
            patience: Number of GEPA iterations without improvement before stopping (0 = disabled)
            min_delta: Minimum score improvement to consider as progress
        """
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = patience > 0

        self.best_score = -float('inf')
        self.iterations_without_improvement = 0
        self.total_iterations = 0
        self.should_stop = False

        print(f"\n{'=' * 60}")
        print("Early Stopping Configuration")
        print(f"{'=' * 60}")
        if self.enabled:
            print(f"  Enabled: Yes")
            print(f"  Patience: {self.patience} GEPA iterations")
            print(f"  Min delta: {self.min_delta}")
            print(f"  Detection method: GEPA log interception (100% reliable)")
            print(f"  Will stop if no improvement for {self.patience} consecutive iterations")
        else:
            print(f"  Enabled: No (patience = 0)")
        print(f"{'=' * 60}\n")

    def update(self, score: float) -> bool:
        """
        Update tracker with new score from a GEPA iteration.

        Called by GEPALoggingHandler when it detects a full validation log message.

        Args:
            score: Validation score from GEPA log message

        Returns:
            True if should stop early, False otherwise
        """
        if not self.enabled:
            return False

        self.total_iterations += 1

        # Check if this is an improvement
        if score > self.best_score + self.min_delta:
            # Improvement found
            improvement = score - self.best_score
            self.best_score = score
            self.iterations_without_improvement = 0
            print(f"  [Early Stop] Iteration {self.total_iterations}: Improvement +{improvement:.4f} → new best: {self.best_score:.4f}")
        else:
            # No improvement
            self.iterations_without_improvement += 1
            print(f"  [Early Stop] Iteration {self.total_iterations}: No improvement ({self.iterations_without_improvement}/{self.patience})")

            # Check if we should stop
            if self.iterations_without_improvement >= self.patience:
                self.should_stop = True
                print(f"\n{'!' * 60}")
                print(f"EARLY STOPPING TRIGGERED")
                print(f"{'!' * 60}")
                print(f"  No improvement for {self.patience} consecutive GEPA iterations")
                print(f"  Best score: {self.best_score:.4f}")
                print(f"  Total iterations: {self.total_iterations}")
                print(f"{'!' * 60}\n")
                return True

        return False

    def get_summary(self) -> dict:
        """Get summary of early stopping status."""
        return {
            "enabled": self.enabled,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_score": float(self.best_score) if self.best_score != -float('inf') else None,
            "total_iterations": self.total_iterations,
            "iterations_without_improvement": self.iterations_without_improvement,
            "stopped_early": self.should_stop,
        }


class EarlyStoppingException(Exception):
    """Exception raised to signal early stopping.

    Note: We also use KeyboardInterrupt for immediate termination since it propagates
    through GEPA's exception handling, while custom exceptions are caught and logged.
    """
    pass


class GEPALoggingHandler(logging.Handler):
    """Custom logging handler to intercept GEPA log messages and trigger early stopping.

    GEPA logs specific messages after full validation runs:
    - "Iteration X: Base program full valset score: Y"
    - "Iteration X: Selected program N score: Y"

    We parse these to reliably detect full validation runs and check early stopping.
    """

    def __init__(self, early_stop_tracker: EarlyStoppingTracker):
        super().__init__()
        self.early_stop_tracker = early_stop_tracker

        # Regex patterns to match GEPA's full validation log messages
        self.base_pattern = re.compile(r'Iteration (\d+): Base program full valset score: ([\d.]+)')
        self.selected_pattern = re.compile(r'Iteration (\d+): Selected program \d+ score: ([\d.]+)')

    def emit(self, record):
        """Process log records and trigger early stopping if needed."""
        if not self.early_stop_tracker.enabled:
            return

        try:
            message = record.getMessage()

            # Check if this is a full validation log message
            match = self.base_pattern.search(message) or self.selected_pattern.search(message)

            if match:
                iteration = int(match.group(1))
                score = float(match.group(2))

                # Update early stopping tracker
                should_stop = self.early_stop_tracker.update(score)

                if should_stop:
                    # Early stopping triggered - raise KeyboardInterrupt
                    print(f"\n{'!' * 60}")
                    print(f"FORCING IMMEDIATE STOP VIA KeyboardInterrupt")
                    print(f"  Detected full validation via GEPA log")
                    print(f"  Iteration {iteration}: score = {score:.4f}")
                    print(f"  Best score: {self.early_stop_tracker.best_score:.4f}")
                    print(f"  No improvement for {self.early_stop_tracker.iterations_without_improvement} validations")
                    print(f"{'!' * 60}\n")
                    raise KeyboardInterrupt("Early stopping triggered: no improvement")

        except Exception as e:
            # Don't let errors in the handler break the logging system
            if isinstance(e, KeyboardInterrupt):
                raise  # Re-raise KeyboardInterrupt
            pass




# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def iteration_info_of(detailed_results) -> dict:
    """Iteration bookkeeping in the shape SIFTA's result object reports.

    DSPy's `DspyGEPAResult` (3.0.x) exposes no `get_iteration_info`; it only
    lists the accepted candidates in discovery order and the metric-call count
    at which each was found. So for GEPA the "iteration" of a candidate is its
    discovery rank, and rejected proposals are not recoverable from the result.
    """
    if hasattr(detailed_results, "get_iteration_info"):
        return detailed_results.get_iteration_info()
    n = len(detailed_results.candidates)
    candidate_to_iteration = {0: -1}
    candidate_to_iteration.update({idx: idx - 1 for idx in range(1, n)})
    return {
        "candidate_to_iteration": candidate_to_iteration,
        "iteration_to_candidate": {idx - 1: idx for idx in range(1, n)},
        "rejected_iterations": [],
        "num_iterations": n - 1,
        "iteration_semantics": "discovery rank of accepted candidates; rejected proposals not exposed by DspyGEPAResult",
    }


def extract_program_instructions(module):
    """Extract all predictor instructions from a module."""
    instructions = {}
    for pred_name, pred in module.named_predictors():
        instructions[pred_name] = pred.signature.instructions
    return instructions


def extract_full_signature_info(module):
    """Extract complete signature information including fields and instructions."""
    signatures = {}
    for pred_name, pred in module.named_predictors():
        sig = pred.signature
        signatures[pred_name] = {
            "instructions": sig.instructions,
            "input_fields": {k: str(v) for k, v in sig.input_fields.items()},
            "output_fields": {k: str(v) for k, v in sig.output_fields.items()},
        }
    return signatures


def log_prompt_to_file(log_dir: Path, round_num: int, iteration: int, candidate_idx: int, instructions: dict, score: float = None):
    """Log prompt instructions to a human-readable file."""
    prompt_dir = log_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True)

    # Create filename
    if iteration == -1:
        filename = f"round_{round_num}_initial.txt"
    else:
        filename = f"round_{round_num}_iter_{iteration:03d}_candidate_{candidate_idx}.txt"

    filepath = prompt_dir / filename

    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"GEPA Column-Level Baseline - Prompt Log\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Round: {round_num}\n")
        if iteration >= 0:
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Candidate Index: {candidate_idx}\n")
        if score is not None:
            f.write(f"Score: {score:.4f}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        for pred_name, instruction in instructions.items():
            f.write(f"[{pred_name}]\n")
            f.write("-" * 80 + "\n")
            f.write(instruction + "\n")
            f.write("\n" + "=" * 80 + "\n\n")

    return filepath


def create_prompt_evolution_log(log_dir: Path, round_num: int, detailed_results):
    """Create a detailed log showing prompt evolution throughout optimization."""
    prompt_dir = log_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True)

    evolution_file = prompt_dir / f"round_{round_num}_evolution.md"

    with open(evolution_file, 'w') as f:
        f.write(f"# GEPA Column-Level Optimization - Prompt Evolution\n\n")
        f.write(f"**Round:** {round_num}\n\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")

        # Get iteration info
        iteration_info = iteration_info_of(detailed_results)
        candidate_to_iteration = iteration_info["candidate_to_iteration"]
        iteration_to_candidate = iteration_info["iteration_to_candidate"]

        f.write(f"## Summary\n\n")
        f.write(f"- **Total Iterations:** {iteration_info['num_iterations']}\n")
        f.write(f"- **Total Candidates:** {len(detailed_results.candidates)}\n")
        f.write(f"- **Accepted Candidates:** {len(detailed_results.candidates) - 1} (excluding base)\n")
        f.write(f"- **Rejected Iterations:** {len(iteration_info['rejected_iterations'])}\n")
        f.write(f"- **Best Candidate Index:** {detailed_results.best_idx}\n")
        f.write(f"- **Best Score:** {detailed_results.val_aggregate_scores[detailed_results.best_idx]:.4f}\n\n")
        f.write("---\n\n")

        # Log each iteration
        f.write(f"## Iteration Timeline\n\n")

        # Base candidate
        f.write(f"### Iteration -1: Base Program\n\n")
        f.write(f"- **Candidate Index:** 0\n")
        f.write(f"- **Score:** {detailed_results.val_aggregate_scores[0]:.4f}\n")
        f.write(f"- **Status:** Base (starting point)\n\n")
        f.write("```\n")
        f.write(f"Prompt file: round_{round_num}_initial.txt\n")
        f.write("```\n\n")

        # All other iterations
        for iteration in range(iteration_info['num_iterations'] + 1):
            if iteration in iteration_to_candidate:
                candidate_idx = iteration_to_candidate[iteration]
                score = detailed_results.val_aggregate_scores[candidate_idx]

                # Check if this is the best candidate
                is_best = (candidate_idx == detailed_results.best_idx)
                best_marker = " 🌟 **BEST**" if is_best else ""

                f.write(f"### Iteration {iteration}: Candidate {candidate_idx}{best_marker}\n\n")
                f.write(f"- **Candidate Index:** {candidate_idx}\n")
                f.write(f"- **Score:** {score:.4f}\n")
                f.write(f"- **Status:** Accepted\n")

                # Show parent
                if candidate_idx < len(detailed_results.parents):
                    parent_list = detailed_results.parents[candidate_idx]
                    if parent_list:
                        f.write(f"- **Parent Candidate(s):** {parent_list}\n")

                f.write("\n```\n")
                f.write(f"Prompt file: round_{round_num}_iter_{iteration:03d}_candidate_{candidate_idx}.txt\n")
                f.write("```\n\n")

            elif iteration in iteration_info['rejected_iterations']:
                f.write(f"### Iteration {iteration}: Rejected ❌\n\n")
                f.write(f"- **Status:** Proposal rejected (did not improve score)\n\n")

        f.write("---\n\n")

        # Final summary with scores
        f.write(f"## Score Summary\n\n")
        f.write(f"| Candidate | Iteration | Score | Status |\n")
        f.write(f"|-----------|-----------|-------|--------|\n")

        for candidate_idx, score in enumerate(detailed_results.val_aggregate_scores):
            iteration = candidate_to_iteration.get(candidate_idx, -1)
            is_best = (candidate_idx == detailed_results.best_idx)
            status = "🌟 BEST" if is_best else ("Base" if candidate_idx == 0 else "Accepted")
            f.write(f"| {candidate_idx} | {iteration} | {score:.4f} | {status} |\n")

        f.write("\n---\n\n")
        f.write(f"*Generated: {datetime.now().isoformat()}*\n")

    return evolution_file


def write_summary(
        final_program_data: dict | None = None,
):
    """Write optimization summary to JSON file."""
    summary = {
        "run_timestamp": run_timestamp,
        "max_rounds": MAX_ROUNDS,
        "llm_name": LLM_NAME,
        "reflection_llm_name": REFLECTION_LLM_NAME,
        # Baseline configuration
        "baseline_type": "gepa_column",
        "metric_type": "f1_score" if METRIC_TYPE == "f1" else "fail_precision",
        "metric_rev": None if METRIC_TYPE == "f1" else METRIC_REV,
        # Split strategy
        "split_strategy_from": SPLIT_STRATEGY_FROM,
        "split_seed": SPLIT_SEED,
        "random_seed": RANDOM_SEED,
        "gepa_seed": GEPA_SEED,
        # Initial program configuration
        "initial_program_type": INITIAL_PROGRAM_TYPE,
        "initial_program_instructions": extract_program_instructions(initial_program),
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
        # Test set configurations (for downstream evaluation)
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
        # GEPA configuration
        "gepa_max_full_evals": GEPA_MAX_FULL_EVALS,
        "gepa_max_metric_calls": GEPA_MAX_METRIC_CALLS,
        "budget_matched_from": args.match_budget_from,
        "gepa_num_threads": GEPA_NUM_THREADS,
        "gepa_reflection_minibatch_size": GEPA_REFLECTION_MINIBATCH_SIZE,
        "gepa_skip_perfect_score": GEPA_SKIP_PERFECT_SCORE,
        "gepa_use_merge": GEPA_USE_MERGE,
        # Early stopping configuration
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        # Initial program evaluation
        "initial_program_evaluation": initial_program_data if 'initial_program_data' in globals() else None,
        # History
        "optimization_history": optimization_history,
        # Tokens, cost, calls and API latency, split by task and reflection LM.
        "llm_usage": usage_summary(),
        **opt_budget.summary_fields(),
        "wall_clock_s": round(time.time() - run_started, 3),
    }
    if final_program_data is not None:
        summary["final_program"] = final_program_data

    summary_file = log_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved optimization summary to: {summary_file}")


# =============================================================================
# LOGGING SETUP
# =============================================================================

while True:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(get_project_root()) / "optimization_runs" / f"gepa_column_run_{run_timestamp}"
    try:
        log_dir.mkdir(parents=True, exist_ok=False)
        break
    except FileExistsError:
        time.sleep(1)
print(f"Logging optimization history to: {log_dir}")

optimization_history = []
latest_program_list = [initial_program]

# Evaluate initial program
print("\nEvaluating initial program...")
# Every evaluation from here on counts toward the budget, the initial evaluation included.
budget = opt_budget.install(args.validation_passes, len(eval_set))
gepa_state.apply()  # the logged val subscores of the start program stay its own
initial_evaluation = evaluate(initial_program)
initial_score = initial_evaluation['score']
print(f"Initial program eval F1 score: {initial_score:.4f}")

# Log initial program
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

# Log initial prompts to human-readable file
initial_instructions = extract_program_instructions(initial_program)
initial_prompt_file = log_prompt_to_file(log_dir, -1, -1, 0, initial_instructions, initial_score)
print(f"Saved initial prompts to: {initial_prompt_file}")

# Persist run metadata early for recovery on partial runs
write_summary()

# =============================================================================
# OPTIMIZATION LOOP
# =============================================================================

round_num = 0

while round_num < MAX_ROUNDS and not budget.exhausted:
    print(f"\n{'=' * 60}")
    print(f"=== Round {round_num + 1}/{MAX_ROUNDS} ===")
    print(f"{'=' * 60}")

    current_program = latest_program_list[-1]

    # Evaluate current program on eval set
    print("Evaluating on eval set...")
    evaluation_results = evaluate(current_program)
    current_score = evaluation_results['score']
    print(f"Current program eval F1 score: {current_score:.4f}")

    # Extract and log program instructions
    current_instructions = extract_program_instructions(current_program)
    round_data = {
        "round": round_num,
        "score": float(current_score),
        "instructions": current_instructions,
        "timestamp": datetime.now().isoformat(),
    }

    # Run GEPA optimization
    print(f"\nRunning GEPA optimization with train set size: {len(train_set)}")

    # Create early stopping tracker for this round
    early_stop_tracker = EarlyStoppingTracker(
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA
    )

    # Set up logging handler to intercept GEPA log messages for early stopping
    gepa_logger = logging.getLogger('dspy.teleprompt.gepa.gepa')
    logging_handler = None
    if EARLY_STOP_PATIENCE > 0:
        logging_handler = GEPALoggingHandler(early_stop_tracker)
        logging_handler.setLevel(logging.INFO)
        gepa_logger.addHandler(logging_handler)

    # GEPA takes exactly one of max_metric_calls / max_full_evals / auto.
    budget_kwargs = (
        {"max_metric_calls": GEPA_MAX_METRIC_CALLS} if GEPA_MAX_METRIC_CALLS is not None
        else {"max_full_evals": GEPA_MAX_FULL_EVALS}
    )
    budget_kwargs = opt_budget.engine_budget_kwargs(args.validation_passes, budget_kwargs)
    optimizer = GEPA(
        metric=metric_with_feedback,
        **budget_kwargs,
        num_threads=GEPA_NUM_THREADS,
        track_stats=True,
        use_merge=GEPA_USE_MERGE,
        reflection_minibatch_size=GEPA_REFLECTION_MINIBATCH_SIZE,
        skip_perfect_score=GEPA_SKIP_PERFECT_SCORE,
        seed=GEPA_SEED,
        reflection_lm=tracked_lm(
            model=litellm_route(REFLECTION_LLM_NAME),
            usage_role=REFLECTION_LM,
            temperature=REFLECTION_LLM_TEMPERATURE,
            max_tokens=REFLECTION_LLM_MAX_TOKENS,
        ),
    )

    # Run optimization with early stopping exception handling
    try:
        optimized_program = optimizer.compile(
            current_program,
            trainset=train_set,  # Use full train set without condensing
            valset=eval_set,
        )
        early_stopped = False
    except KeyboardInterrupt as e:
        print(f"\n{'=' * 60}")
        print(f"Optimization stopped early (KeyboardInterrupt caught): {e}")
        print(f"{'=' * 60}\n")
        # Use the current program (GEPA's best program is in optimizer state)
        # Try to get the best program from GEPA if available
        if hasattr(optimizer, '_best_program') and optimizer._best_program is not None:
            optimized_program = optimizer._best_program
        else:
            optimized_program = current_program
        early_stopped = True
    except EarlyStoppingException as e:
        print(f"\n{'=' * 60}")
        print(f"Optimization stopped early (exception caught): {e}")
        print(f"{'=' * 60}\n")
        # Use the current best program
        optimized_program = current_program
        early_stopped = True

    # Clean up logging handler
    if logging_handler is not None:
        gepa_logger.removeHandler(logging_handler)

    # Check if early stopping was triggered (even if exception didn't propagate)
    # GEPA may catch the exception internally during reflection/proposal,
    # so we check the tracker's state as a fallback
    if not early_stopped and early_stop_tracker.enabled and early_stop_tracker.should_stop:
        print(f"\n{'=' * 60}")
        print(f"Early stopping detected (via tracker flag)")
        print(f"{'=' * 60}")
        print(f"  Early stopping was triggered during optimization")
        print(f"  Best score: {early_stop_tracker.best_score:.4f}")
        print(f"  Total iterations: {early_stop_tracker.total_iterations}")
        print(f"{'=' * 60}\n")
        # Use the current program (GEPA's best program is already in optimized_program)
        early_stopped = True

    latest_program_list.append(optimized_program)
    round_num += 1

    # Log early stopping information
    early_stop_summary = early_stop_tracker.get_summary()
    round_data["early_stopping"] = early_stop_summary
    if early_stopped:
        print(f"✓ Early stopping saved iterations: stopped at {early_stop_summary['total_iterations']}/{GEPA_MAX_FULL_EVALS}")
        print(f"✓ Exiting optimization early - no further rounds will be run")

    # Log optimized program instructions
    optimized_instructions = extract_program_instructions(optimized_program)
    round_data["optimized_instructions"] = optimized_instructions

    # Extract and log detailed GEPA results if available
    if hasattr(optimized_program, 'detailed_results'):
        detailed_results = optimized_program.detailed_results

        # Extract instructions from all candidate programs
        all_candidate_instructions = []
        for candidate_module in detailed_results.candidates:
            candidate_instructions = extract_program_instructions(candidate_module)
            all_candidate_instructions.append(candidate_instructions)

        # Get iteration tracking information
        iteration_info = iteration_info_of(detailed_results)

        # Log GEPA optimization trajectory
        round_data["gepa_details"] = {
            "num_candidates": len(detailed_results.candidates),
            "all_candidate_instructions": all_candidate_instructions,
            "val_aggregate_scores": detailed_results.val_aggregate_scores,
            "val_subscores": detailed_results.val_subscores,
            "parents": detailed_results.parents,
            "discovery_eval_counts": detailed_results.discovery_eval_counts,
            "best_candidate_idx": detailed_results.best_idx,
            "total_metric_calls": detailed_results.total_metric_calls,
            "num_full_val_evals": detailed_results.num_full_val_evals,
            # Iteration tracking
            "candidate_to_iteration": {int(k): int(v) for k, v in iteration_info["candidate_to_iteration"].items()},
            "iteration_to_candidate": {int(k): int(v) for k, v in iteration_info["iteration_to_candidate"].items()},
            "rejected_iterations": iteration_info["rejected_iterations"],
            "num_iterations": iteration_info["num_iterations"],
            "iteration_semantics": iteration_info.get("iteration_semantics", "optimizer iteration numbers"),
        }
        print(f"Round {round_num}: GEPA ran {iteration_info['num_iterations']} iterations")
        print(f"  Accepted: {len(detailed_results.candidates) - 1} candidates (excluding base)")
        print(f"  Rejected: {len(iteration_info['rejected_iterations'])} proposals")
        print(
            f"  Best candidate idx: {detailed_results.best_idx} (iteration {iteration_info['candidate_to_iteration'].get(detailed_results.best_idx, '?')}) with F1 score: {detailed_results.val_aggregate_scores[detailed_results.best_idx]:.4f}")

        # === LOG ALL PROMPTS TO FILES ===
        print(f"\nLogging prompts to files...")

        # Log base candidate prompt (iteration -1)
        base_instructions = all_candidate_instructions[0]
        base_score = detailed_results.val_aggregate_scores[0]
        log_prompt_to_file(log_dir, round_num, -1, 0, base_instructions, base_score)
        print(f"  Logged base prompt (candidate 0)")

        # Log all accepted candidate prompts
        iteration_to_candidate = iteration_info["iteration_to_candidate"]
        for iteration, candidate_idx in iteration_to_candidate.items():
            instructions = all_candidate_instructions[candidate_idx]
            score = detailed_results.val_aggregate_scores[candidate_idx]
            log_prompt_to_file(log_dir, round_num, iteration, candidate_idx, instructions, score)
            print(f"  Logged prompt for iteration {iteration}, candidate {candidate_idx}, score={score:.4f}")

        # Create prompt evolution markdown log
        print(f"\nCreating prompt evolution log...")
        evolution_file = create_prompt_evolution_log(log_dir, round_num, detailed_results)
        print(f"  Saved prompt evolution to: {evolution_file}")

        print(f"\n✓ Logged {len(iteration_to_candidate) + 1} prompts (1 base + {len(iteration_to_candidate)} candidates)")

    # Save round data
    round_file = log_dir / f"round_{round_num}.json"
    with open(round_file, 'w') as f:
        json.dump(round_data, f, indent=2)

    optimization_history.append(round_data)

    print(f"Round {round_num} complete. Total programs: {len(latest_program_list)}")
    print(f"Saved round {round_num} data to: {round_file}")

    write_summary()

    # Exit optimization loop if early stopping was triggered
    if early_stopped:
        print(f"\n{'!' * 60}")
        print(f"Breaking out of optimization loop due to early stopping")
        print(f"{'!' * 60}\n")
        break

# =============================================================================
# FINAL EVALUATION
# =============================================================================

print(f"\n{'=' * 60}")
print("=== Final Evaluation ===")
print(f"{'=' * 60}")

final_program = latest_program_list[-1]
final_evaluation = evaluate(final_program)
final_score = final_evaluation['score']
print(f"Final program F1 score: {final_score:.4f}")

# Log final program
final_instructions = extract_program_instructions(final_program)
final_data = {
    "round": "final",
    "score": float(final_score),
    "instructions": final_instructions,
    "timestamp": datetime.now().isoformat(),
}

final_file = log_dir / "final_program.json"
with open(final_file, 'w') as f:
    json.dump(final_data, f, indent=2)
print(f"Saved final program to: {final_file}")

write_summary(
    final_program_data=final_data,
)

print("\nOptimization complete")

# Final summary of prompt logs
print(f"\n{'=' * 60}")
print("=== Prompt Logging Summary ===")
print(f"{'=' * 60}")
prompt_dir = log_dir / "prompts"
if prompt_dir.exists():
    prompt_files = list(prompt_dir.glob("*.txt"))
    evolution_files = list(prompt_dir.glob("*_evolution.md"))

    print(f"\nPrompt log directory: {prompt_dir}")
    print(f"  Individual prompt files: {len(prompt_files)}")
    print(f"  Evolution logs: {len(evolution_files)}")

    print("\nPrompt files by round:")
    for round_idx in range(MAX_ROUNDS + 1):
        round_files = list(prompt_dir.glob(f"round_{round_idx}_*.txt"))
        if round_files:
            print(f"  Round {round_idx}: {len(round_files)} prompts")

    if evolution_files:
        print("\nEvolution logs (detailed timeline):")
        for evo_file in sorted(evolution_files):
            print(f"  - {evo_file.name}")

    print(f"\n💡 Tip: Open the evolution logs (*.md) for a readable summary of prompt changes")
else:
    print("\n⚠️  No prompt logs found")

print(f"\n{'=' * 60}")

workflow.cleanup_spark_sessions()

from workflow_sifta import hard_exit  # noqa: E402
hard_exit(0)
