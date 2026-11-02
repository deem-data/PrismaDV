"""Run the SIFTA optimization loop on EIDBench-synth.

The module constants below are the defaults; `--dataset`, `--sifta_seed`,
`--random_seed`, `--strategy` and `--max_rounds` override them for one run, so
workflow_sifta.run can drive this without editing it.
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import dspy
import numpy as np
from dspy.dsp.utils.settings import settings

from prismadv.llm_backend.model_routes import litellm_route
from prismadv.llm.dspy.models.column_wise_module import HumanDesignedConstraintGenerationModule, ConstraintGenerationModule
from prismadv.utils import (
    get_project_root,
    load_dotenv,
)
from sifta.dspy_sifta.teleprompt.sifta.sifta import SIFTA
from sifta.strategies.instruction_proposal import GlobalInstructionProposalSignature
from workflow_sifta.metrics import create_metric, create_metric_with_feedback
from workflow_sifta.metrics.optimization_metrics import METRIC_REV
from workflow_sifta.runtime import budget as opt_budget
from workflow_sifta._0_splits import load_split, make_split
from workflow_sifta.runtime.usage import (REFLECTION_LM, REFLECTION_SEED_OFFSET, TASK_LM, replicate_rollout_id,
                                          reset_usage, set_rollout_id, tracked_lm, usage_summary)
from workflow_sifta.optimization_fns import OptimizationWorkflow
from workflow_sifta.optimization_fns.optimization_labels import (NUM_BATCHES_SEED, SILENT_AS_SAFE_SEED, choose_batches,
                                                                 choose_silent_as_safe)

RANDOM_SEED = 1
SIFTA_SEED = 0

LLM_NAME = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 32000

REFLECTION_LLM_NAME = "gpt-5.2"
REFLECTION_LLM_TEMPERATURE = 0.6
REFLECTION_LLM_MAX_TOKENS = 32000

TRAIN_DATASET_NAME = "hr_analytics"
TRAIN_SUBTASK = "general_task"

INITIAL_PROGRAM_TYPE = "base"

# The split ratios live in workflow_sifta/_0_splits.py.
EVAL_SAMPLE_SIZE = 20
SPLIT_SEED = None
REPLICATE_SEED = 0

MAX_ROUNDS = 3

SIFTA_MAX_FULL_EVALS = 3
VALIDATION_PASSES = None
SIFTA_NUM_THREADS = 8
SIFTA_REFLECTION_MINIBATCH_SIZE = 3
SIFTA_SKIP_PERFECT_SCORE = True
SIFTA_USE_GLOBAL_PROPOSER = True
# Ablation ladder: "score" | "stats" | "assumptions" | "full"
SIFTA_FEEDBACK_LEVEL = "full"
# Hyperparameter: how many constraints each feedback shows in detail (the rest are counted only).
SIFTA_FEEDBACK_CONSTRAINTS = 5
# "Undiscovered silent failures" ablation: this fraction of the silent train and eval units is labelled safe for the
# optimization (workflow_sifta.optimization_fns.optimization_labels). 0: the true labels.
SILENT_AS_SAFE = 0.0
# Hyperparameter: how many data batches (error labels) the optimization observes; train and eval share them, as in
# the main arm (workflow_sifta.optimization_fns.optimization_labels.choose_batches). None: every train+eval label of
# the split. The scripts and the test units do not change.
NUM_BATCHES = None

BATCH_SAMPLER_TEMPERATURE = 0.5
BATCH_SAMPLER_STRATEGY = "worst_first"


def extract_program_instructions(module):
    """Extract all predictor instructions from a module."""
    instructions = {}
    for pred_name, pred in module.named_predictors():
        instructions[pred_name] = pred.signature.instructions
    return instructions


def create_condensed_set(eval_results: dict) -> list:
    """Filter evaluation results to keep only examples with non-NaN scores."""
    condensed = []
    for i in range(len(eval_results['results'])):
        example = eval_results['results'][i][0]
        score = eval_results['results'][i][2]
        if not np.isnan(score):
            condensed.append(example)
    return condensed


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help=f"train dataset (default {TRAIN_DATASET_NAME})")
    ap.add_argument("--sifta_seed", type=int, default=None,
                    help=f"SIFTA's own seed: candidate and minibatch sampling (default {SIFTA_SEED})")
    ap.add_argument("--random_seed", type=int, default=None,
                    help=f"split seed; arms compared against each other must share it (default {RANDOM_SEED})")
    ap.add_argument("--strategy", default=None, choices=["random", "worst_first", "best_first"],
                    help=f"batch sampler strategy (default {BATCH_SAMPLER_STRATEGY})")
    ap.add_argument("--initial_program", default=None, choices=["base", "human"],
                    help=f"program the optimization starts from (default {INITIAL_PROGRAM_TYPE})")
    ap.add_argument("--feedback_level", default=None, choices=["score", "stats", "assumptions", "full"],
                    help="how much of each failure the reflection LM sees: score only; + per-constraint "
                         "stats; + the source assumptions; + the code they came from "
                         f"(default {SIFTA_FEEDBACK_LEVEL})")
    ap.add_argument("--feedback_constraints", type=int, default=None,
                    help="how many constraints each feedback shows in detail "
                         f"(default {SIFTA_FEEDBACK_CONSTRAINTS}); worst_first and full feedback only")
    ap.add_argument("--silent_as_safe", type=float, default=None,
                    help="fraction of the silent train and eval units labelled safe for the optimization only "
                         f"(default {SILENT_AS_SAFE}); worst_first and full feedback only")
    ap.add_argument("--num_batches", type=int, default=None,
                    help="how many train+eval data batches the optimization observes, shared by train and eval "
                         "(default: every train+eval label); main setup only")
    ap.add_argument("--max_rounds", type=int, default=None, help=f"default {MAX_ROUNDS}")
    ap.add_argument("--llm", default=None, help=f"task LM, bare name (default {LLM_NAME})")
    ap.add_argument("--reflection_llm", default=None,
                    help=f"reflection LM, bare name (default {REFLECTION_LLM_NAME})")
    ap.add_argument("--split_seed", type=int, default=None,
                    help="replicate N: read benchmarks/EIDBench-synth/<dataset>/splits/split<N>.yaml and use N as "
                         "the optimizer seed too (replaces --random_seed and --sifta_seed)")
    ap.add_argument("--replicate_seed", type=int, default=None,
                    help="with --split_seed: k > 0 reruns the split with fresh LLM samples and minibatch draws "
                         f"(default {REPLICATE_SEED}, the original run)")
    ap.add_argument("--validation_passes", type=int, default=None,
                    help="budget: K validation passes, i.e. K x |eval set| program evaluations in total "
                         f"(workflow_sifta/runtime/budget.py); default: SIFTA's own max_full_evals={SIFTA_MAX_FULL_EVALS}")
    return ap.parse_args(argv)


def apply_overrides(args: argparse.Namespace) -> None:
    """Rebind the module constants the run reads, so the summary records what actually ran."""
    global TRAIN_DATASET_NAME, SIFTA_SEED, RANDOM_SEED, BATCH_SAMPLER_STRATEGY, MAX_ROUNDS
    global LLM_NAME, REFLECTION_LLM_NAME, SIFTA_FEEDBACK_LEVEL, INITIAL_PROGRAM_TYPE, VALIDATION_PASSES, SPLIT_SEED
    global SILENT_AS_SAFE, SIFTA_FEEDBACK_CONSTRAINTS, REPLICATE_SEED, NUM_BATCHES
    VALIDATION_PASSES = args.validation_passes
    SPLIT_SEED = args.split_seed
    if args.replicate_seed is not None:
        REPLICATE_SEED = args.replicate_seed
    if REPLICATE_SEED < 0 or (REPLICATE_SEED and SPLIT_SEED is None):
        raise ValueError("--replicate_seed must be >= 0 and needs --split_seed")
    if SPLIT_SEED is not None:
        if args.random_seed is not None or args.sifta_seed is not None:
            raise ValueError("--split_seed sets both seeds; leave out --random_seed and --sifta_seed")
        args.random_seed = SPLIT_SEED
        args.sifta_seed = replicate_rollout_id(SPLIT_SEED, REPLICATE_SEED)
    if args.initial_program is not None:
        INITIAL_PROGRAM_TYPE = args.initial_program
    if args.feedback_level is not None:
        SIFTA_FEEDBACK_LEVEL = args.feedback_level
    GlobalInstructionProposalSignature.set_feedback_level(SIFTA_FEEDBACK_LEVEL)
    if args.dataset is not None:
        TRAIN_DATASET_NAME = args.dataset
    if args.sifta_seed is not None:
        SIFTA_SEED = args.sifta_seed
    if args.random_seed is not None:
        RANDOM_SEED = args.random_seed
    if args.strategy is not None:
        BATCH_SAMPLER_STRATEGY = args.strategy
    if args.max_rounds is not None:
        MAX_ROUNDS = args.max_rounds
    if args.llm is not None:
        LLM_NAME = args.llm
    if args.reflection_llm is not None:
        REFLECTION_LLM_NAME = args.reflection_llm
    if args.silent_as_safe is not None:
        SILENT_AS_SAFE = args.silent_as_safe
    if not 0 <= SILENT_AS_SAFE <= 1:
        raise ValueError(f"--silent_as_safe must be in [0, 1], got {SILENT_AS_SAFE}")
    if SILENT_AS_SAFE > 0 and (BATCH_SAMPLER_STRATEGY, SIFTA_FEEDBACK_LEVEL) != ("worst_first", "full"):
        raise ValueError("--silent_as_safe is planned with the main setup only (worst_first, full feedback)")
    GlobalInstructionProposalSignature.set_silent_note(SILENT_AS_SAFE > 0)
    if args.feedback_constraints is not None:
        SIFTA_FEEDBACK_CONSTRAINTS = args.feedback_constraints
    if SIFTA_FEEDBACK_CONSTRAINTS < 1:
        raise ValueError(f"--feedback_constraints must be at least 1, got {SIFTA_FEEDBACK_CONSTRAINTS}")
    if SIFTA_FEEDBACK_CONSTRAINTS != 5 and (BATCH_SAMPLER_STRATEGY, SIFTA_FEEDBACK_LEVEL, SILENT_AS_SAFE) != (
            "worst_first", "full", 0.0):
        raise ValueError("--feedback_constraints is planned with the main setup only "
                         "(worst_first, full feedback, true labels)")
    if args.num_batches is not None:
        NUM_BATCHES = args.num_batches
    if NUM_BATCHES is not None and (BATCH_SAMPLER_STRATEGY, SIFTA_FEEDBACK_LEVEL, SILENT_AS_SAFE,
                                    SIFTA_FEEDBACK_CONSTRAINTS) != ("worst_first", "full", 0.0, 5):
        raise ValueError("--num_batches is planned with the main setup only "
                         "(worst_first, full feedback, true labels, 5 feedback constraints)")
    if NUM_BATCHES is not None and SPLIT_SEED is None:
        raise ValueError("--num_batches draws from a split file's train+eval labels and needs --split_seed")


def main(argv=None) -> int:
    """Split the benchmark, evaluate the initial program, then optimize it."""
    apply_overrides(parse_args(argv))
    print(f"dataset={TRAIN_DATASET_NAME} strategy={BATCH_SAMPLER_STRATEGY} "
          f"random_seed={RANDOM_SEED} sifta_seed={SIFTA_SEED} max_rounds={MAX_ROUNDS} "
          f"llm={LLM_NAME} reflection_llm={REFLECTION_LLM_NAME} feedback_level={SIFTA_FEEDBACK_LEVEL}")
    settings.max_errors = 10
    load_dotenv()

    reset_usage()
    # replicate N: its own cached answers
    set_rollout_id(None if SPLIT_SEED is None else replicate_rollout_id(SPLIT_SEED, REPLICATE_SEED))
    lm = tracked_lm(litellm_route(LLM_NAME), temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS,
                    usage_role=TASK_LM)
    dspy.configure(lm=lm)

    dataset_subtasks = {TRAIN_DATASET_NAME: [TRAIN_SUBTASK]}

    if SPLIT_SEED is not None:
        split = load_split(TRAIN_DATASET_NAME, SPLIT_SEED)
        random.seed(SPLIT_SEED)  # draws the eval sample below
    else:
        # The split drawn from RANDOM_SEED; make_split leaves `random` in the state the eval sample is drawn from.
        split = make_split(TRAIN_DATASET_NAME, RANDOM_SEED)
    train_eval_label_list = split["train_eval_labels"]
    cross_new_data_test_label_list = split["test_labels"]
    train_script_name_list = split["train_scripts"]
    eval_script_name_list = split["eval_scripts"]
    train_eval_script_name_list = train_script_name_list + eval_script_name_list
    cross_script_test_script_name_list = split["test_scripts"]

    print(
        f"Label split: train_eval={len(train_eval_label_list)}, cross_new_data_test={len(cross_new_data_test_label_list)}")
    print(f"  Train/Eval labels: {train_eval_label_list}")
    print(f"  Cross-new-data test labels: {cross_new_data_test_label_list}")

    # The batches the optimization observes, only ever from the split's train+eval labels (the test labels stay
    # unseen). Drawn with their own RNG: the global `random` state, which draws the eval sample, stays unchanged.
    if NUM_BATCHES is None:
        train_label_list = eval_label_list = train_eval_label_list
    else:
        train_label_list = eval_label_list = choose_batches(train_eval_label_list, NUM_BATCHES)
        print(f"num_batches={NUM_BATCHES}: train and eval batches {train_label_list}")

    # The train labels drive the condense pass and the batch sampler (both see the train set only); every example is
    # scored on its own labels (workflow_sifta.metrics.optimization_metrics.example_labels).
    workflow = OptimizationWorkflow(
        train_dataset_subtasks=dataset_subtasks,
        train_processed_data_label_list=train_label_list,
        val_processed_data_label_list=eval_label_list,
        project_root=get_project_root(),
    )

    dataset_name = list(dataset_subtasks.keys())[0]
    subtask_name = dataset_subtasks[dataset_name][0]

    # Train and eval units are drawn apart, so each set relabels its own fraction of its silent units.
    silent_as_safe_units = {
        part: choose_silent_as_safe(workflow.project_managers[dataset_name], dataset_name, subtask_name, scripts,
                                    labels, SILENT_AS_SAFE) if SILENT_AS_SAFE > 0 else []
        for part, scripts, labels in (("train", train_script_name_list, train_label_list),
                                      ("eval", eval_script_name_list, eval_label_list))}
    workflow.silent_as_safe = frozenset(u for units in silent_as_safe_units.values() for u in units)
    if SILENT_AS_SAFE > 0:
        print(f"silent_as_safe={SILENT_AS_SAFE}: relabelled as safe "
              f"{ {part: len(units) for part, units in silent_as_safe_units.items()} } silent units")

    print(
        f"\nScript split: train_eval={len(train_eval_script_name_list)}, new_script_test={len(cross_script_test_script_name_list)}")
    print(f"  Train scripts ({len(train_script_name_list)}): {train_script_name_list}")
    print(f"  Eval scripts ({len(eval_script_name_list)}): {eval_script_name_list}")
    print(f"  New script test scripts ({len(cross_script_test_script_name_list)}): {cross_script_test_script_name_list}")

    new_script_name_list = cross_script_test_script_name_list

    print(f"\n=== Test Set Configuration ===")
    print(
        f"Test 1 - Cross new data: train+eval scripts ({len(train_eval_script_name_list)}) + new data labels ({len(cross_new_data_test_label_list)})")
    print(
        f"Test 2 - New script + train data: new scripts ({len(new_script_name_list)}) + train data labels ({len(train_eval_label_list)})")
    print(
        f"Test 3 - New script + new data: new scripts ({len(new_script_name_list)}) + new data labels ({len(cross_new_data_test_label_list)})")
    print(f"  New script list: {new_script_name_list}")

    print("\nPreparing datasets...")
    print("Preparing train dataset (will be condensed in optimization loop)...")

    initial_train_set = workflow.prepare_single_column_training_dataset(
        script_name_list=train_script_name_list,
        processed_data_label="0",
        new_processed_data_label_list=train_label_list,
    )
    print(f"Initial train set size: {len(initial_train_set)}")

    print("Preparing eval dataset (sampling, no condensing)...")

    eval_set_all = workflow.prepare_single_column_training_dataset(
        script_name_list=eval_script_name_list,
        processed_data_label="0",
        new_processed_data_label_list=eval_label_list,
    )

    if len(eval_set_all) > EVAL_SAMPLE_SIZE:
        eval_set = random.sample(eval_set_all, EVAL_SAMPLE_SIZE)
    else:
        eval_set = eval_set_all

    print(f"Eval set size: {len(eval_set)} (sampled from {len(eval_set_all)} total)")

    if INITIAL_PROGRAM_TYPE == "human":
        initial_program = HumanDesignedConstraintGenerationModule()
        print(f"Using HumanDesignedConstraintGenerationModule as initial program")
    elif INITIAL_PROGRAM_TYPE == "base":
        initial_program = ConstraintGenerationModule()
        print(f"Using ConstraintGenerationModule (base) as initial program")
    else:
        raise ValueError(f"Invalid INITIAL_PROGRAM_TYPE: {INITIAL_PROGRAM_TYPE}. Must be 'base' or 'human'")

    metric_for_condense = create_metric(
        workflow=workflow,
        llm_name=LLM_NAME,
        convert_nan_to_zero=False,
    )

    metric = create_metric(
        workflow=workflow,
        llm_name=LLM_NAME,
        convert_nan_to_zero=True,
    )

    metric_with_feedback = create_metric_with_feedback(
        workflow=workflow,
        llm_name=LLM_NAME,
        convert_nan_to_zero=True,
        feedback_level=SIFTA_FEEDBACK_LEVEL,
        max_constraints=SIFTA_FEEDBACK_CONSTRAINTS,
    )

    evaluate = dspy.Evaluate(
        devset=eval_set,
        metric=metric,
        num_threads=SIFTA_NUM_THREADS,
        display_table=False,
        display_progress=True,
        provide_traceback=False,
    )

    # Nested rather than module level: it reads the split lists, the log dir and
    # the running history, which are this function's locals now that the work no
    # longer happens at import time.
    def write_summary(
            condensed_train_size: int | None,
            final_program_data: dict | None = None,
    ):
        """Write optimization summary to JSON file."""
        summary = {
            "run_timestamp": run_timestamp,
            "max_rounds": MAX_ROUNDS,
            "split_seed": SPLIT_SEED,
            "replicate_seed": REPLICATE_SEED,
            "random_seed": RANDOM_SEED,
            "sifta_seed": SIFTA_SEED,
            "llm_name": LLM_NAME,
            "reflection_llm_name": REFLECTION_LLM_NAME,
            "initial_program_type": INITIAL_PROGRAM_TYPE,
            "initial_program_instructions": extract_program_instructions(initial_program),
            "train_dataset_name": TRAIN_DATASET_NAME,
            "dataset_subtasks": dataset_subtasks,
            "all_processed_data_labels": train_eval_label_list + cross_new_data_test_label_list,
            "train_eval_label_list": train_eval_label_list,
            "cross_new_data_test_label_list": cross_new_data_test_label_list,
            "train_eval_script_name_list": train_eval_script_name_list,
            "train_script_name_list": train_script_name_list,
            "eval_script_name_list": eval_script_name_list,
            "new_script_name_list": new_script_name_list,
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
            "initial_train_set_size": len(initial_train_set),
            "eval_set_size": len(eval_set),
            "eval_set_all_size": len(eval_set_all),
            "condensed_train_set_size": condensed_train_size,
            "batch_sampler_temperature": BATCH_SAMPLER_TEMPERATURE,
            "batch_sampler_strategy": BATCH_SAMPLER_STRATEGY,
            "use_global_proposer": SIFTA_USE_GLOBAL_PROPOSER,
            "feedback_level": SIFTA_FEEDBACK_LEVEL,
            "feedback_constraints": SIFTA_FEEDBACK_CONSTRAINTS,
            "reflection_prompt_feedback_level": GlobalInstructionProposalSignature.feedback_level,
            "reflection_prompt_silent_note": GlobalInstructionProposalSignature.silent_note,
            "silent_as_safe": SILENT_AS_SAFE,
            "silent_as_safe_seed": SILENT_AS_SAFE_SEED,
            "silent_as_safe_units": {part: [list(u) for u in units] for part, units in silent_as_safe_units.items()},
            "num_batches": NUM_BATCHES,
            "num_batches_seed": NUM_BATCHES_SEED,
            "train_label_list": train_label_list,
            "eval_label_list": eval_label_list,
            "metric_rev": METRIC_REV,
            "initial_program_evaluation": initial_program_data,
            "optimization_history": optimization_history,
            # Real cost of the run, for matching a baseline's budget against it.
            "total_metric_computations": sum(
                r.get("metric_computations", {}).get("round_total", 0) for r in optimization_history
            ),
            # Tokens, cost, calls and API latency for the whole run, split by
            # task LM and reflection LM.
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

    run_started = time.time()
    while True:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(get_project_root()) / "optimization_runs" / f"run_{run_timestamp}"
        try:
            log_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            time.sleep(1)
    print(f"Logging optimization history to: {log_dir}")

    optimization_history = []
    # Accumulates the fail-precision sampler's precompute cost, which happens inside
    # optimize() and is not covered by SIFTA's max_metric_calls.
    budget_tracker = {}
    latest_program_list = [initial_program]

    # Every evaluation from here on counts toward the budget, the initial evaluation and condense passes included.
    budget = opt_budget.install(VALIDATION_PASSES, len(eval_set))

    print("\nEvaluating initial program...")
    initial_evaluation = evaluate(initial_program)
    initial_score = initial_evaluation['score']
    print(f"Initial program eval score: {initial_score:.4f}")

    initial_program_data = {
        "round": -1,
        "score": float(initial_score),
        "instructions": extract_program_instructions(initial_program),
        "timestamp": datetime.now().isoformat(),
    }

    initial_file = log_dir / "round_-1_initial.json"
    with open(initial_file, 'w') as f:
        json.dump(initial_program_data, f, indent=2)
    print(f"Saved initial program to: {initial_file}")

    write_summary(condensed_train_size=None)

    round_num = 0
    condensed_train_set = []

    # The budget is split evenly over the rounds' engine runs; the condense pass, the evaluation of the current program
    # and the sampler precompute run outside the engine and are recorded but not charged (workflow_sifta/runtime/budget.py).
    passes_per_round = None
    if VALIDATION_PASSES is not None:
        if VALIDATION_PASSES % MAX_ROUNDS:
            raise ValueError(f"--validation_passes {VALIDATION_PASSES} does not split evenly over {MAX_ROUNDS} rounds")
        passes_per_round = VALIDATION_PASSES // MAX_ROUNDS

    while round_num < MAX_ROUNDS and not budget.exhausted:
        budget.start_round(passes_per_round)
        print(f"\n{'=' * 60}")
        print(f"=== Round {round_num + 1}/{MAX_ROUNDS} ===")
        print(f"{'=' * 60}")

        current_program = latest_program_list[-1]

        print("Creating condensed train set...")
        evaluate_for_condense = dspy.Evaluate(
            devset=initial_train_set,
            metric=metric_for_condense,
            num_threads=SIFTA_NUM_THREADS,
            display_table=False,
            display_progress=True,
            provide_traceback=False,
        )
        condense_results = evaluate_for_condense(current_program)
        condensed_train_set = create_condensed_set(condense_results)
        print(f"Condensed train set size: {len(condensed_train_set)}/{len(initial_train_set)}")

        print("Evaluating on eval set...")
        evaluation_results = evaluate(current_program)
        current_score = evaluation_results['score']
        print(f"Current program eval score: {current_score}")

        current_instructions = extract_program_instructions(current_program)
        # A round that starts from the program the previous round started from would send the reflection LM the
        # same requests and, from DSPy's cache, get the same proposals back. Each round's reflection LM therefore
        # has its own rollout_id (the task LM keeps the replicate's, so a program's evaluations stay the same).
        reflection_rollout = {} if SPLIT_SEED is None else {
            "rollout_id": SPLIT_SEED * 100 + round_num + 1 + REFLECTION_SEED_OFFSET * REPLICATE_SEED}
        round_data = {
            "round": round_num,
            "score": float(current_score),
            "instructions": current_instructions,
            "reflection_rollout_id": reflection_rollout.get("rollout_id"),
            "timestamp": datetime.now().isoformat(),
        }

        optimizer = SIFTA(
            metric=metric_with_feedback,
            **opt_budget.engine_budget_kwargs(VALIDATION_PASSES, {"max_full_evals": SIFTA_MAX_FULL_EVALS}),
            num_threads=SIFTA_NUM_THREADS,
            track_stats=True,
            reflection_minibatch_size=SIFTA_REFLECTION_MINIBATCH_SIZE,
            skip_perfect_score=SIFTA_SKIP_PERFECT_SCORE,
            reflection_lm=tracked_lm(
                model=litellm_route(REFLECTION_LLM_NAME),
                temperature=REFLECTION_LLM_TEMPERATURE,
                max_tokens=REFLECTION_LLM_MAX_TOKENS,
                usage_role=REFLECTION_LM,
                **reflection_rollout,
            ),
            use_global_proposer=SIFTA_USE_GLOBAL_PROPOSER,
            workflow=workflow,
            llm_name=LLM_NAME,
            batch_sampler_temperature=BATCH_SAMPLER_TEMPERATURE,
            batch_sampler_strategy=BATCH_SAMPLER_STRATEGY,
            budget_tracker=budget_tracker,
            seed=SIFTA_SEED,
        )

        metric_stats_before = dict(metric_with_feedback.stats)
        precompute_before = budget_tracker.get("sampler_precompute_computations", 0)

        optimized_program = optimizer.compile(
            current_program,
            trainset=condensed_train_set,
            valset=eval_set,
        )
        latest_program_list.append(optimized_program)
        round_num += 1

        # SIFTA's own total_metric_calls counts only the minibatch and valset
        # evaluations inside the search loop. Record what the round actually cost so
        # budget comparisons against the baselines are made on real numbers.
        search_loop = metric_with_feedback.stats["computations"] - metric_stats_before["computations"]
        precompute = budget_tracker.get("sampler_precompute_computations", 0) - precompute_before
        condense_pass = len(initial_train_set)
        # The engine run of this round: charged evaluations and its cap.
        round_data["budget"] = budget.engine_runs[-1] if budget.engine_runs else None
        round_data["metric_computations"] = {
            "search_loop": search_loop,
            "served_from_cache": metric_with_feedback.stats["cache_hits"] - metric_stats_before["cache_hits"],
            "sampler_precompute": precompute,
            "condense_pass": condense_pass,
            "round_total": search_loop + precompute + condense_pass,
        }
        print(
            f"Round {round_num} true cost: {search_loop} search-loop + {precompute} precompute + "
            f"{condense_pass} condense = {search_loop + precompute + condense_pass} constraint validations "
            f"({round_data['metric_computations']['served_from_cache']} served from cache)"
        )

        optimized_instructions = extract_program_instructions(optimized_program)
        round_data["optimized_instructions"] = optimized_instructions

        if hasattr(optimized_program, 'detailed_results'):
            detailed_results = optimized_program.detailed_results

            all_candidate_instructions = []
            for candidate_module in detailed_results.candidates:
                candidate_instructions = extract_program_instructions(candidate_module)
                all_candidate_instructions.append(candidate_instructions)

            iteration_info = detailed_results.get_iteration_info()

            round_data["sifta_details"] = {
                "num_candidates": len(detailed_results.candidates),
                "all_candidate_instructions": all_candidate_instructions,
                "val_aggregate_scores": detailed_results.val_aggregate_scores,
                "val_subscores": detailed_results.val_subscores,
                "parents": detailed_results.parents,
                "discovery_eval_counts": detailed_results.discovery_eval_counts,
                "best_candidate_idx": detailed_results.best_idx,
                "total_metric_calls": detailed_results.total_metric_calls,
                "num_full_val_evals": detailed_results.num_full_val_evals,
                "candidate_to_iteration": {int(k): int(v) for k, v in iteration_info["candidate_to_iteration"].items()},
                "iteration_to_candidate": {int(k): int(v) for k, v in iteration_info["iteration_to_candidate"].items()},
                "rejected_iterations": iteration_info["rejected_iterations"],
                "num_iterations": iteration_info["num_iterations"],
            }
            print(f"Round {round_num}: SIFTA ran {iteration_info['num_iterations']} iterations")
            print(f"  Accepted: {len(detailed_results.candidates) - 1} candidates (excluding base)")
            print(f"  Rejected: {len(iteration_info['rejected_iterations'])} proposals")
            print(
                f"  Best candidate idx: {detailed_results.best_idx} (iteration {iteration_info['candidate_to_iteration'].get(detailed_results.best_idx, '?')}) with score: {detailed_results.val_aggregate_scores[detailed_results.best_idx]:.4f}")

        round_file = log_dir / f"round_{round_num}.json"
        with open(round_file, 'w') as f:
            json.dump(round_data, f, indent=2)

        optimization_history.append(round_data)

        print(f"Round {round_num} complete. Total programs: {len(latest_program_list)}")
        print(f"Saved round {round_num} data to: {round_file}")

        write_summary(
            condensed_train_size=len(condensed_train_set),
        )

    print(f"\n{'=' * 60}")
    print("=== Final Evaluation ===")
    print(f"{'=' * 60}")

    final_program = latest_program_list[-1]
    final_evaluation = evaluate(final_program)
    final_score = final_evaluation['score']
    print(f"Final program score: {final_score}")

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
        condensed_train_size=len(condensed_train_set),
        final_program_data=final_data,
    )

    print("\nOptimization complete")

    workflow.cleanup_spark_sessions()
    return 0


if __name__ == "__main__":
    from workflow_sifta import hard_exit
    hard_exit(main())
