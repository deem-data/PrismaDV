import json
import random
from datetime import datetime
from pathlib import Path

import dspy
import numpy as np
from dspy.dsp.utils.settings import settings

from prismadv.llm.dspy.models.column_wise_module import HumanDesignedConstraintGenerationModule, ConstraintGenerationModule
from prismadv.utils import (
    get_project_root,
    load_dotenv,
)
from sifta.dspy_sifta.teleprompt.sifta.sifta import SIFTA
from workflow_sifta.metrics import create_metric, create_metric_with_feedback
from workflow_sifta.optimization_fns import OptimizationWorkflow

RANDOM_SEED = 1

LLM_NAME = "gpt-4.1-mini"
LLM_TEMPERATURE = 1
LLM_MAX_TOKENS = 32000

REFLECTION_LLM_NAME = "gpt-5.2"
REFLECTION_LLM_TEMPERATURE = 1.0
REFLECTION_LLM_MAX_TOKENS = 32000

TRAIN_DATASET_NAME = "hr_analytics"
TRAIN_SUBTASK = "general_task"

INITIAL_PROGRAM_TYPE = "base"

LABEL_SPLIT_RATIO = 0.5
SCRIPT_SPLIT_RATIO = 0.6
TRAIN_EVAL_SCRIPT_SPLIT_RATIO = 0.5
EVAL_SAMPLE_SIZE = 20

MAX_ROUNDS = 3

SIFTA_MAX_FULL_EVALS = 3
SIFTA_NUM_THREADS = 8
SIFTA_REFLECTION_MINIBATCH_SIZE = 3
SIFTA_SKIP_PERFECT_SCORE = True
SIFTA_USE_GLOBAL_PROPOSER = True

BATCH_SAMPLER_TEMPERATURE = 0.5
BATCH_SAMPLER_STRATEGY = "worst_first"

settings.max_errors = 10
load_dotenv()

lm = dspy.LM(f"openai/{LLM_NAME}", temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
dspy.configure(lm=lm)

dataset_subtasks = {TRAIN_DATASET_NAME: [TRAIN_SUBTASK]}

all_processed_data_labels = [str(i) for i in range(1, 26)]

random.seed(RANDOM_SEED)
shuffled_labels = all_processed_data_labels.copy()
random.shuffle(shuffled_labels)

train_eval_label_count = int(len(shuffled_labels) * LABEL_SPLIT_RATIO)
train_eval_label_list = shuffled_labels[:train_eval_label_count]
cross_new_data_test_label_list = shuffled_labels[train_eval_label_count:]

print(
    f"Label split: train_eval={len(train_eval_label_list)}, cross_new_data_test={len(cross_new_data_test_label_list)}")
print(f"  Train/Eval labels: {train_eval_label_list}")
print(f"  Cross-new-data test labels: {cross_new_data_test_label_list}")

workflow = OptimizationWorkflow(
    train_dataset_subtasks=dataset_subtasks,
    train_processed_data_label_list=train_eval_label_list,
    project_root=get_project_root(),
)

dataset_name = list(dataset_subtasks.keys())[0]
subtask_name = dataset_subtasks[dataset_name][0]
all_script_names = workflow.get_available_script_names_for_subtask(dataset_name, subtask_name)

print(f"Found {len(all_script_names)} scripts for subtask '{subtask_name}'")

shuffled_scripts = all_script_names.copy()
random.shuffle(shuffled_scripts)

train_eval_script_count = int(len(shuffled_scripts) * SCRIPT_SPLIT_RATIO)
train_eval_script_name_list = shuffled_scripts[:train_eval_script_count]
cross_script_test_script_name_list = shuffled_scripts[train_eval_script_count:]

train_script_count = int(len(train_eval_script_name_list) * TRAIN_EVAL_SCRIPT_SPLIT_RATIO)
train_script_name_list = train_eval_script_name_list[:train_script_count]
eval_script_name_list = train_eval_script_name_list[train_script_count:]

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
    new_processed_data_label_list=train_eval_label_list,
)
print(f"Initial train set size: {len(initial_train_set)}")

print("Preparing eval dataset (sampling, no condensing)...")

eval_set_all = workflow.prepare_single_column_training_dataset(
    script_name_list=eval_script_name_list,
    processed_data_label="0",
    new_processed_data_label_list=train_eval_label_list,
)

if len(eval_set_all) > EVAL_SAMPLE_SIZE:
    eval_set = random.sample(eval_set_all, EVAL_SAMPLE_SIZE)
else:
    eval_set = eval_set_all

print(f"Eval set size: {len(eval_set)} (sampled from {len(eval_set_all)} total)")

if INITIAL_PROGRAM_TYPE == "human_designed":
    initial_program = HumanDesignedConstraintGenerationModule()
    print(f"Using HumanDesignedConstraintGenerationModule as initial program")
elif INITIAL_PROGRAM_TYPE == "base":
    initial_program = ConstraintGenerationModule()
    print(f"Using ConstraintGenerationModule (base) as initial program")
else:
    raise ValueError(f"Invalid INITIAL_PROGRAM_TYPE: {INITIAL_PROGRAM_TYPE}. Must be 'human_designed' or 'base'")

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
)

evaluate_for_condense = None

evaluate = dspy.Evaluate(
    devset=eval_set,
    metric=metric,
    num_threads=SIFTA_NUM_THREADS,
    display_table=False,
    display_progress=True,
    provide_traceback=False,
)


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


def write_summary(
        condensed_train_size: int | None,
        final_program_data: dict | None = None,
):
    """Write optimization summary to JSON file."""
    summary = {
        "run_timestamp": run_timestamp,
        "max_rounds": MAX_ROUNDS,
        "llm_name": LLM_NAME,
        "initial_program_type": INITIAL_PROGRAM_TYPE,
        "initial_program_instructions": extract_program_instructions(initial_program),
        "train_dataset_name": TRAIN_DATASET_NAME,
        "dataset_subtasks": dataset_subtasks,
        "all_processed_data_labels": all_processed_data_labels,
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
        "initial_program_evaluation": initial_program_data if 'initial_program_data' in globals() else None,
        "optimization_history": optimization_history,
    }
    if final_program_data is not None:
        summary["final_program"] = final_program_data

    summary_file = log_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved optimization summary to: {summary_file}")


run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path(get_project_root()) / "optimization_runs" / f"run_{run_timestamp}"
log_dir.mkdir(parents=True, exist_ok=True)
print(f"Logging optimization history to: {log_dir}")

optimization_history = []
latest_program_list = [initial_program]

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

while round_num < MAX_ROUNDS:
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
    round_data = {
        "round": round_num,
        "score": float(current_score),
        "instructions": current_instructions,
        "timestamp": datetime.now().isoformat(),
    }

    optimizer = SIFTA(
        metric=metric_with_feedback,
        max_full_evals=SIFTA_MAX_FULL_EVALS,
        num_threads=SIFTA_NUM_THREADS,
        track_stats=True,
        reflection_minibatch_size=SIFTA_REFLECTION_MINIBATCH_SIZE,
        skip_perfect_score=SIFTA_SKIP_PERFECT_SCORE,
        reflection_lm=dspy.LM(
            model=f"openai/{REFLECTION_LLM_NAME}",
            temperature=REFLECTION_LLM_TEMPERATURE,
            max_tokens=REFLECTION_LLM_MAX_TOKENS,
        ),
        use_global_proposer=SIFTA_USE_GLOBAL_PROPOSER,
        workflow=workflow,
        llm_name=LLM_NAME,
        batch_sampler_temperature=BATCH_SAMPLER_TEMPERATURE,
        batch_sampler_strategy=BATCH_SAMPLER_STRATEGY,
    )

    optimized_program = optimizer.compile(
        current_program,
        trainset=condensed_train_set,
        valset=eval_set,
    )
    latest_program_list.append(optimized_program)
    round_num += 1

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
