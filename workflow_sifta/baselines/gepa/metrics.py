"""Metrics for script-level DSPy baseline optimization."""

from __future__ import annotations

import random
from typing import Any, Dict, List

import oyaml as yaml
from workflow_sifta.optimization_fns import OptimizationWorkflow

from prismadv.data_models.constraints_v2 import ConstraintsWithSources, ser_column_group_key
from prismadv.data_models.validated_results import ValidationResults
from sifta.dspy_sifta.teleprompt.sifta.sifta_utils import ScoreWithFeedback
from workflow_prismadv.results.metrics import macro_scores


def _sample_validation_entries(
        validation_results_by_label: Dict[str, ValidationResults],
        sample_size: int,
        rng: random.Random,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for label, validation_results in validation_results_by_label.items():
        for column_group, column_result in validation_results.results.items():
            column_key = ser_column_group_key(column_group)
            for entry in column_result.code:
                entries.append(
                    {
                        "label": label,
                        "column_group": column_key,
                        "suggestion": entry.suggestion,
                        "status": entry.status,
                        "reason_if_failed": entry.reason_if_failed,
                        "level": entry.level,
                    }
                )
    rng.shuffle(entries)
    return entries[:sample_size]


def _validate_on_labels(workflow: OptimizationWorkflow, example, pred: ConstraintsWithSources):
    """Validation results per label and the (safe, predicted safe) pairs of the script's labels."""
    dataset_name = example["dataset_name"]
    subtask_name = example["subtask_name"]
    constraints_with_sources = workflow.validate_constraints_on_training_data(
        dataset_name=dataset_name,
        subtask_name=subtask_name,
        constraints_with_sources=pred,
    )
    validation_results_by_label: Dict[str, ValidationResults] = {}
    safe, predicted_safe = [], []
    for processed_data_label, is_safe in example["new_data_safety"].items():
        validation_results = workflow.validate_constraints_on_test_data(
            dataset_name=dataset_name,
            subtask_name=subtask_name,
            constraints_with_sources=constraints_with_sources,
            processed_data_label=processed_data_label,
            clean=False,
        )
        validation_results_by_label[processed_data_label] = validation_results
        safe.append(bool(is_safe))
        predicted_safe.append(validation_results.check_result()[2] == 0)
    return validation_results_by_label, safe, predicted_safe


def _outcome_lines(safe: List[bool], predicted_safe: List[bool]) -> str:
    caught = sum(not s and not p for s, p in zip(safe, predicted_safe))
    false_alarms = sum(s and not p for s, p in zip(safe, predicted_safe))
    return (f"- Unsafe batches flagged (errors caught): {caught} of {safe.count(False)}\n"
            f"- Safe batches flagged (false alarms): {false_alarms} of {safe.count(True)}\n")


def create_script_level_metric(workflow: OptimizationWorkflow):
    """Metric returning the per-script macro F1 across labels."""

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        if not isinstance(pred, ConstraintsWithSources):
            return 0.0
        if not example.get("new_data_safety", {}):
            return 0.0
        _, safe, predicted_safe = _validate_on_labels(workflow, example, pred)
        return macro_scores(safe, predicted_safe)["f1"]

    return metric


def create_script_level_metric_with_feedback(
        workflow: OptimizationWorkflow,
        feedback_sample_size: int = 5,
        feedback_seed: int = 0,
):
    """Metric with sampled validation feedback for SIFTA."""

    rng = random.Random(feedback_seed)

    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        if not isinstance(pred, ConstraintsWithSources):
            return ScoreWithFeedback(score=0.0, feedback="No constraints were generated.")
        if not example.get("new_data_safety", {}):
            return ScoreWithFeedback(score=0.0, feedback="No safety labels available for this script.")

        validation_results_by_label, safe, predicted_safe = _validate_on_labels(workflow, example, pred)
        f1 = macro_scores(safe, predicted_safe)["f1"]

        samples = _sample_validation_entries(validation_results_by_label, feedback_sample_size, rng)
        samples_text = "(no validation entries)"
        if samples:
            samples_text = yaml.dump(samples, sort_keys=False)

        feedback = (
            "Script-level feedback:\n"
            f"- Macro F1: {f1:.4f}\n"
            + _outcome_lines(safe, predicted_safe)
            + f"- Validation samples (max {feedback_sample_size}):\n{samples_text}"
        )

        return ScoreWithFeedback(score=f1, feedback=feedback)

    return metric_with_feedback
