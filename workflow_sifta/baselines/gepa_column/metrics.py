"""Column-level F1 metrics for GEPA baseline optimization."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import oyaml as yaml

from prismadv.data_models.constraints_v2 import ConstraintsWithSources
from prismadv.data_models.validated_results import ValidationResults
from sifta.adapters.dspy_adapter.dspy_adapter import ScoreWithFeedback
from workflow_sifta.optimization_fns import OptimizationWorkflow


def _compute_f1(tp: int, fp: int, fn: int) -> float:
    """Compute F1 score from confusion matrix."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) > 0:
        return 2 * precision * recall / (precision + recall)
    return float("nan")


def _sample_validation_entries(
    validation_results_by_label: Dict[str, ValidationResults],
    sample_size: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Sample validation entries for feedback."""
    entries: List[Dict[str, Any]] = []
    for label, validation_results in validation_results_by_label.items():
        for column_group, column_result in validation_results.results.items():
            from prismadv.data_models.constraints_v2 import ser_column_group_key
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


def create_column_level_f1_metric(
    workflow: OptimizationWorkflow,
    convert_nan_to_zero: bool = True,
):
    """
    Create a column-level F1 metric for DSPy evaluation.

    This metric evaluates constraints for a single column and computes F1 score
    based on anomaly detection performance across multiple error labels.

    Args:
        workflow: OptimizationWorkflow instance
        convert_nan_to_zero: Whether to convert NaN F1 to 0.0 (default: True)

    Returns:
        Metric function that returns F1 score
    """

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        # Check if pred is valid (should be dict with "assumptions" and "code" keys)
        if not isinstance(pred, dict) or "code" not in pred:
            return 0.0

        column_name = example["target_column"]
        dataset_name = example["dataset_name"]
        subtask_name = example["subtask_name"]
        new_data_safety = example.get("new_data_safety", {})

        if not new_data_safety:
            return 0.0

        # Combine constraints for this single column
        single_column_results = {column_name: pred}
        constraints_with_sources = workflow.combine_constraints(
            single_column_results=single_column_results,
        )

        # Validate constraints on training data
        constraints_with_sources = workflow.validate_constraints_on_training_data(
            dataset_name=dataset_name,
            subtask_name=subtask_name,
            constraints_with_sources=constraints_with_sources,
        )

        # Compute confusion matrix across all error labels
        tp = fp = tn = fn = 0
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
            if not is_safe and not predicted_safe:
                tp += 1  # TP: error detected correctly
            elif not is_safe and predicted_safe:
                fn += 1  # FN: error missed (BAD!)
            elif is_safe and not predicted_safe:
                fp += 1  # FP: false alarm
            else:  # is_safe and predicted_safe
                tn += 1  # TN: safe data accepted correctly

        f1 = _compute_f1(tp, fp, fn)
        if convert_nan_to_zero and not math.isfinite(f1):
            f1 = 0.0
        return f1

    return metric


def create_column_level_f1_metric_with_feedback(
    workflow: OptimizationWorkflow,
    feedback_sample_size: int = 5,
    feedback_seed: int = 0,
    convert_nan_to_zero: bool = True,
):
    """
    Create a column-level F1 metric with feedback for GEPA optimization.

    This metric returns ScoreWithFeedback containing F1 score and detailed
    feedback about constraint performance.

    Args:
        workflow: OptimizationWorkflow instance
        feedback_sample_size: Number of validation entries to sample for feedback
        feedback_seed: Random seed for feedback sampling
        convert_nan_to_zero: Whether to convert NaN F1 to 0.0 (default: True)

    Returns:
        Metric function that returns ScoreWithFeedback
    """

    rng = random.Random(feedback_seed)

    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        # Check if pred is valid
        if not isinstance(pred, dict) or "code" not in pred:
            return ScoreWithFeedback(score=0.0, feedback="No constraints were generated.")

        column_name = example["target_column"]
        dataset_name = example["dataset_name"]
        subtask_name = example["subtask_name"]
        script_name = example["script_name"]
        new_data_safety = example.get("new_data_safety", {})

        if not new_data_safety:
            return ScoreWithFeedback(score=0.0, feedback="No safety labels available for this example.")

        # Combine constraints for this single column
        single_column_results = {column_name: pred}
        constraints_with_sources = workflow.combine_constraints(
            single_column_results=single_column_results,
        )

        # Validate constraints on training data
        constraints_with_sources = workflow.validate_constraints_on_training_data(
            dataset_name=dataset_name,
            subtask_name=subtask_name,
            constraints_with_sources=constraints_with_sources,
        )

        # Compute confusion matrix and collect validation results
        validation_results_by_label: Dict[str, ValidationResults] = {}
        tp = fp = tn = fn = 0
        for processed_data_label, is_safe in new_data_safety.items():
            validation_results = workflow.validate_constraints_on_test_data(
                dataset_name=dataset_name,
                subtask_name=subtask_name,
                constraints_with_sources=constraints_with_sources,
                processed_data_label=processed_data_label,
                clean=False,
            )
            validation_results_by_label[processed_data_label] = validation_results
            num_failed_error = validation_results.check_result()[2]
            predicted_safe = num_failed_error == 0

            # Positive class = "unsafe/error" (standard for anomaly detection)
            # TP = correctly detected error, TN = correctly accepted safe data
            if not is_safe and not predicted_safe:
                tp += 1  # TP: error detected correctly
            elif not is_safe and predicted_safe:
                fn += 1  # FN: error missed (BAD!)
            elif is_safe and not predicted_safe:
                fp += 1  # FP: false alarm
            else:  # is_safe and predicted_safe
                tn += 1  # TN: safe data accepted correctly

        f1 = _compute_f1(tp, fp, fn)
        if convert_nan_to_zero and not math.isfinite(f1):
            f1 = 0.0

        # Sample validation entries for feedback
        samples = _sample_validation_entries(validation_results_by_label, feedback_sample_size, rng)
        samples_text = "(no validation entries)"
        if samples:
            samples_text = yaml.dump(samples, sort_keys=False)

        feedback = (
            "Column-level F1 feedback:\n"
            f"- Script: {script_name}\n"
            f"- Column: {column_name}\n"
            f"- F1: {f1:.4f}\n"
            f"- Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}\n"
            f"- Validation samples (max {feedback_sample_size}):\n{samples_text}"
        )

        return ScoreWithFeedback(score=f1, feedback=feedback)

    return metric_with_feedback
