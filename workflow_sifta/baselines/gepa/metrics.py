"""Metrics for script-level DSPy baseline optimization."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import oyaml as yaml
from workflow_sifta.optimization_fns import OptimizationWorkflow

from prismadv.data_models.constraints_v2 import ConstraintsWithSources, ser_column_group_key
from prismadv.data_models.validated_results import ValidationResults
from sifta.dspy_sifta.teleprompt.sifta.sifta_utils import ScoreWithFeedback


def _compute_f1(tp: int, fp: int, fn: int) -> float:
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


def create_script_level_metric(
        workflow: OptimizationWorkflow,
        convert_nan_to_zero: bool = True,
):
    """Metric returning per-script F1 score across labels."""

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        if not isinstance(pred, ConstraintsWithSources):
            return 0.0

        dataset_name = example["dataset_name"]
        subtask_name = example["subtask_name"]
        new_data_safety = example.get("new_data_safety", {})
        if not new_data_safety:
            return 0.0

        constraints_with_sources = workflow.validate_constraints_on_training_data(
            dataset_name=dataset_name,
            subtask_name=subtask_name,
            constraints_with_sources=pred,
        )

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


def create_script_level_metric_with_feedback(
        workflow: OptimizationWorkflow,
        feedback_sample_size: int = 5,
        feedback_seed: int = 0,
        convert_nan_to_zero: bool = True,
):
    """Metric with sampled validation feedback for SIFTA."""

    rng = random.Random(feedback_seed)

    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        if not isinstance(pred, ConstraintsWithSources):
            return ScoreWithFeedback(score=0.0, feedback="No constraints were generated.")

        dataset_name = example["dataset_name"]
        subtask_name = example["subtask_name"]
        new_data_safety = example.get("new_data_safety", {})
        if not new_data_safety:
            return ScoreWithFeedback(score=0.0, feedback="No safety labels available for this script.")

        constraints_with_sources = workflow.validate_constraints_on_training_data(
            dataset_name=dataset_name,
            subtask_name=subtask_name,
            constraints_with_sources=pred,
        )

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

        samples = _sample_validation_entries(validation_results_by_label, feedback_sample_size, rng)
        samples_text = "(no validation entries)"
        if samples:
            samples_text = yaml.dump(samples, sort_keys=False)

        feedback = (
            "Script-level F1 feedback:\n"
            f"- F1: {f1:.4f}\n"
            f"- Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}\n"
            f"- Validation samples (max {feedback_sample_size}):\n{samples_text}"
        )

        return ScoreWithFeedback(score=f1, feedback=feedback)

    return metric_with_feedback
