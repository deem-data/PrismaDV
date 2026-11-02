"""Column-level macro-F1 metrics for GEPA baseline optimization."""

from __future__ import annotations

import random

import oyaml as yaml

from sifta.adapters.dspy_adapter.dspy_adapter import ScoreWithFeedback
from workflow_prismadv.results.metrics import macro_scores
from workflow_sifta.baselines.gepa.metrics import _outcome_lines, _sample_validation_entries, _validate_on_labels
from workflow_sifta.optimization_fns import OptimizationWorkflow


def _column_constraints(workflow: OptimizationWorkflow, example, pred):
    """The constraints of the example's column only: the metric cannot see the other columns of the script."""
    return workflow.combine_constraints(single_column_results={example["target_column"]: pred})


def create_column_level_f1_metric(workflow: OptimizationWorkflow):
    """
    Create a column-level macro-F1 metric for DSPy evaluation.

    The metric validates the constraints of a single column on every error label of the script and returns the
    macro F1 over the safe and unsafe labels.
    """

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        # Check if pred is valid (should be dict with "assumptions" and "code" keys)
        if not isinstance(pred, dict) or "code" not in pred:
            return 0.0
        if not example.get("new_data_safety", {}):
            return 0.0
        _, safe, predicted_safe = _validate_on_labels(workflow, example, _column_constraints(workflow, example, pred))
        return macro_scores(safe, predicted_safe)["f1"]

    return metric


def create_column_level_f1_metric_with_feedback(
    workflow: OptimizationWorkflow,
    feedback_sample_size: int = 5,
    feedback_seed: int = 0,
):
    """
    Create a column-level macro-F1 metric with feedback for GEPA optimization.

    Args:
        workflow: OptimizationWorkflow instance
        feedback_sample_size: Number of validation entries to sample for feedback
        feedback_seed: Random seed for feedback sampling

    Returns:
        Metric function that returns ScoreWithFeedback
    """

    rng = random.Random(feedback_seed)

    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        # Check if pred is valid
        if not isinstance(pred, dict) or "code" not in pred:
            return ScoreWithFeedback(score=0.0, feedback="No constraints were generated.")
        if not example.get("new_data_safety", {}):
            return ScoreWithFeedback(score=0.0, feedback="No safety labels available for this example.")

        validation_results_by_label, safe, predicted_safe = _validate_on_labels(
            workflow, example, _column_constraints(workflow, example, pred))
        f1 = macro_scores(safe, predicted_safe)["f1"]

        # Sample validation entries for feedback
        samples = _sample_validation_entries(validation_results_by_label, feedback_sample_size, rng)
        samples_text = "(no validation entries)"
        if samples:
            samples_text = yaml.dump(samples, sort_keys=False)

        feedback = (
            "Column-level feedback:\n"
            f"- Script: {example['script_name']}\n"
            f"- Column: {example['target_column']}\n"
            f"- Macro F1: {f1:.4f}\n"
            + _outcome_lines(safe, predicted_safe)
            + f"- Validation samples (max {feedback_sample_size}):\n{samples_text}"
        )

        return ScoreWithFeedback(score=f1, feedback=feedback)

    return metric_with_feedback
