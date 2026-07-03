"""Optimization metrics for DSPy evaluation."""

from typing import Dict, List, Any

import numpy as np

from prismadv.data_models.trajectory import DVTrajectory
from prismadv.llm.dspy.models.sampler.base import TrajectorySampler
from sifta.adapters.dspy_adapter.dspy_adapter import ScoreWithFeedback
from workflow_sifta.optimization_fns import OptimizationWorkflow


def _build_assumption_trace(
    trajectory: DVTrajectory,
    max_lines_per_assumption: int = 15
) -> str:
    """
    Build assumption trace with code snippets for a single trajectory.

    Traces back from constraint to source assumptions and extracts
    relevant code snippets using the assumption's source locations.

    Args:
        trajectory: DVTrajectory containing constraint, assumptions, and script
        max_lines_per_assumption: Maximum lines to show per assumption's code context

    Returns:
        Formatted string with assumption text and code snippets
    """
    constraint = trajectory.constraint
    source_assumption_uids = set(constraint.source_assumptions)

    if not source_assumption_uids:
        return "  (No source assumptions linked to this constraint)"

    lines = []
    for assumption in trajectory.assumptions:
        if assumption.uid not in source_assumption_uids:
            continue

        lines.append(f"  Assumption: \"{assumption.text}\"")

        if assumption.sources:
            try:
                focused_code = trajectory.script.focused_code(assumption.sources)
                if focused_code:
                    code_lines = focused_code.split('\n')
                    if len(code_lines) > max_lines_per_assumption:
                        code_lines = code_lines[:max_lines_per_assumption] + ['    ...']
                    lines.append("  Code Context:")
                    for code_line in code_lines:
                        lines.append(f"    {code_line}")
            except Exception:
                lines.append("  Code Context: (unable to extract)")

    return '\n'.join(lines) if lines else "  (No matching assumptions found)"


def _analyze_trajectories(
    all_trajectories_dict: Dict[str, Dict[str, List[DVTrajectory]]],
    constraint_scores: Dict[str, float]
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all trajectories to compute per-constraint statistics.

    Args:
        all_trajectories_dict: Nested dict {dataset: {script: [trajectories]}}
        constraint_scores: Dict mapping constraint UIDs to fail precision scores

    Returns:
        Dict keyed by constraint_uid containing:
        - suggestion: constraint code
        - fail_precision: score from sampler
        - true_positives: count of failures on unsafe data (correct)
        - false_positives: count of failures on safe data (incorrect)
        - false_negatives: count of passes on unsafe data (missed)
        - total_runs: total trajectory count for this constraint
        - sample_trajectory: a sample DVTrajectory for code context
    """
    analysis = {}

    for dataset_name, script_dict in all_trajectories_dict.items():
        for script_name, trajectories in script_dict.items():
            for traj in trajectories:
                constraint_uid = traj.constraint.uid

                if constraint_uid not in analysis:
                    analysis[constraint_uid] = {
                        "suggestion": traj.constraint.suggestion,
                        "fail_precision": constraint_scores.get(constraint_uid, float('nan')),
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": 0,
                        "total_runs": 0,
                        "sample_trajectory": traj,
                    }

                analysis[constraint_uid]["total_runs"] += 1

                # validation_results.status: True = passed, False = failed
                validation_passed = traj.validation_results.status
                is_safe = traj.is_safe

                if not validation_passed:  # Constraint failed (flagged the data)
                    if not is_safe:
                        # Failed on unsafe data = true positive (correct)
                        analysis[constraint_uid]["true_positives"] += 1
                    else:
                        # Failed on safe data = false positive (incorrect)
                        analysis[constraint_uid]["false_positives"] += 1
                else:  # Constraint passed (did not flag the data)
                    if not is_safe:
                        # Passed on unsafe data = false negative (missed violation)
                        analysis[constraint_uid]["false_negatives"] += 1
                    # Passed on safe data = true negative (correct, not counted)

    return analysis


def _format_detailed_feedback(
    example: Dict[str, Any],
    analysis: Dict[str, Dict[str, Any]],
    validity_info: Dict[str, Any],
    fail_precision: float,
    max_constraints: int = 5
) -> str:
    """
    Format detailed feedback for constraints with score < 1.0.

    Args:
        example: Dict with script_name, target_column, dataset_name, subtask_name
        analysis: Per-constraint analysis from _analyze_trajectories()
        validity_info: Constraint validity info from get_constraint_validity_info()
        fail_precision: Overall fail precision score
        max_constraints: Maximum number of constraints to show details for

    Returns:
        Formatted feedback string
    """
    lines = []

    # Context section
    lines.append("=== Context ===")
    lines.append(f"Script: {example.get('script_name', 'unknown')}")
    lines.append(f"Target Column: {example.get('target_column', 'unknown')}")
    lines.append(f"Dataset: {example.get('dataset_name', 'unknown')} / {example.get('subtask_name', 'unknown')}")
    lines.append("")

    # Validity section
    lines.append("=== Constraint Validity (Training Data) ===")
    lines.append(f"- Valid: {validity_info['valid_count']}/{validity_info['total_count']} ({validity_info['valid_ratio']:.1%})")
    lines.append(f"- Invalid: {validity_info['invalid_count']}")

    if validity_info["invalid_constraints"]:
        lines.append("\nInvalid constraint reasons:")
        for inv in validity_info["invalid_constraints"][:3]:
            suggestion_preview = inv['suggestion'][:80] + '...' if len(inv['suggestion']) > 80 else inv['suggestion']
            lines.append(f"  - `{suggestion_preview}`")
            lines.append(f"    Reason: {inv['reason_if_invalid']}")
        if len(validity_info["invalid_constraints"]) > 3:
            lines.append(f"  ... and {len(validity_info['invalid_constraints']) - 3} more")
    lines.append("")

    # Performance section
    lines.append("=== Constraint Performance (Test Data) ===")
    lines.append(f"Overall Fail Precision: {fail_precision:.2f}")
    lines.append("")

    # Sort constraints by fail precision (worst first), excluding NaN
    sorted_constraints = sorted(
        [(uid, data) for uid, data in analysis.items() if not np.isnan(data["fail_precision"])],
        key=lambda x: x[1]["fail_precision"]
    )

    # Also include constraints with NaN (no failures) at the end if they have false negatives
    nan_constraints = [
        (uid, data) for uid, data in analysis.items()
        if np.isnan(data["fail_precision"]) and data["false_negatives"] > 0
    ]
    sorted_constraints.extend(nan_constraints)

    # Show details for top worst-performing constraints
    for i, (constraint_uid, data) in enumerate(sorted_constraints[:max_constraints], 1):
        fp_str = f"{data['fail_precision']:.2f}" if not np.isnan(data['fail_precision']) else "N/A"
        lines.append(f"Constraint {i} (fail_precision={fp_str}):")

        # Truncate suggestion for display
        suggestion = data["suggestion"]
        if len(suggestion) > 100:
            suggestion = suggestion[:100] + "..."
        lines.append(f"  Code: `{suggestion}`")

        # Statistics
        total = data["total_runs"]
        tp = data["true_positives"]
        fp = data["false_positives"]
        fn = data["false_negatives"]

        lines.append(f"  Stats: {tp} true positives, {fp} false positives, {fn} false negatives (of {total} runs)")

        if fp > 0:
            lines.append(f"  Issue: {fp} false alarms on safe data")
        if fn > 0:
            lines.append(f"  Issue: Missed {fn} violations on unsafe data")

        # Assumption trace with code context
        sample_traj = data["sample_trajectory"]
        assumption_trace = _build_assumption_trace(sample_traj)
        lines.append("\n  Source Assumptions & Code:")
        lines.append(assumption_trace)
        lines.append("")

    if len(sorted_constraints) > max_constraints:
        lines.append(f"... and {len(sorted_constraints) - max_constraints} more constraints")
        lines.append("")

    # Summary section
    lines.append("=== Summary ===")
    total_tp = sum(d["true_positives"] for d in analysis.values())
    total_fp = sum(d["false_positives"] for d in analysis.values())
    total_fn = sum(d["false_negatives"] for d in analysis.values())

    lines.append(f"- True Positives (correct failures on unsafe data): {total_tp}")
    lines.append(f"- False Positives (incorrect failures on safe data): {total_fp}")
    lines.append(f"- False Negatives (missed violations on unsafe data): {total_fn}")

    return '\n'.join(lines)


def create_metric(
        workflow: OptimizationWorkflow,
        llm_name: str,
        convert_nan_to_zero: bool = True,
):
    """
    Create a metric function for DSPy evaluation.

    Args:
        workflow: OptimizationWorkflow instance
        llm_name: Name of the LLM to use
        convert_nan_to_zero: Whether to convert NaN fail_precision to 0.0 (default: True)

    Returns:
        Metric function that takes (example, pred, trace=None, pred_name=None, pred_trace=None)
    """

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        column_name = example["target_column"]
        script_name_list = [example["script_name"]]
        dataset_name = example["dataset_name"]
        subtask_name = example["subtask_name"]
        single_column_results = {}
        single_column_results[column_name] = pred
        # Combine all constraints
        constraints_with_sources = workflow.combine_constraints(
            single_column_results=single_column_results,
        )
        # Validate constraints on training data for this specific dataset/subtask
        constraints_with_sources = workflow.validate_constraints_on_training_data(
            dataset_name=dataset_name,
            subtask_name=subtask_name,
            constraints_with_sources=constraints_with_sources,
        )
        all_trajectories_dict = {}

        for label in workflow.train_processed_data_label_list:
            # Validate constraints on test data (script_name doesn't matter for validation)
            validation_results = workflow.validate_constraints_on_test_data(
                dataset_name=dataset_name,
                subtask_name=subtask_name,
                constraints_with_sources=constraints_with_sources,
                processed_data_label=label,
                clean=False,
            )
            # Construct validation_results_dict with script_name from example
            script_name = example["script_name"]
            validation_results_dict = {dataset_name: {script_name: validation_results}}
            trajectories_dict = workflow.create_trajectories_from_constraints(
                script_name_list=script_name_list,
                processed_data_label=label,
                llm_name=llm_name,
                constraints_with_sources=constraints_with_sources,
                validation_results_dict=validation_results_dict,
                clean=False,
            )
            for dataset_name, script_dict in trajectories_dict.items():
                if dataset_name not in all_trajectories_dict:
                    all_trajectories_dict[dataset_name] = {}
                for script_name, trajectories in script_dict.items():
                    if script_name not in all_trajectories_dict[dataset_name]:
                        all_trajectories_dict[dataset_name][script_name] = []
                    all_trajectories_dict[dataset_name][script_name].extend(trajectories)

        aggregated_trajectories = workflow.aggregate_trajectories_for_sampling(
            trajectories_dict=all_trajectories_dict,
            llm_name=llm_name,
        )

        # Handle empty aggregated_trajectories
        if len(aggregated_trajectories) == 0:
            return 0.0  # Return 0 score for cases with no trajectories

        sampler = TrajectorySampler.from_aggregated_trajectories(aggregated_trajectories)
        sampler.calculate()
        assert len(
            aggregated_trajectories) <= 1, f"Expected at most 1 trajectory key, got {len(aggregated_trajectories)}"
        sampled_results = sampler.sample(
            temperature=0.0, num_top_column_groups=1, num_top_constraints=10
        )
        assert len(sampled_results) == 1, f"Expected exactly 1 sampled result, got {len(sampled_results)}"
        fail_precision = list(sampled_results.values())[0].overall_fail_precision

        if convert_nan_to_zero:
            fail_precision = 0.0 if np.isnan(fail_precision) else fail_precision

        return fail_precision

    return metric


def create_metric_with_feedback(
        workflow: OptimizationWorkflow,
        llm_name: str,
        convert_nan_to_zero: bool = True,
):
    """
    Create a metric function with feedback for DSPy SIFTA optimization.

    Args:
        workflow: OptimizationWorkflow instance
        llm_name: Name of the LLM to use
        convert_nan_to_zero: Whether to convert NaN fail_precision to 0.0 (default: True)

    Returns:
        Metric function that takes (gold, pred, trace=None, pred_name=None, pred_trace=None)
        and returns ScoreWithFeedback
    """

    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        column_name = example["target_column"]
        script_name_list = [example["script_name"]]
        dataset_name = example["dataset_name"]
        subtask_name = example["subtask_name"]
        script_name = example["script_name"]
        single_column_results = {}
        single_column_results[column_name] = pred
        # Combine all constraints
        constraints_with_sources = workflow.combine_constraints(
            single_column_results=single_column_results,
        )
        # Validate constraints on training data for this specific dataset/subtask
        constraints_with_sources = workflow.validate_constraints_on_training_data(
            dataset_name=dataset_name,
            subtask_name=subtask_name,
            constraints_with_sources=constraints_with_sources,
        )

        # Get validity information for constraints
        validity_info = constraints_with_sources.get_constraint_validity_info()

        all_trajectories_dict = {}
        for label in workflow.train_processed_data_label_list:
            # Validate constraints on test data (script_name doesn't matter for validation)
            validation_results = workflow.validate_constraints_on_test_data(
                dataset_name=dataset_name,
                subtask_name=subtask_name,
                constraints_with_sources=constraints_with_sources,
                processed_data_label=label,
                clean=False,
            )
            # Construct validation_results_dict with script_name from example
            validation_results_dict = {dataset_name: {script_name: validation_results}}
            trajectories_dict = workflow.create_trajectories_from_constraints(
                script_name_list=script_name_list,
                processed_data_label=label,
                llm_name=llm_name,
                constraints_with_sources=constraints_with_sources,
                validation_results_dict=validation_results_dict,
                clean=False,
            )
            for dataset_name, script_dict in trajectories_dict.items():
                if dataset_name not in all_trajectories_dict:
                    all_trajectories_dict[dataset_name] = {}
                for script_name, trajectories in script_dict.items():
                    if script_name not in all_trajectories_dict[dataset_name]:
                        all_trajectories_dict[dataset_name][script_name] = []
                    all_trajectories_dict[dataset_name][script_name].extend(trajectories)
                    # print(
                    #     f"    Created {len(trajectories)} trajectories for {dataset_name}/{script_name}"
                    # )
        aggregated_trajectories = workflow.aggregate_trajectories_for_sampling(
            trajectories_dict=all_trajectories_dict,
            llm_name=llm_name,
        )

        # Handle empty aggregated_trajectories
        if len(aggregated_trajectories) == 0:
            return ScoreWithFeedback(score=0.0, feedback="No valid trajectories generated.")

        sampler = TrajectorySampler.from_aggregated_trajectories(aggregated_trajectories)
        sampler.calculate()
        assert len(aggregated_trajectories) <= 1
        sampled_results = sampler.sample(
            temperature=0.0, num_top_column_groups=1, num_top_constraints=10
        )
        assert len(sampled_results) == 1
        sampled_result = list(sampled_results.values())[0]
        fail_precision = sampled_result.overall_fail_precision

        # Handle NaN - convert to 0 if configured
        original_fail_precision = fail_precision
        if np.isnan(fail_precision) and convert_nan_to_zero:
            fail_precision = 0.0

        # Only provide detailed feedback for the constraint generator (ir_generation)
        # Other modules (dataflow_inspector, assumption_generation) get minimal feedback
        is_constraint_generator = pred_name == "ir_generation"

        if not is_constraint_generator:
            # Minimal feedback for non-constraint modules
            if np.isnan(original_fail_precision):
                fb_text = f"Score: {fail_precision:.2f} (all constraints passed, no failures detected)"
            else:
                fb_text = f"Score: {fail_precision:.2f}"
            return ScoreWithFeedback(score=fail_precision, feedback=fb_text)

        # Detailed feedback only for ir_generation (constraint generator)
        if np.isnan(original_fail_precision):
            fb_text = f"""=== Context ===
Script: {example.get('script_name', 'unknown')}
Target Column: {example.get('target_column', 'unknown')}
Dataset: {example.get('dataset_name', 'unknown')} / {example.get('subtask_name', 'unknown')}

=== Constraint Validity (Training Data) ===
- Valid: {validity_info['valid_count']}/{validity_info['total_count']} ({validity_info['valid_ratio']:.1%})
- Invalid: {validity_info['invalid_count']}

=== Test Data Performance ===
All constraints passed on the test data. The potential cases are:
- The constraints are too general and do not capture any failures.
- The test data does not contain any violations for the given constraints.
"""
        elif fail_precision < 1.0:
            # Get constraint scores from sampled result
            constraint_scores = sampled_result.constraint_scores

            # Analyze trajectories for detailed statistics
            analysis = _analyze_trajectories(all_trajectories_dict, constraint_scores)

            # Format detailed feedback
            fb_text = _format_detailed_feedback(
                example=example,
                analysis=analysis,
                validity_info=validity_info,
                fail_precision=fail_precision,
                max_constraints=5
            )
        else:
            # Perfect score - simple feedback
            fb_text = f"""=== Context ===
Script: {example.get('script_name', 'unknown')}
Target Column: {example.get('target_column', 'unknown')}
Dataset: {example.get('dataset_name', 'unknown')} / {example.get('subtask_name', 'unknown')}

=== Constraint Validity (Training Data) ===
- Valid: {validity_info['valid_count']}/{validity_info['total_count']} ({validity_info['valid_ratio']:.1%})
- Invalid: {validity_info['invalid_count']}

=== Test Data Performance ===
Perfect fail precision score: {fail_precision:.2f}
All constraint failures correctly identified unsafe data.
"""

        return ScoreWithFeedback(score=fail_precision, feedback=fb_text)

    return metric_with_feedback
