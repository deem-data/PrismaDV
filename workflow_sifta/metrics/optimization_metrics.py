"""Optimization metrics for DSPy evaluation."""

import hashlib
from collections import OrderedDict
from typing import Dict, List, Any

import numpy as np

from prismadv.data_models.trajectory import DVTrajectory
from prismadv.llm.dspy.models.sampler.base import TrajectorySampler
from sifta.adapters.dspy_adapter.dspy_adapter import ScoreWithFeedback
from workflow_sifta.optimization_fns import OptimizationWorkflow


FEEDBACK_LEVELS = ("score", "stats", "assumptions", "full")
METRIC_REV = "error_level"
NO_ERROR_CHECKS = ("No error-level constraints: a warning-level constraint never rejects a batch, so this column "
                   "catches no error.")


def example_labels(example, workflow: OptimizationWorkflow) -> List[str]:
    """The batches an example is scored on: the labels of its ground truth (new_data_safety). Train and eval examples
    can have different ones (workflow_sifta._1_optimization --num_batches); an example without ground truth uses the
    workflow's train labels."""
    safety = example.get("new_data_safety")
    return list(safety) if safety else workflow.train_processed_data_label_list


def error_level_only(trajectories_dict: Dict[str, Dict[str, List[DVTrajectory]]]):
    return {dataset: {script: [t for t in trajectories if t.constraint.level == "error"]
                      for script, trajectories in scripts.items()}
            for dataset, scripts in trajectories_dict.items()}


def _build_assumption_trace(
    trajectory: DVTrajectory,
    max_lines_per_assumption: int = 15,
    include_code: bool = True,
) -> str:
    """
    Build assumption trace with code snippets for a single trajectory.

    Traces back from constraint to source assumptions and extracts
    relevant code snippets using the assumption's source locations.

    Args:
        trajectory: DVTrajectory containing constraint, assumptions, and script
        max_lines_per_assumption: Maximum lines to show per assumption's code context
        include_code: Whether to walk the assumption's source locations into the script
            and attach the originating code. Setting this False keeps the assumption
            text but drops the data-code graph backtrace, isolating its contribution.

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

        if include_code and assumption.sources:
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

    Counts use the safe-as-positive convention, matching fail_precision =
    TN / (TN + FN): a constraint that passes predicts "safe" (positive), and a
    constraint that fails predicts "erroneous" (negative).

    Returns:
        Dict keyed by constraint_uid containing:
        - suggestion: constraint code
        - fail_precision: score from sampler
        - true_negatives: count of failures on erroneous data (correct rejection)
        - false_negatives: count of failures on safe data (false alarm)
        - false_positives: count of passes on erroneous data (missed error)
        - true_positives: count of passes on safe data (correct accept)
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
                        "level": traj.constraint.level,
                        "fail_precision": constraint_scores.get(constraint_uid, float('nan')),
                        "true_negatives": 0,
                        "false_negatives": 0,
                        "false_positives": 0,
                        "true_positives": 0,
                        "total_runs": 0,
                        "sample_trajectory": traj,
                    }

                analysis[constraint_uid]["total_runs"] += 1

                # validation_results.status: True = passed, False = failed
                validation_passed = traj.validation_results.status
                is_safe = traj.is_safe

                if not validation_passed:  # Constraint failed = predicted erroneous
                    if not is_safe:
                        analysis[constraint_uid]["true_negatives"] += 1
                    else:
                        analysis[constraint_uid]["false_negatives"] += 1
                else:  # Constraint passed = predicted safe
                    if not is_safe:
                        analysis[constraint_uid]["false_positives"] += 1
                    else:
                        analysis[constraint_uid]["true_positives"] += 1

    return analysis


def _format_batch_outcome(
    all_trajectories_dict: Dict[str, Dict[str, List[DVTrajectory]]],
    max_constraints: int = 5,
) -> str:
    batches: Dict[str, Dict[str, Any]] = {}
    checks: Dict[str, Dict[str, Any]] = {}
    for script_dict in all_trajectories_dict.values():
        for trajectories in script_dict.values():
            for traj in trajectories:
                batch = batches.setdefault(traj.processed_data_label, {"safe": traj.is_safe, "flagged": False})
                failed = not traj.validation_results.status
                if traj.constraint.level != "error" or not failed:
                    continue
                batch["flagged"] = True
                check = checks.setdefault(traj.constraint.uid, {"suggestion": traj.constraint.suggestion,
                                                                "safe": set(), "unsafe": set()})
                check["safe" if traj.is_safe else "unsafe"].add(traj.processed_data_label)
    n_safe = sum(b["safe"] for b in batches.values())
    n_unsafe = len(batches) - n_safe
    fa = sum(b["safe"] and b["flagged"] for b in batches.values())
    caught = sum(not b["safe"] and b["flagged"] for b in batches.values())
    lines = ["=== Batch-Level Outcome (a batch is flagged if any error-level check fails) ===",
             f"- Safe batches: {n_safe}; flagged by mistake (false alarms): {fa}",
             f"- Unsafe batches: {n_unsafe}; flagged (errors caught): {caught}"]
    noisy = sorted((c for c in checks.values() if c["safe"]), key=lambda c: (-len(c["safe"]), len(c["unsafe"])))
    if noisy:
        lines.append("Error-level checks that flag safe batches (each flagged safe batch is a false alarm "
                     "unless removed or relaxed):")
        for c in noisy[:max_constraints]:
            s = c["suggestion"] if len(c["suggestion"]) <= 100 else c["suggestion"][:100] + "..."
            lines.append(f"  - `{s}`: flags {len(c['safe'])} safe and {len(c['unsafe'])} unsafe batches")
        if len(noisy) > max_constraints:
            lines.append(f"  ... and {len(noisy) - max_constraints} more checks that flag safe batches")
    return "\n".join(lines)


def _format_detailed_feedback(
    example: Dict[str, Any],
    analysis: Dict[str, Dict[str, Any]],
    validity_info: Dict[str, Any],
    fail_precision: float,
    max_constraints: int = 5,
    feedback_level: str = "full",
) -> str:
    """
    Format detailed feedback for constraints with score < 1.0.

    Args:
        example: Dict with script_name, target_column, dataset_name, subtask_name
        analysis: Per-constraint analysis from _analyze_trajectories()
        validity_info: Constraint validity info from get_constraint_validity_info()
        fail_precision: Overall fail precision score
        max_constraints: Maximum number of constraints to show details for
        feedback_level: "stats" stops after the per-constraint counts, "assumptions"
            adds the source assumption text, "full" also attaches the originating code.

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

    # Also include constraints with NaN (no failures) at the end if they missed errors
    nan_constraints = [
        (uid, data) for uid, data in analysis.items()
        if np.isnan(data["fail_precision"]) and data["false_positives"] > 0
    ]
    sorted_constraints.extend(nan_constraints)

    # Show details for top worst-performing constraints
    for i, (constraint_uid, data) in enumerate(sorted_constraints[:max_constraints], 1):
        fp_str = f"{data['fail_precision']:.2f}" if not np.isnan(data['fail_precision']) else "N/A"
        lines.append(f"Constraint {i} (level={data['level']}, fail_precision={fp_str}):")

        # Truncate suggestion for display
        suggestion = data["suggestion"]
        if len(suggestion) > 100:
            suggestion = suggestion[:100] + "..."
        lines.append(f"  Code: `{suggestion}`")

        # Statistics (safe = positive class)
        total = data["total_runs"]
        tn = data["true_negatives"]
        fn = data["false_negatives"]
        fp = data["false_positives"]

        if data["level"] == "error":
            lines.append(
                f"  Stats: {tn} true negatives (correct rejections), {fn} false negatives (false alarms), "
                f"{fp} false positives (missed errors) (of {total} runs)"
            )
            if fn > 0:
                lines.append(f"  Issue: {fn} false alarms on safe data")
            if fp > 0:
                lines.append(f"  Issue: Missed {fp} violations on erroneous data")
        else:
            lines.append(
                f"  Stats: fails on {tn} erroneous and {fn} safe batches, passes on {fp} erroneous batches "
                f"(of {total} runs). Warning level: its failures reject no batch."
            )

        # Assumption trace, optionally with the code it was derived from
        if feedback_level in ("assumptions", "full"):
            include_code = feedback_level == "full"
            sample_traj = data["sample_trajectory"]
            assumption_trace = _build_assumption_trace(sample_traj, include_code=include_code)
            lines.append("\n  Source Assumptions & Code:" if include_code else "\n  Source Assumptions:")
            lines.append(assumption_trace)
        lines.append("")

    if len(sorted_constraints) > max_constraints:
        lines.append(f"... and {len(sorted_constraints) - max_constraints} more constraints")
        lines.append("")

    # Summary section (safe = positive class, so fail_precision = TN / (TN + FN)); error-level checks only,
    # as only they count for the score
    lines.append("=== Summary (error-level checks) ===")
    error_checks = [d for d in analysis.values() if d["level"] == "error"]
    total_tn = sum(d["true_negatives"] for d in error_checks)
    total_fn = sum(d["false_negatives"] for d in error_checks)
    total_fp = sum(d["false_positives"] for d in error_checks)
    total_tp = sum(d["true_positives"] for d in error_checks)

    lines.append(f"- True Negatives (correct failures on erroneous data): {total_tn}")
    lines.append(f"- False Negatives (incorrect failures on safe data): {total_fn}")
    lines.append(f"- False Positives (missed violations on erroneous data): {total_fp}")
    lines.append(f"- True Positives (correct passes on safe data): {total_tp}")

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

        for label in example_labels(example, workflow):
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
            trajectories_dict=error_level_only(all_trajectories_dict),
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
        cache_size: int = 64,
        feedback_level: str = "full",
        max_constraints: int = 5,
):
    """
    Create a metric function with feedback for DSPy SIFTA optimization.

    SIFTA calls this metric once per example to score it, then again once per
    predictor to collect that predictor's feedback. Only the feedback *text*
    depends on `pred_name`; the validation work behind it does not. The expensive
    part is therefore memoized on (example, prediction) so building a reflective
    dataset for N predictors costs one evaluation rather than N + 1.

    Args:
        workflow: OptimizationWorkflow instance
        llm_name: Name of the LLM to use
        convert_nan_to_zero: Whether to convert NaN fail_precision to 0.0 (default: True)
        cache_size: Number of (example, prediction) evaluations to retain
        feedback_level: How much of the feedback to expose, as an ablation ladder.
        max_constraints: How many constraints the feedback shows in detail, in the per-constraint list and in the
            list of error-level checks that flag safe batches (default 5); the rest are counted only.

    Returns:
        Metric function that takes (gold, pred, trace=None, pred_name=None, pred_trace=None)
        and returns ScoreWithFeedback. The function carries a `stats` dict recording
        how many evaluations were actually computed versus served from cache.
    """
    if feedback_level not in FEEDBACK_LEVELS:
        raise ValueError(
            f"Unknown feedback_level: {feedback_level!r}. Must be one of {FEEDBACK_LEVELS}"
        )
    if max_constraints < 1:
        raise ValueError(f"max_constraints must be at least 1, got {max_constraints}")
    evaluation_cache: "OrderedDict[str, dict]" = OrderedDict()
    stats = {"computations": 0, "cache_hits": 0}

    def _cache_key(example, pred) -> str:
        """Identify an (example, prediction) pair; entry UIDs make predictions unique."""
        identity = "|".join(str(example.get(k, "")) for k in
                            ("dataset_name", "subtask_name", "script_name", "target_column"))
        try:
            payload = f"{pred['assumptions']!r}{pred['code']!r}"
        except Exception:
            payload = repr(pred)
        return hashlib.md5(f"{identity}|{payload}".encode()).hexdigest()

    def _evaluate(example, pred) -> dict:
        """Run constraint validation for one example and score it. Independent of pred_name."""
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
        for label in example_labels(example, workflow):
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
            for traj_dataset_name, script_dict in trajectories_dict.items():
                if traj_dataset_name not in all_trajectories_dict:
                    all_trajectories_dict[traj_dataset_name] = {}
                for traj_script_name, trajectories in script_dict.items():
                    if traj_script_name not in all_trajectories_dict[traj_dataset_name]:
                        all_trajectories_dict[traj_dataset_name][traj_script_name] = []
                    all_trajectories_dict[traj_dataset_name][traj_script_name].extend(trajectories)
        # The score counts error-level constraints only; the feedback below still shows every constraint.
        aggregated_trajectories = workflow.aggregate_trajectories_for_sampling(
            trajectories_dict=error_level_only(all_trajectories_dict),
            llm_name=llm_name,
        )

        # Handle empty aggregated_trajectories
        if len(aggregated_trajectories) == 0:
            has_any = any(ts for scripts in all_trajectories_dict.values() for ts in scripts.values())
            return {"empty": True, "feedback": NO_ERROR_CHECKS if has_any else "No valid trajectories generated."}

        sampler = TrajectorySampler.from_aggregated_trajectories(aggregated_trajectories)
        sampler.calculate()
        assert len(aggregated_trajectories) <= 1
        sampled_results = sampler.sample(
            temperature=0.0, num_top_column_groups=1, num_top_constraints=10
        )
        assert len(sampled_results) == 1
        sampled_result = list(sampled_results.values())[0]

        return {
            "empty": False,
            "fail_precision": sampled_result.overall_fail_precision,
            "validity_info": validity_info,
            "sampled_result": sampled_result,
            "all_trajectories_dict": all_trajectories_dict,
        }

    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        key = _cache_key(example, pred)
        if key in evaluation_cache:
            evaluation_cache.move_to_end(key)
            evaluation = evaluation_cache[key]
            stats["cache_hits"] += 1
        else:
            evaluation = _evaluate(example, pred)
            stats["computations"] += 1
            evaluation_cache[key] = evaluation
            while len(evaluation_cache) > cache_size:
                evaluation_cache.popitem(last=False)

        if evaluation["empty"]:
            return ScoreWithFeedback(score=0.0, feedback=evaluation["feedback"])

        validity_info = evaluation["validity_info"]
        sampled_result = evaluation["sampled_result"]
        all_trajectories_dict = evaluation["all_trajectories_dict"]
        fail_precision = evaluation["fail_precision"]

        # Handle NaN - convert to 0 if configured
        original_fail_precision = fail_precision
        if np.isnan(fail_precision) and convert_nan_to_zero:
            fail_precision = 0.0

        # Only provide detailed feedback for the constraint generator (ir_generation)
        # Other modules (dataflow_inspector, assumption_generation) get minimal feedback
        is_constraint_generator = pred_name == "ir_generation"

        if feedback_level == "score" or not is_constraint_generator:
            # Minimal feedback for non-constraint modules
            if np.isnan(original_fail_precision):
                fb_text = f"Score: {fail_precision:.2f} (no error-level constraint failed, so no batch was rejected)"
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
No error-level constraint failed on the test data, so no batch was rejected. The potential cases are:
- The error-level constraints are too general and do not capture any failures.
- The test data does not contain any violations for the given constraints.
- Constraints that would catch the errors are at warning level, and a warning never rejects a batch.
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
                max_constraints=max_constraints,
                feedback_level=feedback_level,
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
All constraint failures correctly identified erroneous data.
"""

        fb_text += "\n" + _format_batch_outcome(all_trajectories_dict, max_constraints)
        return ScoreWithFeedback(score=fail_precision, feedback=fb_text)

    metric_with_feedback.stats = stats
    return metric_with_feedback
