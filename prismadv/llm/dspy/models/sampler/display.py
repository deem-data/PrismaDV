from collections import defaultdict
from typing import Dict, List, Tuple, Union, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from prismadv.data_models.trajectory import DVTrajectoryColumnGroupSuite

from prismadv.llm.dspy.models.sampler.types import (
    TrajectoryKey,
    ConstraintUID,
    SampledTrajectoryResult,
)
from prismadv.llm.dspy.models.sampler.utils import format_score, get_constraint_text


def format_result(
    results: Union[
        Dict[TrajectoryKey, SampledTrajectoryResult],
        Dict[TrajectoryKey, Dict[str, Any]]
    ],
    result_type: str = "auto",
) -> str:
    """
    Format results from either sample() or showall() in a formatted way.

    Constraints are deduplicated by constraint_uid before printing, so each unique
    constraint is shown only once even if it appears in multiple trajectories with
    different data labels.

    Args:
        results: Results from either sample() or showall()
        result_type: Type of results - "sample", "showall", or "auto" (auto-detect)

    Returns:
        A formatted string.
    """
    if not results:
        return "No results to display."

    # Auto-detect result type if not specified
    if result_type == "auto":
        first_value = next(iter(results.values()))
        if isinstance(first_value, SampledTrajectoryResult):
            result_type = "sample"
        elif isinstance(first_value, dict) and "column_groups" in first_value:
            result_type = "showall"
        else:
            raise ValueError(
                "Could not auto-detect result type. Please specify result_type='sample' or 'showall'"
            )

    if result_type == "sample":
        return format_sample_results(results)  # type: ignore[arg-type]
    if result_type == "showall":
        return format_showall_results(results)  # type: ignore[arg-type]

    raise ValueError(f"Unknown result_type: {result_type}. Use 'sample' or 'showall'")


# Backward-compatible wrapper (optional)
def print_result(
    results: Union[
        Dict[TrajectoryKey, SampledTrajectoryResult],
        Dict[TrajectoryKey, Dict[str, Any]]
    ],
    result_type: str = "auto",
) -> None:
    print(format_result(results, result_type=result_type))


def format_sample_results(results: Dict[TrajectoryKey, SampledTrajectoryResult]) -> str:
    """Format results from sample() method, deduplicating constraints by constraint_uid."""
    lines: List[str] = []

    for key, result in results.items():
        dataset_name, subtask_name, script_name, llm_name = key

        lines.append("\n" + "=" * 80)
        lines.append(f"Trajectory Key: ({dataset_name}, {subtask_name}, {script_name}, {llm_name})")
        lines.append("=" * 80)
        lines.append(f"Overall Fail Precision: {format_score(result.overall_fail_precision)}")
        lines.append(f"\nSampled Suites ({len(result.suites)} total):")

        for suite_idx, suite in enumerate(result.suites, 1):
            lines.append(f"\n  Suite {suite_idx}: Column Group = {suite.column_group}")

            # Deduplicate constraints by constraint_uid
            unique_constraints: Dict[ConstraintUID, Dict[str, Any]] = {}
            for trajectory in suite.trajectories:
                try:
                    constraint_uid = trajectory.constraint.uid
                    if constraint_uid not in unique_constraints:
                        score = result.constraint_scores.get(constraint_uid, np.nan)
                        constraint_text = getattr(trajectory.constraint, "suggestion", "")
                        unique_constraints[constraint_uid] = {
                            "score": score,
                            "constraint_text": constraint_text,
                            "count": 1,
                        }
                    else:
                        unique_constraints[constraint_uid]["count"] += 1
                except (AttributeError, Exception):
                    continue

            lines.append(f"  Unique Constraints ({len(unique_constraints)} total):")

            for constraint_uid, constraint_info in unique_constraints.items():
                score_str = format_score(constraint_info["score"])
                constraint_text = constraint_info["constraint_text"]
                count = constraint_info["count"]

                lines.append(f"    - UID: {constraint_uid}")
                lines.append(f"      Score: {score_str}")
                if count > 1:
                    lines.append(f"      (Appears in {count} trajectories with different data)")
                if constraint_text:
                    display_text = (
                        constraint_text[:100] + "..." if len(constraint_text) > 100 else constraint_text
                    )
                    lines.append(f"      Constraint: {display_text}")

        lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def format_showall_results(results: Dict[TrajectoryKey, Dict[str, Any]]) -> str:
    """Format results from showall() method, deduplicating constraints by constraint_uid."""
    lines: List[str] = []

    for key, result_data in results.items():
        dataset_name, subtask_name, script_name, llm_name = key
        overall_fail_precision = result_data["overall_fail_precision"]
        column_groups = result_data["column_groups"]

        lines.append("\n" + "=" * 80)
        lines.append(f"Trajectory Key: ({dataset_name}, {subtask_name}, {script_name}, {llm_name})")
        lines.append("=" * 80)
        lines.append(f"Overall Fail Precision: {format_score(overall_fail_precision)}")
        lines.append("\nColumn Groups:")

        for column_group_str, group_data in column_groups.items():
            column_group_overall = group_data["overall_score"]
            constraints_data = group_data["constraints"]

            lines.append(f"\n  Column Group: {column_group_str}")
            lines.append(f"  Overall Score: {format_score(column_group_overall)}")
            lines.append(f"  Unique Constraints ({len(constraints_data)} total):")

            for constraint_info in constraints_data:
                score_str = format_score(constraint_info["score"])
                constraint_text = constraint_info["constraint_text"]
                constraint_uid = constraint_info["constraint_uid"]

                lines.append(f"    - UID: {constraint_uid}")
                lines.append(f"      Score: {score_str}")
                if constraint_text:
                    display_text = (
                        constraint_text[:100] + "..." if len(constraint_text) > 100 else constraint_text
                    )
                    lines.append(f"      Constraint: {display_text}")

        lines.append("\n" + "=" * 80 + "\n")

    return "\n".join(lines)


def format_constraint_explanation(result: Dict[str, Any]) -> str:
    """Format a constraint explanation as a string."""
    lines: List[str] = []

    constraint_uid = result["constraint_uid"]
    constraint_text = result["constraint_text"]
    fail_precision_score = result["fail_precision_score"]
    trajectories_info = result["trajectories"]
    summary = result["summary"]

    lines.append("\n" + "=" * 80)
    lines.append(f"Constraint Explanation: {constraint_uid}")
    lines.append("=" * 80)

    if constraint_text:
        lines.append("\nConstraint Text:")
        lines.append(f"  {constraint_text}")

    lines.append(f"\nFail Precision Score: {format_score(fail_precision_score)}")

    lines.append("\nSummary Statistics:")
    lines.append(f"  Total Trajectories: {summary['total_trajectories']}")
    lines.append(f"  Safe Data: {summary['safe_data_count']}")
    lines.append(f"  Unsafe Data: {summary['unsafe_data_count']}")
    lines.append(f"  Total Failures: {summary['total_failures']}")
    lines.append(f"  Correct Failures (failed on unsafe data): {summary['correct_failures']}")
    lines.append(f"  False Positives (failed on safe data): {summary['false_positives']}")
    lines.append(f"  False Negatives (passed on unsafe data): {summary['false_negatives']}")
    lines.append(f"  Calculated Fail Precision: {format_score(summary['fail_precision'])}")

    lines.append(f"\nDetailed Trajectory Information ({len(trajectories_info)} total):")
    lines.append("-" * 80)

    grouped_by_key: Dict[TrajectoryKey, List[Dict[str, Any]]] = defaultdict(list)
    for traj_info in trajectories_info:
        key = traj_info["trajectory_key"]
        grouped_by_key[key].append(traj_info)

    for key, trajs in grouped_by_key.items():
        dataset_name, subtask_name, script_name, llm_name = key
        lines.append(f"\n  Trajectory Key: ({dataset_name}, {subtask_name}, {script_name}, {llm_name})")
        lines.append(f"  Column Group: {trajs[0]['column_group']}")

        for traj_info in trajs:
            data_label = traj_info["processed_data_label"]
            is_safe = traj_info["is_safe"]
            validation_passed = traj_info["validation_passed"]
            constraint_failed = traj_info["constraint_failed"]

            if constraint_failed:
                if is_safe:
                    issue_type = "❌ FALSE POSITIVE (failed on safe data)"
                else:
                    issue_type = "✅ CORRECT FAILURE (failed on unsafe data)"
            else:
                if not is_safe:
                    issue_type = "⚠️  FALSE NEGATIVE (passed on unsafe data)"
                else:
                    issue_type = "✅ CORRECT PASS (passed on safe data)"

            lines.append(f"    - Data Label: {data_label}")
            lines.append(f"      Is Safe: {is_safe}")
            lines.append(f"      Validation Passed: {validation_passed}")
            lines.append(f"      Issue Type: {issue_type}")

    lines.append("\n" + "=" * 80 + "\n")
    return "\n".join(lines)


# Backward-compatible wrapper (optional)
def print_constraint_explanation(result: Dict[str, Any]) -> None:
    print(format_constraint_explanation(result))


# ---- KEEP THESE FUNCTIONS (unchanged) ----

def create_score_distribution_plot(
    error_scores: np.ndarray,
    warning_scores: np.ndarray,
    num_bins: int
) -> None:
    """
    Create and display a histogram of score distributions by level.

    Args:
        error_scores: Array of scores for error-level constraints
        warning_scores: Array of scores for warning-level constraints
        num_bins: Number of histogram bins
    """
    # Define bin edges over [0.00, 1.0] range
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    # Compute bin counts
    error_counts = compute_bin_counts(error_scores, bin_edges)
    warning_counts = compute_bin_counts(warning_scores, bin_edges)

    # Set up plot
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.35

    plt.figure(figsize=(4, 2.5))

    # Set y-axis limit with some headroom
    max_count = max(error_counts.max(), warning_counts.max())
    plt.ylim(0, max(15, max_count * 1.1))

    # Create grouped bar chart
    plt.bar(bin_centers - bar_width / 2, error_counts, width=bar_width, label="error")
    plt.bar(bin_centers + bar_width / 2, warning_counts, width=bar_width, label="warning")

    # Configure plot
    plt.xlabel("Fail precision score")
    plt.ylabel("Number of constraints")
    plt.title("Distribution of fail precision by constraint level")
    plt.xticks(bin_centers, [f"{c:.2f}" for c in bin_centers], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_bin_counts(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Compute histogram bin counts for given values.

    Args:
        values: Array of values to bin
        bin_edges: Array of bin edge positions

    Returns:
        Array of counts per bin
    """
    counts = np.zeros(len(bin_edges) - 1, dtype=int)

    for value in values:
        # Find appropriate bin index
        # Handle edge case: if value equals or exceeds the last bin edge, put it in the last bin
        # This handles both exact 1.0 values and floating-point precision issues
        if value >= bin_edges[-1]:
            bin_idx = len(counts) - 1
        else:
            bin_idx = np.searchsorted(bin_edges, value, side="right") - 1

        # Ensure bin_idx is within valid range
        if 0 <= bin_idx < len(counts):
            counts[bin_idx] += 1

    return counts


def extract_trajectory_score(
    trajectory: Any,
    fail_precision_dict: Dict[Any, Dict[ConstraintUID, float]]
) -> Union[float, None]:
    """
    Extract the fail precision score for a specific trajectory.

    Args:
        trajectory: Trajectory object
        fail_precision_dict: Fail precision scores

    Returns:
        Fail precision score, or None if unavailable
    """
    column_group = getattr(trajectory, "column_group", None)
    constraints = getattr(trajectory, "constraints", None)

    if constraints is None or column_group is None:
        return None

    constraint_uid = getattr(constraints, "uid", None)
    if constraint_uid is None:
        return None

    # Get the fail precision score for this constraint
    try:
        score = fail_precision_dict[column_group][constraint_uid]
        if score is not None and not np.isnan(score):
            return score
    except (KeyError, Exception):
        pass

    return None