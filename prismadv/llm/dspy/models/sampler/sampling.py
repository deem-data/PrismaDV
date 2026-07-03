from typing import Dict, List, Tuple

import numpy as np

from prismadv.data_models.trajectory import DVTrajectoryColumnGroupSuite

from prismadv.llm.dspy.models.sampler.types import ColumnGroup, ConstraintUID, FailPrecisionScore
from prismadv.llm.dspy.models.sampler.utils import is_nan, find_suite_by_constraint


def select_top_fail_precision(
    fail_precision_dict: Dict[ColumnGroup, Dict[ConstraintUID, FailPrecisionScore]],
    num_top_column_groups: int,
    num_top_constraints: int,
    overall_key: str = "overall",
) -> Dict[ColumnGroup, Dict[ConstraintUID, FailPrecisionScore]]:
    """
    Select top column groups and constraints based on fail precision scores.
    
    First, ranks column groups by their overall fail precision score.
    Then, for each top column group, selects the top N constraints by score.
    
    Args:
        fail_precision_dict: Fail precision scores per column group and constraint
        num_top_column_groups: Number of top column groups to select
        num_top_constraints: Number of top constraints per column group
        overall_key: Key used for overall scores in the fail_precision_dict
        
    Returns:
        Dict mapping selected column groups to their top constraints with scores
    """
    if num_top_column_groups <= 0 or num_top_constraints <= 0:
        return {}

    # Step 1: Select top column groups by overall score
    top_column_groups = select_top_column_groups(
        fail_precision_dict,
        num_top_column_groups,
        overall_key
    )

    # Step 2: For each selected column group, select top constraints
    result = {}
    for column_group in top_column_groups:
        top_constraints = select_top_constraints_for_group(
            fail_precision_dict.get(column_group, {}),
            num_top_constraints,
            overall_key
        )
        result[column_group] = top_constraints

    return result


def select_top_column_groups(
    fail_precision_dict: Dict[ColumnGroup, Dict[ConstraintUID, FailPrecisionScore]],
    num_top: int,
    overall_key: str
) -> List[ColumnGroup]:
    """
    Select top column groups based on their overall fail precision scores.
    
    Args:
        fail_precision_dict: Fail precision scores per column group
        num_top: Number of top column groups to select
        overall_key: Key for overall scores
        
    Returns:
        List of top column groups ordered by score (descending)
    """
    column_group_scores = []

    for column_group, scores_dict in fail_precision_dict.items():
        overall_score = scores_dict.get(overall_key, None)
        if overall_score is not None and not is_nan(overall_score):
            column_group_scores.append((column_group, overall_score))

    # Sort by score descending and take top N
    column_group_scores.sort(key=lambda x: x[1], reverse=True)
    return [cg for cg, _ in column_group_scores[:num_top]]


def select_top_constraints_for_group(
    scores_dict: Dict[ConstraintUID, FailPrecisionScore],
    num_top: int,
    overall_key: str
) -> Dict[ConstraintUID, FailPrecisionScore]:
    """
    Select top constraints for a single column group.
    
    Args:
        scores_dict: Constraint scores for a column group
        num_top: Number of top constraints to select
        overall_key: Key to exclude from constraint selection
        
    Returns:
        Dict mapping top constraint UIDs to their scores
    """
    # Filter out overall key and invalid scores
    constraint_items = [
        (constraint_uid, score)
        for constraint_uid, score in scores_dict.items()
        if constraint_uid != overall_key
           and score is not None
           and not is_nan(score)
    ]

    # Sort by score descending and keep top N
    constraint_items.sort(key=lambda x: x[1], reverse=True)
    top_constraints = constraint_items[:num_top]

    return {uid: score for uid, score in top_constraints}


def build_constraint_scores_mapping(
    suites: List[DVTrajectoryColumnGroupSuite],
    fail_precision_dict: Dict[ColumnGroup, Dict[ConstraintUID, FailPrecisionScore]]
) -> Dict[ConstraintUID, FailPrecisionScore]:
    """
    Build a mapping from constraint UIDs to their fail precision scores.

    Args:
        suites: List of DVTrajectoryColumnGroupSuite objects
        fail_precision_dict: Fail precision scores per column group and constraint

    Returns:
        Dict mapping constraint UIDs to fail precision scores
    """
    constraint_scores = {}

    for suite in suites:
        column_group = suite.column_group
        column_group_scores = fail_precision_dict.get(column_group, {})

        for trajectory in suite.trajectories:
            try:
                constraint_uid = trajectory.constraint.uid
                if constraint_uid in column_group_scores:
                    score = column_group_scores[constraint_uid]
                    if score is not None and not is_nan(score):
                        # If multiple trajectories share the same constraint_uid, keep the highest score
                        if constraint_uid not in constraint_scores or constraint_scores[constraint_uid] < score:
                            constraint_scores[constraint_uid] = float(score)
            except (AttributeError, Exception):
                continue

    return constraint_scores


def build_constraint_scores_mapping_from_selected(
    suites: List[DVTrajectoryColumnGroupSuite],
    top_fail_precision: Dict[ColumnGroup, Dict[ConstraintUID, FailPrecisionScore]]
) -> Dict[ConstraintUID, FailPrecisionScore]:
    """
    Build a mapping from constraint UIDs to their fail precision scores using only selected constraints.
    This ensures no NaN scores are included.

    Args:
        suites: List of DVTrajectoryColumnGroupSuite objects (already filtered to selected constraints)
        top_fail_precision: Selected constraints with their scores (no NaN scores)

    Returns:
        Dict mapping constraint UIDs to fail precision scores (no NaN scores)
    """
    constraint_scores = {}

    for suite in suites:
        column_group = suite.column_group
        selected_constraints = top_fail_precision.get(column_group, {})

        for trajectory in suite.trajectories:
            try:
                constraint_uid = trajectory.constraint.uid
                if constraint_uid in selected_constraints:
                    score = selected_constraints[constraint_uid]
                    # All scores in top_fail_precision are already filtered to be non-NaN
                    if constraint_uid not in constraint_scores or constraint_scores[constraint_uid] < score:
                        constraint_scores[constraint_uid] = float(score)
            except (AttributeError, Exception):
                continue

    return constraint_scores


def filter_suites_to_selected_constraints(
    suites: List[DVTrajectoryColumnGroupSuite],
    top_fail_precision: Dict[ColumnGroup, Dict[ConstraintUID, FailPrecisionScore]]
) -> List[DVTrajectoryColumnGroupSuite]:
    """
    Filter suites to only include trajectories with selected constraint_uids.
    Also deduplicates suites by column_group, keeping only one suite per column_group.
    
    Args:
        suites: List of suites to filter
        top_fail_precision: Dict mapping column groups to selected constraint_uids
        
    Returns:
        List of filtered and deduplicated suites
    """
    # Build set of selected constraint_uids per column_group
    selected_constraints_by_group = {}
    for column_group, constraints_dict in top_fail_precision.items():
        selected_constraints_by_group[column_group] = set(constraints_dict.keys())
    
    # Filter and deduplicate suites
    seen_column_groups = set()
    filtered_suites = []
    
    for suite in suites:
        column_group = suite.column_group
        
        # Skip if we've already processed this column_group
        if column_group in seen_column_groups:
            continue
        
        # Get selected constraint_uids for this column_group
        selected_uids = selected_constraints_by_group.get(column_group, set())
        
        if not selected_uids:
            continue
        
        # Filter trajectories to only include those with selected constraint_uids
        filtered_trajectories = []
        for trajectory in suite.trajectories:
            try:
                constraint_uid = trajectory.constraint.uid
                if constraint_uid in selected_uids:
                    filtered_trajectories.append(trajectory)
            except (AttributeError, Exception):
                continue
        
        # Only add suite if it has at least one matching trajectory
        if filtered_trajectories:
            filtered_suite = DVTrajectoryColumnGroupSuite(
                column_group=column_group,
                trajectories=filtered_trajectories
            )
            filtered_suites.append(filtered_suite)
            seen_column_groups.add(column_group)
    
    return filtered_suites


def build_candidate_list(
    column_group_suites: List[DVTrajectoryColumnGroupSuite],
    top_fail_precision: Dict[ColumnGroup, Dict[ConstraintUID, FailPrecisionScore]]
) -> List[Tuple[DVTrajectoryColumnGroupSuite, float]]:
    """
    Build a list of candidate column group suites with their fail precision scores.
    
    Args:
        column_group_suites: List of DVTrajectoryColumnGroupSuite objects
        top_fail_precision: Top constraints and scores for each column group
        
    Returns:
        List of (suite, score) tuples, one per (column_group, constraint_uid)
    """
    candidates = []

    for column_group, constraints in top_fail_precision.items():
        for constraint_uid, score in constraints.items():
            # Find the suite matching this constraint and column group
            matching_suite = find_suite_by_constraint(
                column_group_suites,
                column_group,
                constraint_uid,
            )

            # Add the suite if found
            if matching_suite:
                candidates.append((matching_suite, float(score)))

    return candidates


def sample_from_candidates(
    candidates: List[Tuple[DVTrajectoryColumnGroupSuite, float]],
    max_samples: int,
    temperature: float
) -> List[DVTrajectoryColumnGroupSuite]:
    """
    Sample column group suites from candidates based on their scores and temperature.
    
    Args:
        candidates: List of (suite, score) pairs
        max_samples: Maximum number of suites to sample
        temperature: Sampling temperature (0.0 = deterministic, >0.0 = stochastic)
        
    Returns:
        List of sampled DVTrajectoryColumnGroupSuite objects
    """
    suites, scores = zip(*candidates)
    scores_array = np.array(scores, dtype=float)
    num_to_sample = min(max_samples, len(candidates))

    if temperature == 0.0:
        # Deterministic: select top-n by score
        top_indices = np.argsort(-scores_array)[:num_to_sample]
    else:
        # Stochastic: sample using softmax with temperature
        top_indices = sample_with_softmax(
            scores_array,
            num_to_sample,
            temperature
        )

    return [suites[i] for i in top_indices]


def sample_with_softmax(
    scores: np.ndarray,
    num_samples: int,
    temperature: float
) -> np.ndarray:
    """
    Sample indices using softmax probability distribution.
    
    Args:
        scores: Array of scores
        num_samples: Number of samples to draw
        temperature: Temperature parameter for softmax
        
    Returns:
        Array of sampled indices
    """
    # Apply temperature scaling
    temp = max(temperature, 1e-8)  # Avoid division by zero
    logits = scores / temp

    # Numerical stability: subtract max before exponentiating
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum()

    # Sample without replacement
    return np.random.choice(
        len(scores),
        size=num_samples,
        replace=False,
        p=probabilities,
    )
