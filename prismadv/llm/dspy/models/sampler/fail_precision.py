from collections import defaultdict
from typing import Dict, List, Tuple, Union, Any

import numpy as np

from prismadv.data_models.constraints_v2 import CodeEntry
from prismadv.data_models.trajectory import DVTrajectoryColumnGroupSuite

from prismadv.llm.dspy.models.sampler.types import ColumnGroup, ConstraintUID, FailPrecisionScore


def fail_precision(
    column_group_suites: List[DVTrajectoryColumnGroupSuite]
) -> Dict[ColumnGroup, Dict[Union[str, ConstraintUID], FailPrecisionScore]]:
    """
    Compute fail precision scores for trajectory column group suites.
    
    Fail precision measures how well a constraint identifies actual data quality issues:
    fail_precision = (correct failures) / (total failures)
    where a "correct failure" means the constraint failed validation AND the data is unsafe.
    
    A high fail precision score means when the constraint fails, it's usually correct
    (the data is actually unsafe). A low score means the constraint often fails on safe data
    (false positives).
    
    Args:
        column_group_suites: List of DVTrajectoryColumnGroupSuite objects
        
    Returns:
        Nested dict: {column_group: {constraint_uid: fail_precision_score, "overall": overall_score}}
    """
    # Step 1: Build evaluation matrix for each column group
    evaluation_matrices = build_evaluation_matrices(column_group_suites)

    # Step 2: Compute fail precision from evaluation matrices
    fail_precision_dict = compute_fail_precision_scores(evaluation_matrices)

    return fail_precision_dict


def build_evaluation_matrices(
    column_group_suites: List[DVTrajectoryColumnGroupSuite]
) -> Dict[ColumnGroup, Dict[Tuple[ConstraintUID, str], Dict[str, Any]]]:
    """
    Build evaluation matrices for computing fail precision.
    
    Args:
        column_group_suites: List of DVTrajectoryColumnGroupSuite objects
        
    Returns:
        Nested dict mapping column groups to constraint evaluation data:
        {column_group: {(constraint_uid, data_label): {prediction, is_safe, constraint}}}
    """
    evaluation_matrices = {}

    for suite in column_group_suites:
        column_group = suite.column_group
        trajectories = suite.trajectories

        if not trajectories:
            evaluation_matrices[column_group] = {}
            continue

        # Validate consistency of trajectories in this group
        reference_trajectory = trajectories[0]
        evaluation_matrix = {}

        for trajectory in trajectories:
            # Consistency checks: all trajectories should share key attributes
            assert trajectory.dataset_name == reference_trajectory.dataset_name, \
                "Inconsistent dataset_name in trajectory group"
            assert trajectory.script_path == reference_trajectory.script_path, \
                "Inconsistent script_path in trajectory group"
            assert trajectory.llm_name == reference_trajectory.llm_name, \
                "Inconsistent llm_name in trajectory group"
            assert isinstance(trajectory.constraint, CodeEntry), \
                "Trajectory constraints must be a CodeEntry instance"

            # Extract evaluation data
            constraint = trajectory.constraint
            processed_data_label = trajectory.processed_data_label
            is_safe = trajectory.is_safe

            # Get validation result, defaulting to False if unavailable
            try:
                validation_result = trajectory.validation_results.status
            except Exception:
                validation_result = False

            # Store evaluation data
            evaluation_key = (constraint.uid, processed_data_label)

            if evaluation_key in evaluation_matrix:
                print(f"Duplicate Key Found: {evaluation_key}")

            evaluation_matrix[evaluation_key] = {
                "prediction": validation_result,
                "is_safe": is_safe,
                "constraint": constraint.suggestion,
            }

        evaluation_matrices[column_group] = evaluation_matrix

    return evaluation_matrices


def compute_fail_precision_scores(
    evaluation_matrices: Dict[ColumnGroup, Dict[Tuple[ConstraintUID, str], Dict[str, Any]]]
) -> Dict[ColumnGroup, Dict[Union[str, ConstraintUID], FailPrecisionScore]]:
    """
    Compute fail precision scores from evaluation matrices.
    
    Args:
        evaluation_matrices: Evaluation data for each column group
        
    Returns:
        Nested dict with fail precision scores per column group and constraint
    """
    fail_precision_dict = defaultdict(dict)

    for column_group, evaluation_matrix in evaluation_matrices.items():
        # Collect (prediction, is_safe) pairs for computing fail precision
        all_pairs = []
        per_constraint_pairs = defaultdict(list)

        for (constraint_uid, _data_label), evaluation_data in evaluation_matrix.items():
            prediction = evaluation_data["prediction"]
            is_safe = evaluation_data["is_safe"]

            pair = (prediction, is_safe)
            all_pairs.append(pair)
            per_constraint_pairs[constraint_uid].append(pair)

        # Compute overall fail precision for this column group
        fail_precision_dict[column_group]["overall"] = compute_fail_precision(all_pairs)

        # Compute per-constraint fail precision within this column group
        for constraint_uid, pairs in per_constraint_pairs.items():
            fail_precision_dict[column_group][constraint_uid] = compute_fail_precision(pairs)

    return fail_precision_dict


def compute_fail_precision(pairs: List[Tuple[bool, bool]]) -> float:
    """
    Compute fail precision from (prediction, is_safe) pairs.
    
    Fail precision = (correct failures) / (total failures)
    A correct failure occurs when prediction is False AND is_safe is False.
    This means the constraint correctly identified that unsafe data failed validation.
    
    Args:
        pairs: List of (prediction, is_safe) tuples where:
            - prediction: Whether validation passed (True) or failed (False)
            - is_safe: Whether the data is actually safe (True) or unsafe (False)
            
    Returns:
        Fail precision score, or np.nan if no failures occurred
    """
    total_failures = 0
    correct_failures = 0

    for prediction, is_safe in pairs:
        if not prediction:  # Constraint failed validation
            total_failures += 1
            if not is_safe:  # Data was actually unsafe (correctly identified issue)
                correct_failures += 1

    return (correct_failures / total_failures) if total_failures > 0 else np.nan


def compute_overall_fail_precision_for_key(
    column_group_suites: List[DVTrajectoryColumnGroupSuite],
    fail_precision_dict: Dict[ColumnGroup, Dict[Union[str, ConstraintUID], FailPrecisionScore]]
) -> float:
    """
    Compute overall fail precision score for a trajectory key by aggregating across all column groups.
    
    This aggregates all (prediction, is_safe) pairs from all column groups to compute a true overall score.
    
    Args:
        column_group_suites: List of DVTrajectoryColumnGroupSuite objects for this key
        fail_precision_dict: Fail precision scores per column group (for fallback)
        
    Returns:
        Overall fail precision score, or np.nan if no valid pairs found
    """
    all_pairs = []
    
    # Collect all (prediction, is_safe) pairs across all column groups
    for suite in column_group_suites:
        for trajectory in suite.trajectories:
            try:
                validation_result = trajectory.validation_results.status
            except Exception:
                validation_result = False
            
            is_safe = trajectory.is_safe
            all_pairs.append((validation_result, is_safe))
    
    # Compute overall fail precision from all pairs
    if not all_pairs:
        return np.nan
    
    return compute_fail_precision(all_pairs)
