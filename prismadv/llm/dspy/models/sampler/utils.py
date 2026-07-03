from typing import Union, Any, List

import numpy as np

from prismadv.data_models.trajectory import DVTrajectoryColumnGroupSuite

from prismadv.llm.dspy.models.sampler.types import ColumnGroup, ConstraintUID


def format_score(score: Union[float, int]) -> str:
    """
    Format a score for display, handling NaN values.
    
    Args:
        score: The score to format
        
    Returns:
        Formatted score string
    """
    if score is None:
        return "NaN"
    try:
        if np.isnan(score):
            return "NaN"
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def get_constraint_text(
    suite: DVTrajectoryColumnGroupSuite,
    constraint_uid: ConstraintUID
) -> str:
    """
    Get the constraint text for a given constraint UID from a suite.
    
    Args:
        suite: The column group suite
        constraint_uid: The constraint UID to look up
        
    Returns:
        Constraint text string, or empty string if not found
    """
    for trajectory in suite.trajectories:
        try:
            if trajectory.constraint.uid == constraint_uid:
                return getattr(trajectory.constraint, "suggestion", "")
        except (AttributeError, Exception):
            continue
    return ""


def is_nan(value: Any) -> bool:
    """
    Safely check if a value is NaN without raising exceptions.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is NaN, False otherwise
    """
    try:
        return np.isnan(value)
    except (TypeError, ValueError):
        return False


def find_suite_by_constraint(
    column_group_suites: List[DVTrajectoryColumnGroupSuite],
    column_group: ColumnGroup,
    constraint_uid: ConstraintUID
) -> Union[DVTrajectoryColumnGroupSuite, None]:
    """
    Find a column group suite matching a specific column group and constraint UID.
    
    Args:
        column_group_suites: List of DVTrajectoryColumnGroupSuite objects
        column_group: The column group to match
        constraint_uid: The constraint UID to search for
        
    Returns:
        Matching DVTrajectoryColumnGroupSuite or None if not found
    """
    # Find the suite with matching column_group
    for suite in column_group_suites:
        if suite.column_group != column_group:
            continue

        # Check if any trajectory in this suite has the matching constraint_uid
        for trajectory in suite.trajectories:
            try:
                traj_uid = trajectory.constraint.uid
            except (AttributeError, Exception):
                traj_uid = None

            if traj_uid == constraint_uid:
                return suite

    return None
