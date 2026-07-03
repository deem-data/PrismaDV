from typing import List, Dict, Any

import dspy


class ColumnCorrelationDiscoverySig(dspy.Signature):
    """
    You are part of a task-aware data validation system that generates
    data quality constraints for specific downstream tasks by analyzing
    both the dataset and the code that processes it.

    You serve as the *Column Correlation Discovery* component.
    Identify sets of columns that are correlated and require joint
    constraints based on their relationships and how they are used
    in the downstream code.

    Correlation categories include:
      - Consistency Constraint
      - Order/Range Dependency
      - Functional Dependency
      - Conditional Completeness / Exclusivity
      - Task-Driven Dependency
      - Temporal Consistency
      - Others

    Return a JSON array where each element has:
      - "correlated_columns": list of related column names
      - "correlation_type": type of correlation
    """
    columns_to_consider: List[str] = dspy.InputField(
        description="List of columns to consider for correlation detection.")
    considered_columns_desc: str = dspy.InputField(
        description="YAML description of the columns to consider.")
    code_script: str = dspy.InputField(
        description="The code snippet that processes the data.")
    downstream_task_description: str = dspy.InputField(
        description="Description of the downstream task.")
    correlated_groups: List[Dict[str, Any]] = dspy.OutputField(
        description="List of detected correlated column groups with their correlation types.")
