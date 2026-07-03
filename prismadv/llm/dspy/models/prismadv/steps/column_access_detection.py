from typing import List

import dspy


class ColumnAccessDetectionSig(dspy.Signature):
    """
    You are part of a task-aware data validation system that generates
    data quality constraints for specific downstream tasks by analyzing
    both the dataset and the code that processes it.

    You serve as the *Column Access Detection* component.
    Identify which columns from the dataset are accessed or utilized
    in the given code snippet so we can focus constraint generation
    on these relevant columns.

    Only return the column names exactly as they appear in the raw
    table definition (`columns_desc`). For derived columns, map them
    back and return only the original raw column names used to derive them.

    Return a list of comma-separated column names,
    e.g. `foo, bar, baz`.
    """
    columns_desc: str = dspy.InputField()
    code_script: str = dspy.InputField()
    downstream_task_description: str = dspy.InputField()
    columns: List[str] = dspy.OutputField()
