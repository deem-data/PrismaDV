"""DSPy signatures for constraint generation.

These signatures are designed to be concise and auto-optimizable by SIFTA.
"""

from typing import Any, Dict, List

import dspy

from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager


class SingleColumnCodeDataFlowInspectorSig(dspy.Signature):
    """Identify lines where a target column is used in a code snippet.

    Inputs:
    - code_script: The code snippet to analyze.
    - target_column: The column name to track.
    - sink_variable: Optional sink variable to focus on. If empty, consider the whole snippet.

    Output:
    - sources: Return only valid JSON: { "sources": [{ "start_line": int, "end_line": int }, ... ] }.
      Use exact 1-based line numbers from the snippet.
      Include every line that reads, writes, filters, transforms, merges, or passes through the target column.
      Merge adjacent lines into one range; non-adjacent lines must be separate ranges.
      If sink_variable is not empty, include only lines that ultimately affect that sink.
    """
    code_script: str = dspy.InputField()
    target_column: str = dspy.InputField()
    sink_variable: str = dspy.InputField()
    sources: List[Dict[str, int]] = dspy.OutputField()


class AssumptionGenerationSig(dspy.Signature):
    """Generate assumptions for the target column.

    Inputs:
    - target_column: The name of the target column.
    - target_column_desc: Description of the target column.
    - focused_code: The relevant code snippet focusing on the target column.
    - downstream_task_description: Description of the downstream task.

    Output:
    - assumptions: List of assumption objects: { "text": str, "sources": [{ "start_line": int, "end_line": int }] }.
      Returned inside: { "assumptions": [...] }.
      JSON numbers must not have leading zeros.
    """
    target_column: str = dspy.InputField()
    target_column_desc: str = dspy.InputField()
    focused_code: str = dspy.InputField()
    downstream_task_description: str = dspy.InputField()

    assumptions: List[Dict[str, Any]] = dspy.OutputField()


class IRGenerationSig(dspy.Signature):
    __doc__ = f"""Generate PyDeequ validation code from assumptions and requirements.

    Inputs:
    - target_column: The name of the target column.
    - target_column_desc: Description of the target column.
    - code_snippet: The code snippet containing column usage.
    - downstream_task_description: Description of the downstream task.
    - assumptions: The assumptions generated for this column.

    Output:
    - constraint_code: List of constraint objects: {{"suggestion": PyDeequ code (starting with "." or a function), "level": "warning"|"error", "linked assumptions": [indices]}}.

    Available Row-Level Functions:
    {chr(10).join(DeequFunctionManager().get_constraints(is_row_level=True))}

    Available Aggregate-Level Functions:
    {chr(10).join(DeequFunctionManager().get_constraints(is_row_level=False))}
    """
    target_column: str = dspy.InputField()
    target_column_desc: str = dspy.InputField()
    code_snippet: str = dspy.InputField()
    downstream_task_description: str = dspy.InputField()
    assumptions: Any = dspy.InputField()

    constraint_code: List[Dict[str, Any]] = dspy.OutputField()