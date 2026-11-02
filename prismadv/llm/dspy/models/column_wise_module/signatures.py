"""DSPy signatures for constraint generation.

These signatures are designed to be concise and auto-optimizable by SIFTA.
"""

from typing import Any, Dict, List

import dspy

from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager

BASIC_INSTRUCTIONS = {
    "dataflow_inspector": (
        "List every line of the code that touches the target column: reads it, writes it, filters on it, "
        "transforms it, merges it, or passes it along. Use 1-based line numbers; put adjacent lines in one range.\n\n"
        'Output only JSON: { "sources": [{ "start_line": int, "end_line": int }, ...] }'
    ),
    "assumption_generation": (
        "Code makes assumptions about its input data, often without stating them. For example, fillna on a "
        "column means the column may contain nulls, and an assert sets a bound the data must meet. From the "
        "code that uses the target column and the task description, write these assumptions as statements "
        "about the input data that a Deequ check can test.\n\n"
        'Output only valid JSON (numbers without leading zeros): '
        '{ "assumptions": [{ "text": str, "sources": [{ "start_line": int, "end_line": int }] }, ...] }'
    ),
    "ir_generation": (
        "Translate the assumptions into PyDeequ checks. Set level \"error\" if data that fails the check would "
        "crash the code, and \"warning\" if the code still runs but the data quality is not ideal. Link each "
        "check to the indices (from 0) of the assumptions it tests.\n\n"
        'Output: {"constraint_code": [{"suggestion": PyDeequ code (starting with "." or a function), '
        '"level": "warning"|"error", "linked assumptions": [indices]}, ...]}\n\n'
        "Available Row-Level Functions:\n"
        + "\n".join(DeequFunctionManager().get_constraints(is_row_level=True))
        + "\n\nAvailable Aggregate-Level Functions:\n"
        + "\n".join(DeequFunctionManager().get_constraints(is_row_level=False))
    ),
}


class SingleColumnCodeDataFlowInspectorSig(dspy.Signature):
    __doc__ = BASIC_INSTRUCTIONS["dataflow_inspector"]
    code_script: str = dspy.InputField()
    target_column: str = dspy.InputField()
    sink_variable: str = dspy.InputField()
    sources: List[Dict[str, int]] = dspy.OutputField()


class AssumptionGenerationSig(dspy.Signature):
    __doc__ = BASIC_INSTRUCTIONS["assumption_generation"]
    target_column: str = dspy.InputField()
    target_column_desc: str = dspy.InputField()
    focused_code: str = dspy.InputField()
    downstream_task_description: str = dspy.InputField()

    assumptions: List[Dict[str, Any]] = dspy.OutputField()


class IRGenerationSig(dspy.Signature):
    __doc__ = BASIC_INSTRUCTIONS["ir_generation"]
    target_column: str = dspy.InputField()
    target_column_desc: str = dspy.InputField()
    code_snippet: str = dspy.InputField()
    downstream_task_description: str = dspy.InputField()
    assumptions: Any = dspy.InputField()

    constraint_code: List[Dict[str, Any]] = dspy.OutputField()
