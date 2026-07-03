"""Script-level constraint generation module for DSPy baselines."""

from __future__ import annotations

from typing import Any, Dict, List

import dspy

from prismadv.data_models.code_container import CodeContainer
from prismadv.data_models.constraints_v2 import (
    AssumptionEntry,
    CodeEntry,
    ColumnConstraintsWithSources,
    ConstraintsWithSources,
    de_column_group_key,
)
from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager


class ScriptLevelConstraintSig(dspy.Signature):
    """Generate constraints for all columns in a script in one pass."""

    __doc__ = f"""Generate PyDeequ validation constraints for all relevant columns.

    Available Row-Level Functions:
    {chr(10).join(DeequFunctionManager().get_constraints(is_row_level=True))}

    Available Aggregate-Level Functions:
    {chr(10).join(DeequFunctionManager().get_constraints(is_row_level=False))}
    """

    columns_desc: str = dspy.InputField(
        description="YAML description of all columns (schema + stats)."
    )
    code_script: str = dspy.InputField(
        description="Script code with 1-based line numbers for source references."
    )
    downstream_task_description: str = dspy.InputField(
        description="Description of the downstream task."
    )

    constraints: List[Dict[str, Any]] = dspy.OutputField(
        description=(
            "List of constraint blocks. Each block is a JSON object with: "
            "{\"column_group\": str|[str,...], "
            "\"assumptions\": [{\"text\": str, \"sources\": [{\"start_line\": int, \"end_line\": int}]}], "
            "\"code\": [{\"suggestion\": str, \"level\": \"error\"|\"warning\", "
            "\"linked_assumptions\": [int,...]}]}."
        )
    )


def _normalize_column_group(value: Any) -> str | frozenset:
    if isinstance(value, str):
        return de_column_group_key(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [str(item) for item in value if item is not None]
        if len(items) == 1:
            return items[0]
        return frozenset(items)
    return str(value)


def _parse_constraint_block(block: Dict[str, Any]) -> tuple[str | frozenset, ColumnConstraintsWithSources] | None:
    column_group_raw = block.get("column_group") or block.get("column") or block.get("column_group_key")
    if column_group_raw is None:
        return None

    column_group = _normalize_column_group(column_group_raw)

    assumptions = []
    for assumption_dict in block.get("assumptions", []) or []:
        if not isinstance(assumption_dict, dict):
            continue
        try:
            assumptions.append(AssumptionEntry.from_dict(assumption_dict))
        except Exception:
            continue

    code_entries = []
    for code_dict in block.get("code", []) or []:
        if not isinstance(code_dict, dict) or "suggestion" not in code_dict:
            continue

        linked = (
            code_dict.get("linked_assumptions")
            or code_dict.get("linked assumptions")
            or code_dict.get("source_assumptions")
            or []
        )

        source_assumptions: List[str] = []
        if isinstance(linked, list):
            for item in linked:
                if isinstance(item, int) and 0 <= item < len(assumptions):
                    source_assumptions.append(assumptions[item].uid)
                elif isinstance(item, str):
                    source_assumptions.append(item)

        code_entries.append(
            CodeEntry.from_dict(
                {
                    "suggestion": code_dict["suggestion"],
                    "level": code_dict.get("level", "warning"),
                    "source_assumptions": source_assumptions,
                }
            )
        )

    return column_group, ColumnConstraintsWithSources(assumptions=assumptions, code=code_entries)


def parse_constraints_prediction(prediction: Any) -> ConstraintsWithSources:
    """Convert a DSPy prediction into ConstraintsWithSources."""
    constraints_with_sources = ConstraintsWithSources()

    raw_constraints = None
    if isinstance(prediction, dict):
        raw_constraints = prediction.get("constraints")
    elif hasattr(prediction, "constraints"):
        raw_constraints = prediction.constraints

    if raw_constraints is None:
        return constraints_with_sources

    if isinstance(raw_constraints, dict):
        raw_constraints = raw_constraints.get("constraints", [])

    if not isinstance(raw_constraints, list):
        return constraints_with_sources

    for block in raw_constraints:
        if not isinstance(block, dict):
            continue
        parsed = _parse_constraint_block(block)
        if parsed is None:
            continue
        column_group, column_constraints = parsed
        if column_group not in constraints_with_sources.data_map:
            constraints_with_sources.data_map[column_group] = ColumnConstraintsWithSources()
        constraints_with_sources.data_map[column_group].assumptions.extend(column_constraints.assumptions)
        constraints_with_sources.data_map[column_group].code.extend(column_constraints.code)

    return constraints_with_sources


class ScriptLevelConstraintGenerationModule(dspy.Module):
    """Single-prompt script-level constraint generator."""

    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(ScriptLevelConstraintSig)

    def forward(
        self,
        code_script: str,
        columns_desc: str,
        downstream_task_description: str,
    ) -> ConstraintsWithSources:
        if not isinstance(code_script, CodeContainer):
            code_script = CodeContainer(code_script)
        prediction = self.generator(
            columns_desc=columns_desc,
            code_script=code_script.with_line_numbers(),
            downstream_task_description=downstream_task_description,
        )
        return parse_constraints_prediction(prediction)
