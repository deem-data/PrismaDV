"""DSPy modules for column-wise constraint generation."""

from typing import List

import dspy

from prismadv.data_models.constraints_v2 import AssumptionEntry, CodeEntry, SourceLocations
from prismadv.llm.dspy.models.column_wise_module.column_discovery import ColumnDiscoveryModule
from prismadv.llm.dspy.models.column_wise_module.signatures import (
    SingleColumnCodeDataFlowInspectorSig,
    AssumptionGenerationSig,
    IRGenerationSig,
)
from prismadv.llm.dspy.models.column_wise_module.human_designed_signatures import (
    HumanDesignedDataFlowInspectorSig,
    HumanDesignedAssumptionGenerationSig,
    HumanDesignedIRGenerationSig,
)


def _assumptions_text(assumptions: List[AssumptionEntry]) -> str:
    """Format assumptions as text for the IR generation step."""
    return "\n".join([f"Assumption {i}: " + a.to_string() for i, a in enumerate(assumptions)])


class ConstraintGenerationModule(dspy.Module):
    """Module to run dataflow, generate assumptions, then constraints."""

    def __init__(self):
        super().__init__()
        self.dataflow_inspector = dspy.Predict(SingleColumnCodeDataFlowInspectorSig)
        self.assumption_generation = dspy.Predict(AssumptionGenerationSig)
        self.ir_generation = dspy.Predict(IRGenerationSig)

    def forward(
            self,
            code_script,
            target_column: str,
            target_column_desc: str,
            downstream_task_description: str,
            **kwargs
    ):
        """
        Generate constraints for a single column.

        Args:
            code_script: CodeContainer or str - the code snippet to analyze
            target_column: str - the column name to generate constraints for
            target_column_desc: str - YAML-formatted description of the target column
            downstream_task_description: str - description of the downstream task

        Returns:
            dict with "assumptions" and "code" keys
        """
        from prismadv.data_models.code_container import CodeContainer
        if not isinstance(code_script, CodeContainer):
            code_script = CodeContainer(code_script)

        # 1) Run data-flow to find relevant line ranges for the target column
        dataflow_result = self.dataflow_inspector(
            code_script=str(code_script),
            target_column=target_column,
            sink_variable="Not provided",
        )

        # 2) Convert dataflow result to SourceLocations
        source_locations = SourceLocations.from_dict(dataflow_result.sources)

        # 3) Build focused_code with highlighted line numbers
        if source_locations.sources:
            focused_code = code_script.add_highlighted_line_numbers(source_locations.sources)
        else:
            focused_code = code_script.with_line_numbers()

        # 4) Generate assumptions on the focused code
        assumption_result = self.assumption_generation(
            target_column=target_column,
            target_column_desc=target_column_desc,
            focused_code=focused_code,
            downstream_task_description=downstream_task_description,
        )

        # Convert assumption dicts to AssumptionEntry objects
        assumption_entries = []
        for assumption_dict in assumption_result.assumptions:
            if not isinstance(assumption_dict, dict) or not assumption_dict or 'text' not in assumption_dict:
                continue
            try:
                assumption_entries.append(AssumptionEntry.from_dict(assumption_dict))
            except Exception:
                continue

        # 5) Generate constraints from assumptions
        constraint_result = self.ir_generation(
            target_column=target_column,
            target_column_desc=target_column_desc,
            code_snippet=code_script.with_line_numbers(),
            downstream_task_description=downstream_task_description,
            assumptions=_assumptions_text(assumption_entries),
        )

        # 6) Process constraint_code and link assumptions
        constraint_code = constraint_result.constraint_code
        raw_constraints = [d for d in constraint_code if "suggestion" in d.keys()]
        code_entries = []
        for constraint_dict in raw_constraints:
            linked_indices = constraint_dict.get("linked assumptions", [])
            source_assumptions = [
                assumption_entries[j].uid for j in linked_indices
                if 0 <= j < len(assumption_entries)
            ]
            code_entry = CodeEntry.from_dict({
                "suggestion": constraint_dict["suggestion"],
                "level": constraint_dict.get("level", "warning"),
                "source_assumptions": source_assumptions
            })
            code_entries.append(code_entry)

        return {
            "assumptions": assumption_entries,
            "code": code_entries
        }


class HumanDesignedConstraintGenerationModule(dspy.Module):
    """Module using human-designed signatures with detailed prompts.

    This module uses the same pipeline as ConstraintGenerationModule but with
    signatures that contain the full detailed prompts from the original LangChain
    implementation, preserving the examples and detailed instructions.
    """

    def __init__(self):
        super().__init__()
        self.dataflow_inspector = dspy.Predict(HumanDesignedDataFlowInspectorSig)
        self.assumption_generation = dspy.Predict(HumanDesignedAssumptionGenerationSig)
        self.ir_generation = dspy.Predict(HumanDesignedIRGenerationSig)

    def forward(
            self,
            code_script,
            target_column: str,
            target_column_desc: str,
            downstream_task_description: str,
            **kwargs
    ):
        """
        Generate constraints for a single column using human-designed prompts.

        Args:
            code_script: CodeContainer or str - the code snippet to analyze
            target_column: str - the column name to generate constraints for
            target_column_desc: str - YAML-formatted description of the target column
            downstream_task_description: str - description of the downstream task

        Returns:
            dict with "assumptions" and "code" keys
        """
        from prismadv.data_models.code_container import CodeContainer
        if not isinstance(code_script, CodeContainer):
            code_script = CodeContainer(code_script)

        # 1) Run data-flow to find relevant line ranges for the target column
        dataflow_result = self.dataflow_inspector(
            code_script=str(code_script),
            target_column=target_column,
            sink_variable="Not provided",
        )

        # 2) Convert dataflow result to SourceLocations
        source_locations = SourceLocations.from_dict(dataflow_result.sources)

        # 3) Build focused_code with highlighted line numbers
        if source_locations.sources:
            focused_code = code_script.add_highlighted_line_numbers(source_locations.sources)
        else:
            focused_code = code_script.with_line_numbers()

        # 4) Generate assumptions on the focused code
        assumption_result = self.assumption_generation(
            target_column=target_column,
            target_column_desc=target_column_desc,
            focused_code=focused_code,
            downstream_task_description=downstream_task_description,
        )

        # Convert assumption dicts to AssumptionEntry objects
        assumption_entries = []
        for assumption_dict in assumption_result.assumptions:
            if not isinstance(assumption_dict, dict) or not assumption_dict or 'text' not in assumption_dict:
                continue
            try:
                assumption_entries.append(AssumptionEntry.from_dict(assumption_dict))
            except Exception:
                continue

        # 5) Generate constraints from assumptions
        constraint_result = self.ir_generation(
            target_column=target_column,
            target_column_desc=target_column_desc,
            code_snippet=code_script.with_line_numbers(),
            downstream_task_description=downstream_task_description,
            assumptions=_assumptions_text(assumption_entries),
        )

        # 6) Process constraint_code and link assumptions
        constraint_code = constraint_result.constraint_code
        raw_constraints = [d for d in constraint_code if "suggestion" in d.keys()]
        code_entries = []
        for constraint_dict in raw_constraints:
            linked_indices = constraint_dict.get("linked assumptions", [])
            source_assumptions = [
                assumption_entries[j].uid for j in linked_indices
                if 0 <= j < len(assumption_entries)
            ]
            code_entry = CodeEntry.from_dict({
                "suggestion": constraint_dict["suggestion"],
                "level": constraint_dict.get("level", "warning"),
                "source_assumptions": source_assumptions
            })
            code_entries.append(code_entry)

        return {
            "assumptions": assumption_entries,
            "code": code_entries
        }