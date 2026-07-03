import asyncio
from typing import List, Dict, Any, Union, FrozenSet

import dspy
import oyaml as yaml

from prismadv.data_models.constraints_v2 import AssumptionEntry
from prismadv.data_models.constraints_v2 import CodeEntry
from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager
from prismadv.post_processing.individual_constraint import determine_validity


class IRGenerationSig(dspy.Signature):
    """
    You are the *Code Generation* component of a task-aware data validation system.
    Translate natural-language requirements into executable PyDeequ validation code.

    Use PyDeequ row-level 'satisfies' when possible (expression equals a SQL WHERE).
    You may use predefined row-level helpers if suitable: row_level_functions.

    For aggregate checks (e.g., mean, stddev, distinct counts), use: aggregate_level_functions

    Each item has:
        - "suggestion": string of PyDeequ code
        - "level": "error" or "warning" (default to "warning"; use "error" if violation risks crash or major failure)
        - "linked assumptions": list of 0-based indices mapping back to assumptions
    """
    row_level_functions: str = dspy.InputField(
        description="Descriptions of available row-level functions."
    )
    aggregate_level_functions: str = dspy.InputField(
        description="Descriptions of available aggregate-level functions."
    )
    target_column: str = dspy.InputField(
        description="Name of the target column for which to generate constraints."
    )
    target_column_desc: str = dspy.InputField(
        description="YAML-formatted description of the target column."
    )
    code_snippet: str = dspy.InputField(
        description="Code snippet with line numbers highlighting relevant parts."
    )
    downstream_task_description: str = dspy.InputField(
        description="Description of the downstream task."
    )
    assumptions: Any = dspy.InputField(
        description="List of discovered assumptions for the target column."
    )
    constraint_code: List[Dict[str, Any]] = dspy.OutputField(
        description="List of generated constraint snippets with metadata. The “suggestion” field contains the PyDeequ code, starting with either “.” or the function name (e.g., .isComplete(\"column_name\"))."
    )


def _assumptions_text(assumptions: List["AssumptionEntry"]) -> str:
    return "\n".join([f"Assumption {i}: " + a.to_string() for i, a in enumerate(assumptions)])


async def asingle(runtime: dspy.Predict,
                  columns_to_consider: List[str],
                  column_desc_dict: Dict[str, dict],
                  all_assumptions: Dict[str, List["AssumptionEntry"]],
                  vars: dict,
                  num_retries: int = 10) -> Dict[Union[str, FrozenSet[str]], List[CodeEntry]]:
    row_funcs = "\n".join(DeequFunctionManager().get_constraints(is_row_level=True))
    agg_funcs = "\n".join(DeequFunctionManager().get_constraints(is_row_level=False))

    async def _gen(col: str):
        target_desc = yaml.dump({c: column_desc_dict[c] for c in column_desc_dict if c == col},
                                default_flow_style=False, sort_keys=False)
        payload = {
            "row_level_functions": row_funcs,
            "aggregate_level_functions": agg_funcs,
            "target_column": col,
            "target_column_desc": target_desc,
            "code_snippet": vars["code_script"].with_line_numbers(),
            "assumptions": _assumptions_text(all_assumptions[col]),
            "downstream_task_description": vars["downstream_task_description"]
        }
        print("\tGenerating constraint code...")
        resp = await runtime.acall(**payload)
        try:
            for i in range(len(resp["constraint_code"])):
                resp["constraint_code"][i]["source_assumptions"] = [
                    all_assumptions[col][j].uid for j in resp["constraint_code"][i].get("linked assumptions", [])
                ]
        except Exception as e:
            print(f"Error processing linked assumptions for column {col}: {e}")
            resp["constraint_code"] = []

        # Parse code entries with error handling
        raw = []
        for i, d in enumerate(resp.get("constraint_code", [])):
            try:
                raw.append(CodeEntry.from_dict(d))
            except Exception as e:
                print(f"\t  Warning: Skipping malformed code entry {i} for column {col}: {e}")
                continue
        loop = asyncio.get_running_loop()

        async def _validate(code: CodeEntry):
            code.validity, code.reason_if_invalid = await loop.run_in_executor(
                None, determine_validity, code.suggestion, vars["spark"], vars["data_sample"]
            )
            return code

        validated = await asyncio.gather(*[_validate(c) for c in raw])
        return col, validated

    pairs = await asyncio.gather(*[_gen(c) for c in columns_to_consider])
    return dict(pairs)
