import asyncio
from typing import Any
from typing import Dict, List, FrozenSet, Union

import dspy
import oyaml as yaml

from prismadv.data_models.constraints_v2 import AssumptionEntry
from prismadv.data_models.constraints_v2 import CodeEntry
from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager
from prismadv.post_processing.individual_constraint import determine_validity


class MultiColumnConstraintGenerationSig(dspy.Signature):
    """
    You are part of a task-aware data validation system. You serve as the
    *Code Generation On Multiple Columns* component.

    Goal: Translate multi-column natural-language requirements into executable
    validation code for tabular data, prioritizing row-level constraints via
    PyDeequ's `satisfies` when applicable.

    Guidance:
      - Prefer `.satisfies(<SQL-like boolean expression>, <name>, <agg_check>)`
        for row-level validations. The expression should be what follows SQL WHERE.
      - Optionally use predefined PyDeequ functions if they simplify row-level checks.
      - A constraint may link to multiple assumptions.
      - Skip assumptions that cannot be reasonably translated to constraints.
      - Use level "warning" by default; escalate to "error" only if violation
        risks crash or major downstream degradation.

    Output: JSON object with key "constraint_code", where each item contains:
      {{
        "suggestion": "<validation code string>",
        "level": "<error|warning>",
        "linked assumptions": [<0-based indices>]
      }}
    """
    multi_column_functions: str = dspy.InputField(
        description="Descriptions of available multi-column functions."
    )
    target_columns: List[str] = dspy.InputField(
        description="List of target column names involved in the assumptions."
    )
    target_columns_desc: str = dspy.InputField(
        description="YAML-formatted description of the target columns."
    )
    code_snippet: str = dspy.InputField(
        description="Code snippet with line numbers highlighting relevant parts."
    )
    downstream_task_description: str = dspy.InputField(
        description="Description of the downstream task."
    )
    assumptions: Any = dspy.InputField(
        description="List of discovered assumptions for the target columns."
    )
    constraint_code: List[Dict[str, Any]] = dspy.OutputField(
        description="List of generated constraint code snippets with metadata."
    )


def _assumptions_text(assumptions: List["AssumptionEntry"]) -> str:
    return "\n".join([f"Assumption {i}: " + a.to_string() for i, a in enumerate(assumptions)])


async def amulti(runtime: dspy.Predict,
                 columns_to_consider: List[str],
                 column_desc_dict: Dict[str, dict],
                 multi_assumptions: Dict[FrozenSet[str], List["AssumptionEntry"]],
                 vars: dict,
                 num_retries: int = 10) -> Dict[Union[str, FrozenSet[str]], List[CodeEntry]]:
    async def _gen(group: FrozenSet[str], assumptions: List["AssumptionEntry"]):
        target_desc = yaml.dump({c: column_desc_dict[c] for c in column_desc_dict if c in group},
                                default_flow_style=False, sort_keys=False)
        payload = {
            "multi_column_functions": "\n".join(
                DeequFunctionManager().get_constraints(can_be_used_for_multiple_columns=True)
            ),
            "target_columns": ", ".join(group),
            "target_columns_desc": target_desc,
            "code_snippet": vars["code_script"].with_line_numbers(),
            "downstream_task_description": vars["downstream_task_description"],
            "assumptions": _assumptions_text(assumptions),
        }
        print("\tGenerating constraint code for correlated columns:", group)
        resp = await runtime.acall(**payload)

        try:
            for i in range(len(resp["constraint_code"])):
                resp["constraint_code"][i]["source_assumptions"] = [
                    assumptions[j].uid for j in resp["constraint_code"][i].get("linked assumptions", [])
                ]
        except Exception as e:
            print(f"Error processing linked assumptions for correlated columns {group}: {e}")
            resp["constraint_code"] = []

        # Parse code entries with error handling
        raw_codes = []
        for i, d in enumerate(resp.get("constraint_code", [])):
            try:
                raw_codes.append(CodeEntry.from_dict(d))
            except Exception as e:
                print(f"\t  Warning: Skipping malformed code entry {i} for columns {group}: {e}")
                continue

        loop = asyncio.get_running_loop()

        async def _validate(code: CodeEntry):
            code.validity, code.reason_if_invalid = await loop.run_in_executor(
                None, determine_validity, code.suggestion, vars["spark"], vars["data_sample"]
            )
            return code

        validated = await asyncio.gather(*[_validate(c) for c in raw_codes])
        return group, validated

    pairs = await asyncio.gather(*[_gen(g, a) for g, a in multi_assumptions.items()])
    return dict(pairs)
