import asyncio
from typing import List, Dict, Any, FrozenSet

import dspy
import oyaml as yaml

from prismadv.data_models import AssumptionEntry
from prismadv.data_models.constraints_v2 import SourceLocations


class MultiColumnAssumptionGenerationSig(dspy.Signature):
    """
    You are the *Multi-Column Assumption Generation* component.

    Goal:
      Extract precise, actionable assumptions that involve multiple columns
      from the focused code and data characteristics for the downstream task.
      Assumptions must be suitable for Deequ or similar constraint languages.
      JSON numbers must not have leading zeros.

    """
    target_columns: List[str] = dspy.InputField(description="List of target column names involved in the assumptions.")
    target_columns_desc: str = dspy.InputField(description="YAML-formatted description of the target columns.")
    focused_code: str = dspy.InputField(description="Code snippet with line numbers highlighting relevant parts.")
    downstream_task_description: str = dspy.InputField(description="Description of the downstream task.")
    assumptions: List[Dict[str, Any]] = dspy.OutputField(description="""
    List of generated assumptions.
    [
        {{
        "text": "<Assumption 1>",
        "sources": [
            {{
                "start_line": 1,
                "end_line": 3 # should be integers representing the line numbers, don't add 0 as a prefix.
            }},
            {{
                ...
            }}
        ],
        }}
        {{
            "text": "<Assumption 2>",
            "sources": [...]
        }}
    ]""")


async def amulti(assumption_generator,
                 column_desc_dict: Dict[str, dict],
                 corr_groups: List[dict],
                 multi_locs: Dict[FrozenSet[str], "SourceLocations"],
                 vars: dict) -> Dict[FrozenSet[str], List["AssumptionEntry"]]:
    out: Dict[FrozenSet[str], List["AssumptionEntry"]] = {}

    async def _generate(info: dict):
        cols: List[str] = info["correlated_columns"]
        key: FrozenSet[str] = frozenset(cols)

        target_desc_dict = {c: column_desc_dict[c] for c in cols if c in column_desc_dict}
        target_desc = yaml.dump(target_desc_dict, default_flow_style=False, sort_keys=False)

        if key in multi_locs:
            focused_code = vars["code_script"].add_highlighted_line_numbers(multi_locs[key].sources)
        else:
            focused_code = vars["code_script"]

        payload = {
            "target_columns": ", ".join(cols),
            "target_columns_desc": target_desc,
            "focused_code": focused_code,
            "downstream_task_description": vars["downstream_task_description"],
        }

        print("\tGenerating assumptions for correlated columns:", cols)
        resp = await assumption_generator.acall(**payload)

        assumptions = []
        for i, d in enumerate(resp.get("assumptions", [])):
            try:
                assumptions.append(AssumptionEntry.from_dict(d))
            except Exception as e:
                print(f"\t  Warning: Skipping malformed assumption {i} for columns {cols}: {e}")
                continue

        return key, assumptions

    results = await asyncio.gather(*[_generate(info) for info in corr_groups])
    for key, assumptions in results:
        out[key] = assumptions
    return out
