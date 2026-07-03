import asyncio
from typing import List, Dict, Any, Union, FrozenSet

import dspy
import oyaml as yaml

from prismadv.data_models.constraints_v2 import AssumptionEntry, SourceLocations


class SingleColumnAssumptionGenerationSig(dspy.Signature):
    """
    You are part of a task-aware data validation system and serve as the
    *Column Assumption Generation* component.

    Goal: Generate assumptions for the target column based on data characteristics,
    the provided focused code (data flow for the target column), and the downstream task.

    Guidance:
      - Infer implicit assumptions from code operations (e.g., fillna implies possible nulls).
      - Convert constraints on intermediate data into assumptions about the input data.
      - Describe assumptions so they can be implemented in Deequ or Great Expectations.
      - JSON numbers must not have leading zeros.

    Output format: JSON object with key "assumptions", value is an array of objects:
      {{
        "assumptions": [
          {{
            "text": "<Assumption>",
            "sources": [
              {{ "start_line": 1, "end_line": 3 }}
            ]
          }}
        ]
      }}
    """
    target_column: str = dspy.InputField(
        description="Name of the target column for which to generate assumptions."
    )
    target_column_desc: str = dspy.InputField(
        description="YAML-formatted description of the target column."
    )
    focused_code: str = dspy.InputField(
        description="Code snippet with line numbers highlighting relevant parts."
    )
    downstream_task_description: str = dspy.InputField(
        description="Description of the downstream task."
    )
    assumptions: List[Dict[str, Any]] = dspy.OutputField(
        description="List of generated assumptions for the target column."
    )


async def asingle(runtime: dspy.Predict,
                  column_desc_dict: Dict[str, dict],
                  columns_to_consider: List[str],
                  single_locs: Dict[str, SourceLocations],
                  vars: dict) -> Dict[Union[str, FrozenSet[str]], List[AssumptionEntry]]:
    async def _gen(col: str):
        target_desc = yaml.dump(
            {c: column_desc_dict[c] for c in column_desc_dict if c == col},
            default_flow_style=False, sort_keys=False
        )
        focused = (
            vars["code_script"].add_highlighted_line_numbers(single_locs[col].sources)
            if vars.get("cfg_use_dataflow")
            else vars["code_script"].with_line_numbers()
        )
        payload = {
            "target_column": col,
            "target_column_desc": target_desc,
            "focused_code": focused,
            "downstream_task_description": vars["downstream_task_description"]
        }
        print("\tGenerating assumptions for column:", col)
        resp = await runtime.acall(**payload)

        # Parse assumptions with error handling
        assumptions = []
        for i, d in enumerate(resp.get("assumptions", [])):
            try:
                assumptions.append(AssumptionEntry.from_dict(d))
            except Exception as e:
                print(f"\t  Warning: Skipping malformed assumption {i} for column {col}: {e}")
                continue

        return col, assumptions

    pairs = await asyncio.gather(*[_gen(c) for c in columns_to_consider])
    return dict(pairs)
