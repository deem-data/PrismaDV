import asyncio
from typing import Dict, List

import dspy
import oyaml as yaml

from prismadv.data_models.constraints_v2 import SourceLocations


class SingleColumnCodeDataFlowInspectorSig(dspy.Signature):
    """
    You are an expert in data-flow analysis and code inspection.
    Given a code snippet, a column name, and an optional sink variable,
    list every line that touches the column (read, write, filter, transform,
    merge, or pass-through). If a sink is supplied, include only lines that
    ultimately affect that sink; otherwise consider the entire snippet.

    Output only valid JSON with a "sources" array of contiguous 1-based ranges:
      {{
        "sources": [
          {{ "start_line": 2, "end_line": 2 }},
          {{ "start_line": 5, "end_line": 7 }}
        ]
      }}

    Rules:
      1) Use exact 1-based line numbers from the snippet.
      2) Be exhaustive; any interacting line appears in at least one range.
      3) Adjacent lines are merged into one range; non-adjacent are separate.
      4) Order of ranges does not matter.
      5) If sink_variable is empty, ignore sink and scan whole snippet.
      6) Output only JSON. No extra text.
    """
    code_script: str = dspy.InputField(
        description="The code snippet to analyze for data flow."
    )
    target_column: str = dspy.InputField(
        description="The specific column name to track in the code."
    )
    sink_variable: str = dspy.InputField(
        description="The optional sink variable to focus the analysis on. "
                    "If empty, treat all the columns as sinks."
    )
    sources: List[Dict[str, int]] = dspy.OutputField(
        description="List of line ranges that interact with the target column."
    )


def single(single_column_inspector, column_desc_dict: Dict[str, dict],
           columns_to_consider: List[str], vars: dict) -> Dict[str, SourceLocations]:
    out: Dict[str, SourceLocations] = {}
    for col in columns_to_consider:
        _ = yaml.dump({c: column_desc_dict[c] for c in column_desc_dict if c == col},
                      default_flow_style=False, sort_keys=False)  # reserved for future use
        inspector_vars = {
            "code_script": vars["code_script"],
            "target_column": col,
            "sink_variable": "Not provided",
        }
        print("\tInspecting data flow for column:", col)
        src_dict = single_column_inspector(**inspector_vars)
        out[col] = SourceLocations.from_dict(src_dict["sources"])
    return out


async def asingle(
        single_column_inspector,
        column_desc_dict: Dict[str, dict],
        columns_to_consider: List[str],
        vars: dict,
) -> Dict[str, "SourceLocations"]:
    out: Dict[str, "SourceLocations"] = {}

    async def inspect(col: str):
        _ = yaml.dump(
            {c: column_desc_dict[c] for c in column_desc_dict if c == col},
            default_flow_style=False,
            sort_keys=False,
        )  # reserved for future use
        inspector_vars = {
            "code_script": vars["code_script"],
            "target_column": col,
            "sink_variable": "Not provided",
        }
        print("\tInspecting data flow for column:", col)
        src_dict = await single_column_inspector.acall(**inspector_vars)
        return col, SourceLocations.from_dict(src_dict["sources"])

    results = await asyncio.gather(*(inspect(col) for col in columns_to_consider))
    for col, src in results:
        out[col] = src
    return out
