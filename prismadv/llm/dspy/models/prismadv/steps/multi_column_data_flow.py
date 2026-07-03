import asyncio
from typing import Dict, List, FrozenSet
from typing import TypedDict

import dspy

from prismadv.data_models.constraints_v2 import SourceLocations


class Range(TypedDict):
    start_line: int
    end_line: int


class MultiColumnDataFlowInspectorSig(dspy.Signature):
    """
    You are an expert in data-flow analysis and code inspection. Your task is to analyze
    the provided code snippet and identify data flow paths related to target columns.

    Return valid JSON with a "sources" array of contiguous 1-based ranges:
    {{
      "sources": [
        {{ "start_line": 2, "end_line": 2 }},
        {{ "start_line": 5, "end_line": 7 }}
      ]
    }}
    """
    code_script: str = dspy.InputField(
        description="The code snippet to analyze."
    )
    target_columns: List[str] = dspy.InputField(
        description="List of target column names to track."
    )
    sink_variable: str = dspy.InputField(
        description="Optional sink variable to focus the analysis on."
    )
    sources: List[Range] = dspy.OutputField(
        description='List of {"start_line": int, "end_line": int} ranges.'
    )


def multi(multi_column_inspector,
          column_desc_dict: Dict[str, dict],
          corr_groups: List[dict],
          vars: dict):
    out: Dict[FrozenSet[str], SourceLocations] = {}
    for info in corr_groups:
        cols: List[str] = info["correlated_columns"]
        single_locs = vars["single_column_data_flow"]
        relevant = [single_locs[c] for c in cols if c in single_locs]
        source_basis = SourceLocations.merge_sources(*relevant) if relevant else SourceLocations.from_dict([])
        inspector_vars = {
            "target_columns": ", ".join(cols),
            "code_script": vars["code_script"].add_highlighted_line_numbers(source_basis.sources),
            "correlated_columns": ", ".join(cols),
            "sink_variable": info.get("sink_variable", "Not provided"),
        }
        print("\tInspecting data flow for correlated columns:", cols)
        src = multi_column_inspector(**inspector_vars)
        out[frozenset(cols)] = SourceLocations.from_dict(src["sources"])


async def amulti(multi_column_inspector,
                 column_desc_dict: Dict[str, dict],
                 corr_groups: List[dict],
                 vars: dict) -> Dict[FrozenSet[str], SourceLocations]:
    out: Dict[FrozenSet[str], SourceLocations] = {}
    single_locs = vars["single_column_data_flow"]

    async def _inspect(info: dict):
        cols: List[str] = info["correlated_columns"]
        relevant = [single_locs[c] for c in cols if c in single_locs]
        source_basis = (
            SourceLocations.merge_sources(*relevant)
            if relevant else
            SourceLocations.from_dict([])
        )
        inspector_vars = {
            "target_columns": ", ".join(cols),
            "code_script": vars["code_script"].add_highlighted_line_numbers(source_basis.sources),
            "correlated_columns": ", ".join(cols),
            "sink_variable": info.get("sink_variable", "Not provided"),
        }
        print("\tInspecting data flow for correlated columns:", cols)
        src = await multi_column_inspector.acall(**inspector_vars)
        return frozenset(cols), SourceLocations.from_dict(src["sources"])

    results = await asyncio.gather(*[_inspect(info) for info in corr_groups])
    for key, src_locs in results:
        out[key] = src_locs
    return out
