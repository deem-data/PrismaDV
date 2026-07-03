import asyncio
from typing import Dict, List, FrozenSet

from prismadv.data_models.constraints_v2 import SourceLocations


def multi_column_data_flow_inspection(multi_inspector,
                                      column_desc_dict: Dict[str, dict],
                                      corr_groups: List[dict],
                                      vars: dict) -> Dict[FrozenSet[str], SourceLocations]:
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
        src = multi_inspector.invoke_with_retries(input_variables=inspector_vars)
        out[frozenset(cols)] = SourceLocations.from_dict(src["sources"])
    return out


async def amulti_column_data_flow_inspection(multi_inspector,
                                             column_desc_dict: Dict[str, dict],
                                             corr_groups: List[dict],
                                             vars: dict) -> Dict[
    FrozenSet[str], SourceLocations]:
    async def _inspect(info: dict):
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
        src = await multi_inspector.ainvoke_with_retries(input_variables=inspector_vars)
        return frozenset(cols), SourceLocations.from_dict(src["sources"])

    pairs = await asyncio.gather(*[_inspect(g) for g in corr_groups])
    return dict(pairs)
