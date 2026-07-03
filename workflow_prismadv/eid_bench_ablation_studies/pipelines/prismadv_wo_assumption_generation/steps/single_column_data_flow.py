import asyncio
from typing import Dict, List

import oyaml as yaml

from prismadv.data_models.constraints_v2 import SourceLocations


def single_column_data_flow_inspection(single_inspector, column_desc_dict: Dict[str, dict],
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
        src_dict = single_inspector.invoke_with_retries(input_variables=inspector_vars)
        out[col] = SourceLocations.from_dict(src_dict["sources"])
    return out


async def asingle_column_data_flow_inspection(single_inspector, column_desc_dict: Dict[str, dict],
                                              columns_to_consider: List[str], vars: dict) -> Dict[str, SourceLocations]:
    async def _inspect(col: str):
        _ = yaml.dump({c: column_desc_dict[c] for c in column_desc_dict if c == col},
                      default_flow_style=False, sort_keys=False)
        inspector_vars = {
            "code_script": vars["code_script"],
            "target_column": col,
            "sink_variable": "Not provided",
        }
        print("\tInspecting data flow for column:", col)
        src_dict = await single_inspector.ainvoke(input_variables=inspector_vars)
        return col, SourceLocations.from_dict(src_dict["sources"])

    pairs = await asyncio.gather(*[_inspect(c) for c in columns_to_consider])
    return {c: loc for c, loc in pairs}
