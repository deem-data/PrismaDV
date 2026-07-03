import asyncio
from typing import Dict, List, Union, FrozenSet, Optional

import oyaml as yaml

from prismadv.data_models.constraints_v2 import AssumptionEntry, SourceLocations
from prismadv.llm.langchain.models.prismadv.runtime import Runtime
from prismadv.llm.langchain.models.prismadv.utils import batched_gather
from prismadv.llm.tasks import PrismaDVTasks


def single(runtime: Runtime,
           column_desc_dict: Dict[str, dict],
           columns_to_consider: List[str],
           single_locs: Dict[str, SourceLocations],
           vars: dict,
           num_retries: int = 10) -> Dict[Union[str, FrozenSet[str]], List[AssumptionEntry]]:
    out: Dict[str, List[AssumptionEntry]] = {}
    for col in columns_to_consider:
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
        resp = runtime.run_task(PrismaDVTasks.SINGLE_COLUMN_ASSUMPTION_GENERATION, payload, num_retries)
        out[col] = [AssumptionEntry.from_dict(d) for d in resp["assumptions"]]
    return out


async def asingle(runtime: Runtime,
                  column_desc_dict: Dict[str, dict],
                  columns_to_consider: List[str],
                  single_locs: Dict[str, SourceLocations],
                  vars: dict,
                  num_retries: int = 10,
                  max_concurrent: Optional[int] = None) -> Dict[Union[str, FrozenSet[str]], List[AssumptionEntry]]:
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
        resp = await runtime.arun_task(PrismaDVTasks.SINGLE_COLUMN_ASSUMPTION_GENERATION, payload, num_retries)
        return col, [AssumptionEntry.from_dict(d) for d in resp["assumptions"]]

    coros = [_gen(c) for c in columns_to_consider]
    pairs = await batched_gather(coros, max_concurrent=max_concurrent)
    return dict(pairs)
