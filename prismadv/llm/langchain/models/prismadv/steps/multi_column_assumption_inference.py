import asyncio
from typing import Dict, List, FrozenSet, Union, Optional

import oyaml as yaml

from prismadv.data_models.constraints_v2 import AssumptionEntry, SourceLocations
from prismadv.llm.langchain.models.prismadv.runtime import Runtime
from prismadv.llm.langchain.models.prismadv.utils import batched_gather
from prismadv.llm.tasks import PrismaDVTasks


def multi(runtime: Runtime,
          column_desc_dict: Dict[str, dict],
          corr_groups: List[dict],
          multi_locs: Dict[FrozenSet[str], SourceLocations],
          vars: dict,
          num_retries: int = 10) -> Dict[Union[str, FrozenSet[str]], List[AssumptionEntry]]:
    out: Dict[FrozenSet[str], List[AssumptionEntry]] = {}
    for info in corr_groups:
        key = frozenset(info["correlated_columns"])
        target_desc_dict = {c: column_desc_dict[c] for c in column_desc_dict if c in key}
        target_desc = yaml.dump(target_desc_dict, default_flow_style=False, sort_keys=False)
        focused = vars["code_script"].add_highlighted_line_numbers(multi_locs[key].sources)
        payload = {
            "target_columns": ", ".join(key),
            "target_columns_desc": target_desc,
            "focused_code": focused,
            "downstream_task_description": vars["downstream_task_description"],
        }
        print("\tGenerating assumptions for correlated columns:", key)
        resp = runtime.run_task(PrismaDVTasks.MULTI_COLUMN_ASSUMPTION_GENERATION, payload, num_retries)
        out[key] = [AssumptionEntry.from_dict(d) for d in resp["assumptions"]]
    return out


async def amulti(runtime: Runtime,
                 column_desc_dict: Dict[str, dict],
                 corr_groups: List[dict],
                 multi_locs: Dict[FrozenSet[str], SourceLocations],
                 vars: dict,
                 num_retries: int = 10,
                 max_concurrent: Optional[int] = None) -> Dict[Union[str, FrozenSet[str]], List[AssumptionEntry]]:
    async def _gen(info: dict):
        key = frozenset(info["correlated_columns"])
        target_desc_dict = {c: column_desc_dict[c] for c in column_desc_dict if c in key}
        target_desc = yaml.dump(target_desc_dict, default_flow_style=False, sort_keys=False)
        focused = vars["code_script"].add_highlighted_line_numbers(multi_locs[key].sources)
        payload = {
            "target_columns": ", ".join(key),
            "target_columns_desc": target_desc,
            "focused_code": focused,
            "downstream_task_description": vars["downstream_task_description"],
        }
        print("\tGenerating assumptions for correlated columns:", key)
        resp = await runtime.arun_task(PrismaDVTasks.MULTI_COLUMN_ASSUMPTION_GENERATION, payload, num_retries)
        return key, [AssumptionEntry.from_dict(d) for d in resp["assumptions"]]

    coros = [_gen(g) for g in corr_groups]
    pairs = await batched_gather(coros, max_concurrent=max_concurrent)
    return dict(pairs)
