# steps/codegen.py (append)
import asyncio
from typing import Dict, List, Union, FrozenSet, Optional

import oyaml as yaml

from prismadv.data_models.constraints_v2 import AssumptionEntry
from prismadv.data_models.constraints_v2 import CodeEntry
from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager
from prismadv.llm.langchain.models.prismadv.runtime import Runtime
from prismadv.llm.langchain.models.prismadv.utils import batched_gather
from prismadv.llm.tasks import PrismaDVTasks
from prismadv.post_processing.individual_constraint import determine_validity


def _assumptions_text(assumptions: List["AssumptionEntry"]) -> str:
    return "\n".join([f"Assumption {i}: " + a.to_string() for i, a in enumerate(assumptions)])


def single(runtime: Runtime,
           columns_to_consider: List[str],
           column_desc_dict: Dict[str, dict],
           all_assumptions: Dict[str, List["AssumptionEntry"]],
           vars: dict,
           num_retries: int = 10) -> Dict[Union[str, FrozenSet[str]], List[CodeEntry]]:
    out: Dict[str, List[CodeEntry]] = {}
    row_funcs = "\n".join(DeequFunctionManager().get_constraints(is_row_level=True))
    agg_funcs = "\n".join(DeequFunctionManager().get_constraints(is_row_level=False))
    for col in columns_to_consider:
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
        resp = runtime.run_task(PrismaDVTasks.CODE_GENERATION, payload, num_retries)
        try:
            for i in range(len(resp["constraint_code"])):
                resp["constraint_code"][i]["source_assumptions"] = [
                    all_assumptions[col][j].uid for j in resp["constraint_code"][i].get("linked assumptions", [])
                ]
        except Exception as e:
            print(f"Error processing linked assumptions for column {col}: {e}")
            resp["constraint_code"] = []
        codes = [CodeEntry.from_dict(d) for d in resp["constraint_code"] if "suggestion" in d.keys()]
        for code in codes:
            code.validity, code.reason_if_invalid = determine_validity(code.suggestion, vars["spark"],
                                                                       vars["data_sample"])
            print(code)
        out[col] = codes
    return out


async def asingle(runtime: Runtime,
                  columns_to_consider: List[str],
                  column_desc_dict: Dict[str, dict],
                  all_assumptions: Dict[str, List["AssumptionEntry"]],
                  vars: dict,
                  num_retries: int = 10,
                  max_concurrent: Optional[int] = None) -> Dict[Union[str, FrozenSet[str]], List[CodeEntry]]:
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
        resp = await runtime.arun_task(PrismaDVTasks.CODE_GENERATION, payload, num_retries)
        try:
            for i in range(len(resp["constraint_code"])):
                resp["constraint_code"][i]["source_assumptions"] = [
                    all_assumptions[col][j].uid for j in resp["constraint_code"][i].get("linked assumptions", [])
                ]
        except Exception as e:
            print(f"Error processing linked assumptions for column {col}: {e}")
            resp["constraint_code"] = []
        raw = [CodeEntry.from_dict(d) for d in resp["constraint_code"] if "suggestion" in d.keys()]
        loop = asyncio.get_running_loop()

        async def _validate(code: CodeEntry):
            code.validity, code.reason_if_invalid = await loop.run_in_executor(
                None, determine_validity, code.suggestion, vars["spark"], vars["data_sample"]
            )
            print(code)
            return code

        validated = await batched_gather([_validate(c) for c in raw], max_concurrent=max_concurrent)
        return col, validated

    coros = [_gen(c) for c in columns_to_consider]
    pairs = await batched_gather(coros, max_concurrent=max_concurrent)
    return dict(pairs)
