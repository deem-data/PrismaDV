# steps/codegen.py
import asyncio
from typing import Dict, List, FrozenSet, Union, Optional

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


def multi(runtime: Runtime,
          columns_to_consider: List[str],  # unused but keep signature parity
          column_desc_dict: Dict[str, dict],
          multi_assumptions: Dict[FrozenSet[str], List["AssumptionEntry"]],
          vars: dict,
          num_retries: int = 10) -> Dict[Union[str, FrozenSet[str]], List[CodeEntry]]:
    out: Dict[FrozenSet[str], List[CodeEntry]] = {}
    for group, assumptions in multi_assumptions.items():
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
        resp = runtime.run_task(PrismaDVTasks.MULTI_COLUMN_CODE_GENERATION, payload, num_retries)

        try:
            for i in range(len(resp["constraint_code"])):
                resp["constraint_code"][i]["source_assumptions"] = [
                    assumptions[j].uid for j in resp["constraint_code"][i].get("linked assumptions", [])
                ]
        except Exception as e:
            print(f"Error processing linked assumptions for correlated columns {group}: {e}")
            resp["constraint_code"] = []

        codes = [CodeEntry.from_dict(d) for d in resp["constraint_code"] if "suggestion" in d.keys()]
        for code in codes:
            code.validity, code.reason_if_invalid = determine_validity(
                code.suggestion, vars["spark"],
                vars["data_sample"])
            print(code)
        out[group] = codes
    return out


async def amulti(runtime: Runtime,
                 columns_to_consider: List[str],
                 column_desc_dict: Dict[str, dict],
                 multi_assumptions: Dict[FrozenSet[str], List["AssumptionEntry"]],
                 vars: dict,
                 num_retries: int = 10,
                 max_concurrent: Optional[int] = None) -> Dict[Union[str, FrozenSet[str]], List[CodeEntry]]:
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
        resp = await runtime.arun_task(PrismaDVTasks.MULTI_COLUMN_CODE_GENERATION, payload, num_retries)

        try:
            for i in range(len(resp["constraint_code"])):
                resp["constraint_code"][i]["source_assumptions"] = [
                    assumptions[j].uid for j in resp["constraint_code"][i].get("linked assumptions", [])
                ]
        except Exception as e:
            print(f"Error processing linked assumptions for correlated columns {group}: {e}")
            resp["constraint_code"] = []

        raw_codes = [CodeEntry.from_dict(d) for d in resp["constraint_code"] if "suggestion" in d.keys()]

        loop = asyncio.get_running_loop()

        async def _validate(code: CodeEntry):
            code.validity, code.reason_if_invalid = await loop.run_in_executor(
                None, determine_validity, code.suggestion, vars["spark"], vars["data_sample"]
            )
            print(code)
            return code

        validated = await batched_gather([_validate(c) for c in raw_codes], max_concurrent=max_concurrent)
        return group, validated

    coros = [_gen(g, a) for g, a in multi_assumptions.items()]
    pairs = await batched_gather(coros, max_concurrent=max_concurrent)
    return dict(pairs)
