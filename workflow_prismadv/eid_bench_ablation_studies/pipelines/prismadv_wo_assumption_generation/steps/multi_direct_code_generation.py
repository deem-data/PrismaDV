import asyncio
from typing import List, Dict, FrozenSet

import oyaml as yaml

from prismadv.data_models.constraints_v2 import CodeEntry, SourceLocations
from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager
from prismadv.llm.langchain.models.prismadv.runtime import Runtime
from prismadv.post_processing.individual_constraint import determine_validity
from workflow_prismadv.eid_bench_ablation_studies.pipelines.prismadv_wo_assumption_generation.tasks import PrismaDVTasks


def generation(
        runtime: Runtime,
        column_desc_dict: Dict[str, dict],
        corr_groups: List[dict],
        vars: dict,
        multi_locs: Dict[FrozenSet[str], SourceLocations],
        num_retries: int = 10
):
    out: Dict[FrozenSet[str], List[CodeEntry]] = {}
    for info in corr_groups:
        key = frozenset(info["correlated_columns"])
        target_desc_dict = {c: column_desc_dict[c] for c in column_desc_dict if c in key}
        target_desc = yaml.dump(target_desc_dict, default_flow_style=False, sort_keys=False)
        focused = vars["code_script"].add_highlighted_line_numbers(multi_locs[key].sources)
        row_funcs = "\n".join(DeequFunctionManager().get_constraints(is_row_level=True))
        agg_funcs = "\n".join(DeequFunctionManager().get_constraints(is_row_level=False))

        payload = {
            "target_column": ", ".join(key),
            "target_column_desc": target_desc,
            "code_snippet": focused,
            "downstream_task_description": vars["downstream_task_description"],
            "multi_column_functions": "\n".join(
                DeequFunctionManager().get_constraints(can_be_used_for_multiple_columns=True)
            ),
            "row_level_functions": row_funcs,
            "aggregate_level_functions": agg_funcs,
        }
        print("\tGenerating code for correlated columns:", key)
        resp = runtime.run_task(PrismaDVTasks.MULTI_DIRECT_CODE_GENERATION, payload, num_retries)
        codes = [CodeEntry.from_dict(d) for d in resp["constraint_code"]]
        for code in codes:
            code.validity, code.reason_if_invalid = determine_validity(
                code.suggestion, vars["spark"],
                vars["data_sample"])
            print(code)
        out[key] = codes
    return out


async def ageneration(
        runtime: "Runtime",
        column_desc_dict: Dict[str, dict],
        corr_groups: List[dict],
        vars: dict,
        multi_locs: Dict[FrozenSet[str], "SourceLocations"],
        num_retries: int = 10,
        concurrency: int = 8,
) -> Dict[FrozenSet[str], List["CodeEntry"]]:
    """
    Async version of `generation`. Runs one task per corr_group with bounded concurrency.
    Falls back to threads for blocking calls (runtime.run_task, determine_validity).
    """
    out: Dict[FrozenSet[str], List["CodeEntry"]] = {}

    # Precompute static function catalogs once
    dfm = DeequFunctionManager()
    row_funcs = "\n".join(dfm.get_constraints(is_row_level=True))
    agg_funcs = "\n".join(dfm.get_constraints(is_row_level=False))
    multi_col_funcs = "\n".join(dfm.get_constraints(can_be_used_for_multiple_columns=True))

    sem = asyncio.Semaphore(concurrency)

    async def _process(info: dict):
        key: FrozenSet[str] = frozenset(info["correlated_columns"])
        target_desc_dict = {c: column_desc_dict[c] for c in column_desc_dict if c in key}
        target_desc = yaml.dump(target_desc_dict, default_flow_style=False, sort_keys=False)
        focused = vars["code_script"].add_highlighted_line_numbers(multi_locs[key].sources)

        payload = {
            "target_column": ", ".join(key),
            "target_column_desc": target_desc,
            "code_snippet": focused,
            "downstream_task_description": vars["downstream_task_description"],
            "multi_column_functions": multi_col_funcs,
            "row_level_functions": row_funcs,
            "aggregate_level_functions": agg_funcs,
        }

        async with sem:
            print("\tGenerating code for correlated columns:", key)
            # offload blocking task execution
            resp = await asyncio.to_thread(
                runtime.run_task, PrismaDVTasks.MULTI_DIRECT_CODE_GENERATION, payload, num_retries
            )

        codes = [CodeEntry.from_dict(d) for d in resp["constraint_code"]]

        # validate suggestions concurrently but off the loop
        async def _validate(code: "CodeEntry"):
            validity, reason = await asyncio.to_thread(
                determine_validity,
                code.suggestion,
                vars["spark"],
                vars["data_sample"],
            )
            code.validity = validity
            code.reason_if_invalid = reason
            print(code)
            return code

        validated_codes = await asyncio.gather(*[_validate(c) for c in codes])
        return key, validated_codes

    pairs = await asyncio.gather(*[_process(info) for info in corr_groups])
    for key, codes in pairs:
        out[key] = codes
    return out
