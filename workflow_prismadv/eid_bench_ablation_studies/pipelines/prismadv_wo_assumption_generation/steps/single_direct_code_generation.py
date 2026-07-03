import asyncio
from typing import List, Dict

import oyaml as yaml

from prismadv.data_models.constraints_v2 import CodeEntry, SourceLocations
from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager
from prismadv.llm.langchain.models.prismadv.runtime import Runtime
from prismadv.post_processing.individual_constraint import determine_validity
from workflow_prismadv.eid_bench_ablation_studies.pipelines.prismadv_wo_assumption_generation.tasks import PrismaDVTasks


def generation(
        runtime: Runtime,
        column_desc_dict: Dict[str, dict],
        columns_to_consider: List[str],
        vars: dict,
        single_locs: Dict[str, SourceLocations],
        num_retries: int = 10
):
    out: Dict[str, List[CodeEntry]] = {}
    for column in columns_to_consider:
        key = column
        target_desc_dict = {c: column_desc_dict[c] for c in column_desc_dict if c in [key]}
        target_desc = yaml.dump(target_desc_dict, default_flow_style=False, sort_keys=False)
        focused = vars["code_script"].add_highlighted_line_numbers(single_locs[key].sources)
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
        print("\tGenerating code for single columns:", key)
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
        columns_to_consider: List[str],
        vars: dict,
        single_locs: Dict[str, "SourceLocations"],
        num_retries: int = 10,
        concurrency: int = 8,
) -> Dict[str, List["CodeEntry"]]:
    """
    Async version of single-column generation with bounded concurrency.
    Offloads blocking calls (run_task, determine_validity) to threads.
    """
    out: Dict[str, List["CodeEntry"]] = {}

    # Precompute catalogs once
    dfm = DeequFunctionManager()
    row_funcs = "\n".join(dfm.get_constraints(is_row_level=True))
    agg_funcs = "\n".join(dfm.get_constraints(is_row_level=False))
    multi_col_funcs = "\n".join(dfm.get_constraints(can_be_used_for_multiple_columns=True))

    sem = asyncio.Semaphore(concurrency)

    async def _process(column: str):
        key = column
        target_desc_dict = {c: column_desc_dict[c] for c in column_desc_dict if c in [key]}
        target_desc = yaml.dump(target_desc_dict, default_flow_style=False, sort_keys=False)
        focused = vars["code_script"].add_highlighted_line_numbers(single_locs[key].sources)

        payload = {
            "target_column": key,  # single column
            "target_column_desc": target_desc,
            "code_snippet": focused,
            "downstream_task_description": vars["downstream_task_description"],
            "multi_column_functions": multi_col_funcs,
            "row_level_functions": row_funcs,
            "aggregate_level_functions": agg_funcs,
        }

        async with sem:
            print("\tGenerating code for single columns:", key)
            resp = await asyncio.to_thread(
                runtime.run_task, PrismaDVTasks.MULTI_DIRECT_CODE_GENERATION, payload, num_retries
            )

        codes = [CodeEntry.from_dict(d) for d in resp["constraint_code"]]

        async def _validate(code: "CodeEntry"):
            validity, reason = await asyncio.to_thread(
                determine_validity, code.suggestion, vars["spark"], vars["data_sample"]
            )
            code.validity = validity
            code.reason_if_invalid = reason
            print(code)
            return code

        validated_codes = await asyncio.gather(*[_validate(c) for c in codes])
        return key, validated_codes

    pairs = await asyncio.gather(*[_process(col) for col in columns_to_consider])
    for key, codes in pairs:
        out[key] = codes
    return out
