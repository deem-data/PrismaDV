import asyncio
from collections import defaultdict
from typing import Dict, Union, FrozenSet, Any, List, Optional, Tuple

import oyaml as yaml

from prismadv.data_models import ConstraintsWithSources
from prismadv.llm.langchain.models.prismadv_post_processing.runtime import Runtime
from prismadv.llm.tasks import PrismaDVTasks


def general_consolidate_code(runtime: Runtime, vars: dict, num_retries: int = 10):
    out: Dict[Union[FrozenSet[str], str], List] = defaultdict(list)
    single_column_group_inputs = prepare_inputs(vars)
    for input_item in single_column_group_inputs:
        column_group_key = input_item["column_group_key"]
        idx_uid_map = input_item["idx_uid_map"]
        result = runtime.run_task(PrismaDVTasks.CODE_CONSOLIDATION, input_item, num_retries)
        try:
            result_data = result.get('results', [])
        except:
            result_data = []
        if len(result_data) == 0:
            continue
        mapped = [
            {
                'idx_to_remove': idx_uid_map[i['idx_to_remove']],
                'redundant_with': [idx_uid_map[j] for j in i['redundant_with'] if j in idx_uid_map]
            }
            for i in result_data
            if i['idx_to_remove'] in idx_uid_map
        ]
        out[column_group_key] = mapped
    return out


async def ageneral_consolidate_code(runtime: Runtime, vars: dict, num_retries: int = 10, concurrency: int = 8) -> Dict[
    Union[FrozenSet[str], str], List]:
    out: Dict[Union[FrozenSet[str], str], List] = defaultdict(list)
    single_column_group_inputs = prepare_inputs(vars)

    sem = asyncio.Semaphore(concurrency)

    async def _process(input_item) -> Optional[Tuple[Union[FrozenSet[str], str], List]]:
        async with sem:
            result = await asyncio.to_thread(
                runtime.run_task, PrismaDVTasks.CODE_CONSOLIDATION, input_item, num_retries
            )
            try:
                result_data = result.get('results', [])
            except:
                result_data = []
            if not result_data:
                return None

            idx_uid_map = input_item["idx_uid_map"]
            mapped = [
                {
                    'idx_to_remove': idx_uid_map[i['idx_to_remove']],
                    'redundant_with': [idx_uid_map[j] for j in i['redundant_with'] if j in idx_uid_map]
                }
                for i in result_data
                if i['idx_to_remove'] in idx_uid_map
            ]
            return input_item["column_group_key"], mapped

    results = await asyncio.gather(*[asyncio.create_task(_process(i)) for i in single_column_group_inputs])

    for item in results:
        if item is None:
            continue
        column_group_key, mapped_list = item
        out[column_group_key] = mapped_list

    return out


def prepare_inputs(vars):
    single_column_group_inputs = []
    # scenarios:
    #   1. Multicolumn constraints but not included multi columns, match column_names, if only one column name appears, remove it. No need to invoke llm.
    #   2. single/multi-column same constraints in different expression. lead to maintaining problem.
    #       1) completeness using both iscomplete and satisfies.
    #       2) SQL in different grammar.
    existing_constraints: ConstraintsWithSources = vars["existing_constraints"]
    column_desc_dict: Dict[str, Any] = vars["column_desc_dict"]
    for column_group_key, column_constraints in existing_constraints.data_map.items():
        keys = [column_group_key] if isinstance(column_group_key, str) else list(column_group_key)
        related_column_desc_dict = {k: column_desc_dict[k] for k in keys if k in column_desc_dict}
        related_column_desc = yaml.dump(
            related_column_desc_dict, default_flow_style=False, sort_keys=False
        )
        valid_code_entries = [c for c in column_constraints.code if c.validity]
        if len(valid_code_entries) <= 1:
            continue
        valid_code_entries_str, idx_uid_map = valid_code_entries_to_string(valid_code_entries,
                                                                           column_constraints.assumptions)
        if len(keys) > 1:
            print(keys, len(valid_code_entries))
        elif len(keys) == 1:
            print(keys, len(valid_code_entries))
        single_column_group_inputs.append(
            {
                "column_group_key": column_group_key,
                "column_names": str(column_group_key),
                "related_column_desc": related_column_desc,
                "valid_code_entries_str": valid_code_entries_str,
                "idx_uid_map": idx_uid_map,
            }
        )
    return single_column_group_inputs


def valid_code_entries_to_string(entries, assumptions):
    output = []
    idx_uid_map = {}
    for i, entry in enumerate(entries):
        assumption_sources = [
            a for a in assumptions
            if a.uid in entry.source_assumptions
        ]
        assumptions_source_str = "\n".join(a.to_string() for a in assumption_sources)
        entry_str = (
            f"Code {i}:\n"
            f"  Suggestion: {entry.suggestion}\n"
            f"  Validity: {entry.validity}\n"
            f"  Level: {entry.level}\n"
            f"  Source Assumptions: {assumptions_source_str}"
        )
        output.append(entry_str)
        idx_uid_map[i] = entry.uid
    return "\n".join(output), idx_uid_map
