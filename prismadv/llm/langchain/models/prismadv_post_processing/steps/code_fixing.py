import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, FrozenSet
from typing import Optional

import oyaml as yaml

from prismadv.data_models import ConstraintsWithSources
from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager
from prismadv.llm.langchain.models.prismadv.runtime import Runtime
from prismadv.llm.tasks import PrismaDVTasks
from prismadv.post_processing.individual_constraint import determine_validity


def code_fixing(runtime: Runtime, vars: dict, num_retries: int = 3) -> Dict:
    out: Dict[Union[FrozenSet[str], str], Dict[str, str]] = defaultdict(dict)
    single_code_fixing_inputs, spark_validation, spark_validation_data = prepare_inputs(vars)
    for input_item in single_code_fixing_inputs:
        result = runtime.run_task(PrismaDVTasks.CODE_FIXING, input_item, num_retries)
        try:
            result_data = result.get('result', {}) or {}
        except:
            result_data = {}
        if not result_data.get('keep'):
            continue
        fixed_code = result_data.get('fixed_code')
        validity, reason = determine_validity(fixed_code, spark_validation, spark_validation_data)
        if not validity:
            continue
        column_group = input_item['column_group_key']
        code_uid = input_item['code_uid']
        out[column_group][code_uid] = fixed_code
    return out


async def acode_fixing(runtime: Runtime, vars: dict, num_retries: int = 3, concurrency: int = 8) -> Dict:
    out: Dict[Union[FrozenSet[str], str], Dict[str, str]] = defaultdict(dict)
    single_code_fixing_inputs, spark_validation, spark_validation_data = prepare_inputs(vars)

    sem = asyncio.Semaphore(concurrency)

    async def _process(input_item) -> Optional[Tuple[Union[FrozenSet[str], str], str, str]]:
        async with sem:
            result = await asyncio.to_thread(
                runtime.run_task, PrismaDVTasks.CODE_FIXING, input_item, num_retries
            )
            try:
                result_data = result.get('result', {}) or {}
            except:
                result_data = {}
            if not result_data.get('keep'):
                return None

            fixed_code = result_data.get('fixed_code')
            validity, _ = await asyncio.to_thread(
                determine_validity, fixed_code, spark_validation, spark_validation_data
            )
            if not validity:
                return None

            return input_item['column_group_key'], input_item['code_uid'], fixed_code

    results = await asyncio.gather(*[asyncio.create_task(_process(i)) for i in single_code_fixing_inputs])

    for item in results:
        if item is None:
            continue
        column_group, code_uid, fixed_code = item
        out[column_group][code_uid] = fixed_code

    return out


def prepare_inputs(vars: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Any, Any]:
    existing_constraints: ConstraintsWithSources = vars["existing_constraints"]
    column_desc_dict: Dict[str, Any] = vars["column_desc_dict"]
    spark_validation = vars["spark_validation"]
    spark_validation_data = vars["spark_validation_data"]

    single_code_fixing_inputs: List[Dict[str, Any]] = []

    for column_group_key, column_constraints in existing_constraints.data_map.items():
        keys = [column_group_key] if isinstance(column_group_key, str) else column_group_key

        related_column_desc_dict = {k: column_desc_dict[k] for k in keys if k in column_desc_dict}
        related_column_desc = yaml.dump(
            related_column_desc_dict, default_flow_style=False, sort_keys=False
        )

        for code_entry in (c for c in column_constraints.code if c.validity is False):
            reason = (code_entry.reason_if_invalid or "")
            if "skipping data type constraint" in reason.lower():
                continue

            assumption_sources = [
                a for a in column_constraints.assumptions
                if a.uid in code_entry.source_assumptions
            ]
            assumptions_source_str = "\n".join(a.to_string() for a in assumption_sources)

            relevant_schemas = DeequFunctionManager().get_relevant_constriants_from_code(
                code_entry.suggestion
            )
            relevant_schemas_str = ["\n".join(s.to_string() for s in relevant_schemas)]

            single_code_fixing_inputs.append({
                "column_group_key": column_group_key,
                "invalid_code": code_entry.suggestion,
                "code_level": code_entry.level,
                "code_uid": code_entry.uid,
                "error_message": code_entry.reason_if_invalid,
                "related_column_desc": related_column_desc,
                "assumptions_source_str": assumptions_source_str,
                "relevant_schemas_str": relevant_schemas_str,
            })

    return single_code_fixing_inputs, spark_validation, spark_validation_data
