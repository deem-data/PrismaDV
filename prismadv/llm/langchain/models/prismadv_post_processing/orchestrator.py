import logging

from prismadv.data_models import ConstraintsWithSources
from prismadv.data_models.config import PrismaDVConfig
from prismadv.llm.langchain.abstract import AbstractPrismaLangChainDV
from prismadv.llm.langchain.models.prismadv.costs import run_with_cost, arun_with_cost
from prismadv.llm.langchain.models.prismadv_post_processing.chains import build_chains
from prismadv.llm.langchain.models.prismadv_post_processing.runtime import Runtime
from prismadv.llm.langchain.models.prismadv_post_processing.steps import (
    code_fixing,
    code_consolidation
)
from prismadv.llm_backend.entry import get_langchain_model


class PrismaLangChainDVPostProcessing(AbstractPrismaLangChainDV):
    def __init__(
            self,
            model_name: str,
            downstream_task_description: str,
            temperature: float = 0.6,
            logger: logging.Logger = None
    ):
        if not model_name:
            raise ValueError("Model name is required.")
        self.model = get_langchain_model(model_name, temperature=temperature)
        self.downstream_task_description = downstream_task_description
        self.logger = logger
        self._chains = self._build_chains()
        self.runtime = Runtime(self._chains, logger=self.logger)

    @classmethod
    def from_config(cls, config: PrismaDVConfig, logger: logging.Logger = None):
        return cls(
            model_name=config.llm.model_name,
            downstream_task_description="",
            temperature=config.llm.temperature,
            logger=logger
        )

    def _build_chains(self):
        return build_chains(self.model, self.downstream_task_description)

    def invoke(self, input_variables: dict):
        cost_summary = {}
        fixing_results = {}
        existing_constraints: ConstraintsWithSources = input_variables['existing_constraints']

        with_code_correction = input_variables.get('with_code_correction', True)
        if with_code_correction:
            fixing_results = run_with_cost(
                "code_fixing",
                lambda: code_fixing.code_fixing(
                    self.runtime, input_variables
                ),
                cost_summary
            )
            constraints_after_fixing = self.merge_fixed_code(existing_constraints, fixing_results)
            input_variables['existing_constraints'] = constraints_after_fixing

        consolidation_results = run_with_cost(
            "consolidate_constraints",
            lambda: code_consolidation.general_consolidate_code(self.runtime, input_variables),
            cost_summary,
        )
        final_constraints = self.merge_consolidated_code(
            input_variables['existing_constraints'], consolidation_results
        )

        return final_constraints, fixing_results, consolidation_results, cost_summary

    async def ainvoke(self, input_variables: dict):
        cost_summary = {}
        fixing_results = {}
        existing_constraints: ConstraintsWithSources = input_variables['existing_constraints']

        with_code_correction = input_variables.get('with_code_correction', True)
        if with_code_correction:
            fixing_results = await arun_with_cost(
                "code_fixing",
                lambda: code_fixing.acode_fixing(self.runtime, input_variables),
                cost_summary,
            )
            constraints_after_fixing = self.merge_fixed_code(
                existing_constraints, fixing_results
            )
            input_variables['existing_constraints'] = constraints_after_fixing
        consolidation_results = await arun_with_cost(
            "consolidate_constraints",
            lambda: code_consolidation.ageneral_consolidate_code(self.runtime, input_variables),
            cost_summary,
        )
        final_constraints = self.merge_consolidated_code(
            input_variables['existing_constraints'], consolidation_results
        )
        return final_constraints, fixing_results, consolidation_results, cost_summary

    def merge_fixed_code(self, existing_constraints, fixing_results):
        new_constraints = existing_constraints.copy()
        data_map = new_constraints.data_map
        assert set(fixing_results.keys()).issubset(set(new_constraints.data_map.keys()))
        for column_group_key, fixings in fixing_results.items():
            # print(data_map[column_group_key].code)
            for target_code_uid, new_code in fixings.items():
                for i in range(len(data_map[column_group_key].code)):
                    if data_map[column_group_key].code[i].uid == target_code_uid:
                        data_map[column_group_key].code[i].validity = True
                        data_map[column_group_key].code[i].reason_if_invalid = ""
                        data_map[column_group_key].code[i].suggestion = new_code
                break
        new_constraints.data_map = data_map
        return new_constraints

    def merge_consolidated_code(self, existing_constraints, consolidation_results):
        new_constraints = existing_constraints.copy()
        data_map = new_constraints.data_map
        assert set(consolidation_results.keys()).issubset(set(new_constraints.data_map.keys()))
        for column_group_key, consolidation_result in consolidation_results.items():
            idxs_to_remove = [item['idx_to_remove'] for item in consolidation_result]
            data_map[column_group_key].code = [
                c for c in data_map[column_group_key].code
                if c.uid not in idxs_to_remove
            ]
        new_constraints.data_map = data_map
        return new_constraints
