import logging

from prismadv.code_inspector.llm_code_inspector.multiple.model import \
    ColumnDataFlowInspector as MultiColumnDataFlowInspector
from prismadv.code_inspector.llm_code_inspector.single.model import \
    ColumnDataFlowInspector as SingleColumnDataFlowInspector
from prismadv.data_models.config import PrismaDVConfig
from prismadv.data_models.constraints_v2 import ConstraintsWithSources, de_column_group_key, \
    ColumnConstraintsWithSources
from prismadv.llm.langchain.abstract import AbstractPrismaLangChainDV
from prismadv.llm.langchain.models.prismadv.costs import run_with_cost, arun_with_cost
from prismadv.llm_backend.entry import get_langchain_model
from workflow_prismadv.eid_bench_ablation_studies.pipelines.prismadv_wo_assumption_generation.chains import build_chains
from workflow_prismadv.eid_bench_ablation_studies.pipelines.prismadv_wo_assumption_generation.runtime import Runtime
from workflow_prismadv.eid_bench_ablation_studies.pipelines.prismadv_wo_assumption_generation.steps import (
    column_access,
    column_correlation,
    single_column_data_flow,
    multi_column_data_flow,
    single_direct_code_generation,
    multi_direct_code_generation,
)


class PrismaLangChainDVwoAssumption(AbstractPrismaLangChainDV):
    def __init__(
            self,
            model_name: str,
            downstream_task_description: str,
            use_column_correlation_detection: bool = False,
            temperature: float = 0.6,
            logger: logging.Logger = None
    ):
        if not model_name:
            raise ValueError("Model name is required.")
        self.model = get_langchain_model(model_name, temperature=temperature)
        self.downstream_task_description = downstream_task_description
        self.logger = logger
        self.single_column_inspector = SingleColumnDataFlowInspector(model_name=model_name)
        self.multi_column_inspector = MultiColumnDataFlowInspector(model_name=model_name)
        self.use_column_correlation_detection = use_column_correlation_detection
        self._chains = self._build_chains()
        self.runtime = Runtime(self._chains, logger=self.logger)

    @classmethod
    def from_config(cls, config: PrismaDVConfig, logger: logging.Logger = None):
        return cls(
            model_name=config.llm.model_name,
            use_column_correlation_detection=config.model.correlation_detection,
            downstream_task_description=config.model.downstream_task_description,
            temperature=config.llm.temperature,
            logger=logger
        )

    def _build_chains(self):
        return build_chains(self.model, self.downstream_task_description)

    def invoke(self, input_variables: dict):
        cost_summary = {}
        all_codes = {}
        column_desc_dict = input_variables["column_desc_dict"]
        input_variables[
            "downstream_task_description"] = self.downstream_task_description  # TODO: generate it before invoke
        columns_to_consider = run_with_cost(
            "column_access_detection",
            lambda: column_access.basic_column_access_detection(self.runtime, input_variables),
            cost_summary,
        )
        column_data_flow_locations = run_with_cost(
            "data_flow_inspection",
            lambda: single_column_data_flow.single_column_data_flow_inspection(
                self.single_column_inspector, column_desc_dict, columns_to_consider, input_variables
            ),
            cost_summary,
        )
        if self.use_column_correlation_detection:
            correlated_column_groups_info = run_with_cost(
                "column_correlation_discovery",
                lambda: column_correlation.discovery(
                    self.runtime, column_desc_dict, columns_to_consider, input_variables
                ),
                cost_summary,
            )
            input_variables["single_column_data_flow"] = column_data_flow_locations
            multi_locs = run_with_cost(
                "multi_column_data_flow_inspection",
                lambda: multi_column_data_flow.multi_column_data_flow_inspection(
                    self.multi_column_inspector, column_desc_dict, correlated_column_groups_info, input_variables
                ),
                cost_summary,
            )
            input_variables["multi_column_data_flow"] = multi_locs
            all_multi_column_codes = run_with_cost(
                "multi_column_code_generation",
                lambda: multi_direct_code_generation.generation(
                    self.runtime,
                    column_desc_dict,
                    correlated_column_groups_info,
                    multi_locs,
                    input_variables,
                ),
                cost_summary,
            )
            all_codes.update(all_multi_column_codes)
        all_single_column_codes = run_with_cost(
            "single_column_code_generation",
            lambda: single_direct_code_generation.generation(
                self.runtime,
                column_desc_dict,
                columns_to_consider,
                column_data_flow_locations,
                input_variables,
            ),
            cost_summary,
        )
        all_codes.update(all_single_column_codes)
        constraints_with_sources = ConstraintsWithSources.from_assumptions_and_code_dict(
            all_assumptions, all_codes
        )
        return constraints_with_sources, column_data_flow_locations, cost_summary

    async def ainvoke(self, input_variables: dict):
        cost_summary = {}
        all_codes = {}
        column_desc_dict = input_variables["column_desc_dict"]
        input_variables[
            "downstream_task_description"] = self.downstream_task_description  # TODO: generate it before invoke
        columns_to_consider = run_with_cost(
            "column_access_detection",
            lambda: column_access.basic_column_access_detection(self.runtime, input_variables),
            cost_summary,
        )
        column_data_flow_locations = await arun_with_cost(
            "data_flow_inspection",
            lambda: single_column_data_flow.asingle_column_data_flow_inspection(
                self.single_column_inspector,
                column_desc_dict,
                columns_to_consider,
                input_variables),
            cost_summary,
        )
        if self.use_column_correlation_detection:
            correlated_column_groups_info = run_with_cost(
                "column_correlation_discovery",
                lambda: column_correlation.discovery(
                    self.runtime, column_desc_dict, columns_to_consider, input_variables
                ),
                cost_summary,
            )
            input_variables["single_column_data_flow"] = column_data_flow_locations

            multi_locs = await arun_with_cost(
                "multi_column_data_flow_inspection",
                lambda: multi_column_data_flow.amulti_column_data_flow_inspection(
                    self.multi_column_inspector,
                    column_desc_dict,
                    correlated_column_groups_info,
                    input_variables,
                ),
                cost_summary,
            )
            input_variables["multi_column_data_flow"] = multi_locs
            all_multi_column_codes = await arun_with_cost(
                "multi_column_code_generation",
                lambda: multi_direct_code_generation.ageneration(
                    self.runtime,
                    column_desc_dict=column_desc_dict,
                    corr_groups=correlated_column_groups_info,
                    multi_locs=multi_locs,
                    vars=input_variables,
                ),
                cost_summary,
            )
            all_codes.update(all_multi_column_codes)
        all_single_column_codes = await arun_with_cost(
            "single_column_code_generation",
            lambda: single_direct_code_generation.ageneration(
                runtime=self.runtime,
                column_desc_dict=column_desc_dict,
                columns_to_consider=columns_to_consider,
                single_locs=column_data_flow_locations,
                vars=input_variables,
            ),
            cost_summary,
        )
        all_codes.update(all_single_column_codes)
        code_dict = all_codes
        constraints_with_sources = ConstraintsWithSources()
        code_dict = {de_column_group_key(k): v for k, v in code_dict.items()}
        for column_group, _ in code_dict.items():
            code_entries = code_dict[column_group]
            constraints_with_sources.data_map[column_group] = ColumnConstraintsWithSources(
                assumptions=[],
                code=code_entries,
            )
        return constraints_with_sources, column_data_flow_locations, cost_summary
