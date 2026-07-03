import logging

from prismadv.code_inspector.llm_code_inspector.multiple.model import \
    ColumnDataFlowInspector as MultiColumnDataFlowInspector
from prismadv.code_inspector.llm_code_inspector.single.model import \
    ColumnDataFlowInspector as SingleColumnDataFlowInspector
from prismadv.data_models.config import PrismaDVConfig
from prismadv.data_models.constraints_v2 import ConstraintsWithSources, SourceLocations
from prismadv.llm.langchain.abstract import AbstractPrismaLangChainDV
from prismadv.llm.langchain.models.prismadv.chains import build_chains
from prismadv.llm.langchain.models.prismadv.costs import run_with_cost, arun_with_cost
from prismadv.llm.langchain.models.prismadv.runtime import Runtime
from prismadv.llm.langchain.models.prismadv.steps import (
    column_access,
    column_correlation,
    multi_column_assumption_inference,
    multi_column_code_translation,
    multi_column_data_flow,
    single_column_assumption_inference,
    single_column_code_translation,
    single_column_data_flow
)
from prismadv.llm_backend.entry import get_langchain_model


class PrismaLangChainDV(AbstractPrismaLangChainDV):
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
        self.model_name = model_name
        self.model = get_langchain_model(model_name, temperature=temperature)
        self.downstream_task_description = downstream_task_description
        self.logger = logger
        self.single_column_inspector = SingleColumnDataFlowInspector(model_name=model_name)
        self.multi_column_inspector = MultiColumnDataFlowInspector(model_name=model_name)
        self.use_column_correlation_detection = use_column_correlation_detection
        self._chains = self._build_chains()
        self.runtime = Runtime(self._chains, logger=self.logger)
        # Set max concurrent calls for Gemini models (15 calls per minute limit)
        self.max_concurrent_calls = 15 if 'gemini' in model_name.lower() else None

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
        all_assumptions = {}
        all_codes = {}
        column_correlation_data_flow_locations = {}
        column_desc_dict = input_variables["column_desc_dict"]
        input_variables[
            "downstream_task_description"] = self.downstream_task_description  # TODO: generate it before invoke
        columns_to_consider = run_with_cost(
            "column_access_detection",
            lambda: column_access.basic_column_access_detection(self.runtime, input_variables),
            cost_summary,
        )
        if input_variables["cfg_use_dataflow"]:
            column_data_flow_locations = run_with_cost(
                "data_flow_inspection",
                lambda: single_column_data_flow.single_column_data_flow_inspection(
                    self.single_column_inspector, column_desc_dict, columns_to_consider, input_variables
                ),
                cost_summary,
            )
        else:
            column_data_flow_locations = {column: SourceLocations() for column in columns_to_consider}
        if self.use_column_correlation_detection:
            correlated_column_groups_info = run_with_cost(
                "column_correlation_discovery",
                lambda: column_correlation.discovery(
                    self.runtime, column_desc_dict, columns_to_consider, input_variables
                ),
                cost_summary,
            )
            input_variables["single_column_data_flow"] = column_data_flow_locations
            if input_variables["cfg_use_dataflow"]:
                column_correlation_data_flow_locations = run_with_cost(
                    "multi_column_data_flow_inspection",
                    lambda: multi_column_data_flow.multi_column_data_flow_inspection(
                        self.multi_column_inspector, column_desc_dict, correlated_column_groups_info, input_variables
                    ),
                    cost_summary,
                )
            else:
                column_correlation_data_flow_locations = {frozenset(column['correlated_columns']): SourceLocations() for
                                                          column in correlated_column_groups_info}
            all_multi_column_assumptions = run_with_cost(
                "multi_column_assumption_generation",
                lambda: multi_column_assumption_inference.multi(
                    self.runtime, column_desc_dict, correlated_column_groups_info,
                    column_correlation_data_flow_locations, input_variables,
                ),
                cost_summary,
            )
            all_assumptions.update(all_multi_column_assumptions)
            all_multi_column_codes = run_with_cost(
                "multi_column_code_generation",
                lambda: multi_column_code_translation.multi(
                    self.runtime, columns_to_consider, column_desc_dict,
                    all_multi_column_assumptions, input_variables,
                ),
                cost_summary,
            )
            all_codes.update(all_multi_column_codes)
        all_single_column_assumptions = run_with_cost(
            "single_column_assumption_generation",
            lambda: single_column_assumption_inference.single(
                self.runtime, column_desc_dict, columns_to_consider,
                column_data_flow_locations, input_variables
            ),
            cost_summary,
        )
        all_assumptions.update(all_single_column_assumptions)
        all_single_column_codes = run_with_cost(
            "single_column_code_generation",
            lambda: single_column_code_translation.single(
                self.runtime, columns_to_consider, column_desc_dict,
                all_single_column_assumptions, input_variables
            ),
            cost_summary,
        )
        all_codes.update(all_single_column_codes)
        constraints_with_sources = ConstraintsWithSources.from_assumptions_and_code_dict(
            all_assumptions, all_codes
        )
        column_data_flow_locations = {
            **column_correlation_data_flow_locations,
            **column_data_flow_locations
        }
        return constraints_with_sources, column_data_flow_locations, cost_summary

    async def ainvoke(self, input_variables: dict):
        cost_summary = {}
        all_assumptions = {}
        all_codes = {}
        column_correlation_data_flow_locations = {}
        input_variables[
            "downstream_task_description"] = self.downstream_task_description  # TODO: generate it before invoke
        column_desc_dict = input_variables["column_desc_dict"]
        columns_to_consider = run_with_cost(
            "column_access_detection",
            lambda: column_access.basic_column_access_detection(self.runtime, input_variables),
            cost_summary,
        )
        if input_variables["cfg_use_dataflow"]:
            column_data_flow_locations = await arun_with_cost(
                "data_flow_inspection",
                lambda: single_column_data_flow.asingle_column_data_flow_inspection(
                    self.single_column_inspector,
                    column_desc_dict,
                    columns_to_consider,
                    input_variables,
                    max_concurrent=self.max_concurrent_calls),
                cost_summary,
            )
        else:
            column_data_flow_locations = {column: SourceLocations() for column in columns_to_consider}
        if self.use_column_correlation_detection:
            correlated_column_groups_info = run_with_cost(
                "column_correlation_discovery",
                lambda: column_correlation.discovery(
                    self.runtime, column_desc_dict, columns_to_consider, input_variables
                ),
                cost_summary,
            )
            input_variables["single_column_data_flow"] = column_data_flow_locations
            if input_variables["cfg_use_dataflow"]:
                column_correlation_data_flow_locations = await arun_with_cost(
                    "multi_column_data_flow_inspection",
                    lambda: multi_column_data_flow.amulti_column_data_flow_inspection(
                        self.multi_column_inspector,
                        column_desc_dict,
                        correlated_column_groups_info,
                        input_variables,
                        max_concurrent=self.max_concurrent_calls,
                    ),
                    cost_summary,
                )
            else:
                column_correlation_data_flow_locations = {frozenset(column['correlated_columns']): SourceLocations() for
                                                          column in
                                                          correlated_column_groups_info}
            all_multi_column_assumptions = await arun_with_cost(
                "multi_column_assumption_generation",
                lambda: multi_column_assumption_inference.amulti(
                    self.runtime,
                    column_desc_dict,
                    correlated_column_groups_info,
                    column_correlation_data_flow_locations,
                    input_variables,
                    max_concurrent=self.max_concurrent_calls,
                ),
                cost_summary,
            )
            all_assumptions.update(all_multi_column_assumptions)
            all_multi_column_codes = await arun_with_cost(
                "multi_column_code_generation",
                lambda: multi_column_code_translation.amulti(
                    self.runtime,
                    columns_to_consider,
                    column_desc_dict,
                    all_multi_column_assumptions,
                    input_variables,
                    max_concurrent=self.max_concurrent_calls,
                ),
                cost_summary,
            )
            all_codes.update(all_multi_column_codes)
        all_single_column_assumptions = await arun_with_cost(
            "single_column_assumption_generation",
            lambda: single_column_assumption_inference.asingle(
                self.runtime,
                column_desc_dict,
                columns_to_consider,
                column_data_flow_locations,
                input_variables,
                max_concurrent=self.max_concurrent_calls,
            ),
            cost_summary,
        )
        all_assumptions.update(all_single_column_assumptions)
        all_single_column_codes = await arun_with_cost(
            "single_column_code_generation",
            lambda: single_column_code_translation.asingle(
                self.runtime,
                columns_to_consider,
                column_desc_dict,
                all_assumptions,
                input_variables,
                max_concurrent=self.max_concurrent_calls,
            ),
            cost_summary,
        )
        all_codes.update(all_single_column_codes)
        constraints_with_sources = ConstraintsWithSources.from_assumptions_and_code_dict(
            all_assumptions, all_codes
        )
        column_data_flow_locations = {
            **column_correlation_data_flow_locations,
            **column_data_flow_locations
        }
        return constraints_with_sources, column_data_flow_locations, cost_summary
