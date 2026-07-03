import dspy
import oyaml as yaml

import prismadv.llm.dspy.models.prismadv.steps.multi_column_assumption_inference as multi_column_assumption_inference
import prismadv.llm.dspy.models.prismadv.steps.multi_column_code_translation as multi_column_code_translation
import prismadv.llm.dspy.models.prismadv.steps.multi_column_data_flow as multi_column_data_flow
import prismadv.llm.dspy.models.prismadv.steps.single_column_assumption_inference as single_column_assumption_inference
import prismadv.llm.dspy.models.prismadv.steps.single_column_code_translation as single_column_code_translation
import prismadv.llm.dspy.models.prismadv.steps.single_column_data_flow as single_column_data_flow
from prismadv.data_models import ConstraintsWithSources
from prismadv.llm.dspy.models.prismadv.steps.column_access_detection import ColumnAccessDetectionSig
from prismadv.llm.dspy.models.prismadv.steps.column_correlation import ColumnCorrelationDiscoverySig
from prismadv.llm.dspy.models.prismadv.steps.multi_column_assumption_inference import MultiColumnAssumptionGenerationSig
from prismadv.llm.dspy.models.prismadv.steps.multi_column_code_translation import MultiColumnConstraintGenerationSig
from prismadv.llm.dspy.models.prismadv.steps.multi_column_data_flow import MultiColumnDataFlowInspectorSig
from prismadv.llm.dspy.models.prismadv.steps.single_column_assumption_inference import \
    SingleColumnAssumptionGenerationSig
from prismadv.llm.dspy.models.prismadv.steps.single_column_code_translation import IRGenerationSig
from prismadv.llm.dspy.models.prismadv.steps.single_column_data_flow import SingleColumnCodeDataFlowInspectorSig


class PrismaDspyDV(dspy.Module):
    def __init__(
            self,
            model_name: str,
            downstream_task_description: str,
            use_column_correlation_detection: bool = False,
            temperature: float = 0.6,
    ):
        super().__init__()
        self.lm = dspy.LM(model_name, temperature=temperature, model_type="responses")
        dspy.configure(lm=self.lm)
        self.downstream_task_description = downstream_task_description
        self.use_column_correlation_detection = use_column_correlation_detection
        self._build_signatures()

    def get_cost_and_clean_history(self):
        cost = sum([x['cost'] for x in self.lm.history if x['cost'] is not None])
        self.lm.history = []
        return cost

    def _build_signatures(self):
        self.column_access_detection_sig = ColumnAccessDetectionSig
        self.column_correlation_discovery_sig = ColumnCorrelationDiscoverySig
        self.single_column_data_flow_inspector_sig = SingleColumnCodeDataFlowInspectorSig
        self.multi_column_data_flow_inspector_sig = MultiColumnDataFlowInspectorSig
        self.multi_column_assumption_inference_sig = MultiColumnAssumptionGenerationSig
        self.multi_column_code_translation_sig = MultiColumnConstraintGenerationSig
        self.single_column_assumption_inference_sig = SingleColumnAssumptionGenerationSig
        self.single_column_code_translation_sig = IRGenerationSig

    def forward(self, input_variables: dict):
        """different from langchain version. To optimize the prompts this forward function will only works for single column or single column group"""
        assert (input_variables.get("columns_to_consider") is not None and len(
            input_variables.get("columns_to_consider")) == 1), \
            "The forward function only supports single column or single column group processing. Please use aforward for multi-column processing."


    async def aforward(self, input_variables: dict):
        cost_summary = {}
        all_assumptions = {}
        all_codes = {}
        column_desc_dict = input_variables["column_desc_dict"]
        input_variables[
            "downstream_task_description"] = self.downstream_task_description
        column_access_detection = dspy.Predict(
            signature=self.column_access_detection_sig,
        )
        column_access_result = column_access_detection(**input_variables)
        columns_to_consider = column_access_result.columns
        cost_summary["column_access_detection"] = self.get_cost_and_clean_history()

        single_column_data_flow_inspector = dspy.Predict(
            signature=self.single_column_data_flow_inspector_sig,
        )
        column_data_flow_locations = await single_column_data_flow.asingle(
            single_column_inspector=single_column_data_flow_inspector,
            column_desc_dict=column_desc_dict, columns_to_consider=columns_to_consider,
            vars=input_variables)
        input_variables["single_column_data_flow"] = column_data_flow_locations
        cost_summary["single_column_data_flow"] = self.get_cost_and_clean_history()

        if self.use_column_correlation_detection:
            input_variables[
                "columns_to_consider"] = columns_to_consider
            considered_columns_desc_dict = {
                col: column_desc_dict[col] for col in column_desc_dict if col in columns_to_consider
            }
            considered_columns_desc = yaml.dump(considered_columns_desc_dict, default_flow_style=False, sort_keys=False)
            input_variables[
                "considered_columns_desc"] = considered_columns_desc
            column_correlation_discovery = dspy.Predict(
                signature=self.column_correlation_discovery_sig,
            )
            column_correlation_result = column_correlation_discovery(**input_variables)
            cost_summary["column_correlation_discovery"] = self.get_cost_and_clean_history()
            correlated_groups = column_correlation_result.correlated_groups
            correlated_groups = [group for group in correlated_groups if len(group["correlated_columns"]) > 1]
            multi_column_data_flow_inspector = dspy.Predict(
                signature=self.multi_column_data_flow_inspector_sig,
            )
            column_correlation_data_flow_locations = await multi_column_data_flow.amulti(
                multi_column_inspector=multi_column_data_flow_inspector,
                column_desc_dict=column_desc_dict,
                corr_groups=correlated_groups,
                vars=input_variables
            )
            cost_summary["multi_column_data_flow"] = self.get_cost_and_clean_history()
            multi_column_assumption_inference_runtime = dspy.Predict(
                signature=self.multi_column_assumption_inference_sig,
            )
            all_multi_column_assumptions = await multi_column_assumption_inference.amulti(
                multi_column_assumption_inference_runtime,
                column_desc_dict,
                correlated_groups,
                column_correlation_data_flow_locations,
                input_variables,
            )
            cost_summary["multi_column_assumption_inference"] = self.get_cost_and_clean_history()
            all_assumptions.update(all_multi_column_assumptions)
            multi_column_code_translation_runtime = dspy.Predict(
                signature=self.multi_column_code_translation_sig
            )
            all_multi_column_codes = await multi_column_code_translation.amulti(
                multi_column_code_translation_runtime,
                columns_to_consider,
                column_desc_dict,
                all_multi_column_assumptions,
                input_variables,
            )
            cost_summary["multi_column_code_translation"] = self.get_cost_and_clean_history()
            all_codes.update(all_multi_column_codes)

        single_column_assumption_inference_runtime = dspy.Predict(
            signature=self.single_column_assumption_inference_sig
        )
        all_single_column_assumptions = await single_column_assumption_inference.asingle(
            single_column_assumption_inference_runtime,
            column_desc_dict, columns_to_consider,
            column_data_flow_locations, input_variables
        )
        cost_summary["single_column_assumption_inference"] = self.get_cost_and_clean_history()
        all_assumptions.update(all_single_column_assumptions)
        single_column_code_translation_runtime = dspy.Predict(
            signature=self.single_column_code_translation_sig
        )
        all_single_column_codes = await single_column_code_translation.asingle(
            single_column_code_translation_runtime,
            columns_to_consider,
            column_desc_dict,
            all_single_column_assumptions,
            input_variables
        )
        cost_summary["single_column_code_translation"] = self.get_cost_and_clean_history()
        all_codes.update(all_single_column_codes)
        constraints_with_sources = ConstraintsWithSources.from_assumptions_and_code_dict(
            all_assumptions, all_codes
        )
        if self.use_column_correlation_detection:
            column_data_flow_locations = {
                **column_correlation_data_flow_locations,
                **column_data_flow_locations
            }
        return constraints_with_sources, column_data_flow_locations, cost_summary
