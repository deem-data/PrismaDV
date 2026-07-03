"""Generate constraints with PrismaDV on ICDBench cases"""
import oyaml as yaml
import pandas as pd

from prismadv.code_inspector.llm_code_inspector.single.model import \
    ColumnDataFlowInspector as SingleColumnDataFlowInspector
from prismadv.data_models.code_container import CodeContainer
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.llm.langchain.models.prismadv import PrismaLangChainDV
from workflow_prismadv.icd_bench_experiments import find_constraints_output_path, ALL_EVALUATION_CASES

models = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-5-mini", "gpt-5", "gemini-2.5-flash", "gemini-2.5-pro"]

dq_manager = DeequDataQualityManager()

for model_name in models:
    for evaluation_case in ALL_EVALUATION_CASES:

        constraints_output_path = find_constraints_output_path(evaluation_case, model_name)

        if constraints_output_path.exists():
            print(f"Constraints already exist at {constraints_output_path}. Skipping generation.")
            continue

        # print(evaluation_case)
        # for col, data in evaluation_case.sample_data().items():
        #     print(col, len(data))

        data_sample = pd.DataFrame(evaluation_case.sample_data())
        data_sample_df, spark = dq_manager.spark_df_from_pandas_df(data_sample)

        column_desc_dict = DeequInspectorManager().spark_df_to_column_desc_dict(spark, data_sample_df)
        column_desc = yaml.dump(column_desc_dict, default_flow_style=False, sort_keys=False)

        single_column_inspector = SingleColumnDataFlowInspector(model_name=model_name)

        code = CodeContainer(evaluation_case.downstream_code())
        task_description = ""

        input_variables = {
            "code_script": code,
            "downstream_task_description": task_description,
            "column_desc_dict": column_desc_dict,
            "columns_desc": column_desc,
            "code_snippet": code,
            "cfg_use_dataflow": True,
            "columns_to_consider": [evaluation_case.target_column()],
            "spark": spark,
            "data_sample": data_sample_df,
        }

        try:
            print(model_name, evaluation_case.__class__.__name__, evaluation_case.target_column())
            prismadv = PrismaLangChainDV(model_name=model_name, downstream_task_description=task_description)
            constraints_with_sources, column_data_flow_locations, cost_summary = prismadv.invoke(input_variables)

            res_dict = {
                "constraints": constraints_with_sources.to_dict()["constraints"],
                "column_data_flow_locations": {
                    col: loc.to_dict() for col, loc in column_data_flow_locations.items()
                }
            }
        except Exception as e:
            print(
                f"Error generating constraints for {evaluation_case.__class__.__module__}.{evaluation_case.__class__.__name__}: {e}")
            res_dict = {
                "constraints": [],
                "column_data_flow_locations": {}
            }
            raise

        with open(constraints_output_path, "w") as f:
            yaml.dump(res_dict, f, default_flow_style=False, sort_keys=False)
            print(f"... constraints saved to {constraints_output_path} for {model_name}")
