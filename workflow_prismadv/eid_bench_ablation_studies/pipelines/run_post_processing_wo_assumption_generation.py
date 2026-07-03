import asyncio

import oyaml as yaml

from prismadv.data_models import ConstraintsWithSources
from prismadv.data_models.config import PrismaDVConfig
from prismadv.data_models.constraints_v2 import ser_column_group_key
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.llm.langchain.models.prismadv_post_processing import PrismaLangChainDVPostProcessing
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root


def run_prismadv_post_processing_wo_assumption_generation(
        dataset_name,
        subtask_name,
        processed_data_label,
        constraint_file
):
    with open(f"{constraint_file}", "r") as f:
        raw_constraint_dict = yaml.load(f, Loader=yaml.FullLoader)
    try:
        prismadv_config = PrismaDVConfig.from_dict(raw_constraint_dict["prismadv_config"])
    except Exception:
        print('Not a valid config')
        return

    post_processed_file = constraint_file.with_name(
        constraint_file.name.replace("prismadv_wo_assumption_before_post", "post_processed_prismadv_wo_assumption"))
    if post_processed_file.exists():
        if prismadv_config.io.overwrite is False:
            print(f"Post-processed constraints file {post_processed_file} already exists. Skipping...")
            return
        else:
            print(f"Post-processed constraints file {post_processed_file} already exists. Overwriting...")
            post_processed_file.unlink()

    existing_constraints = ConstraintsWithSources.from_dict(
        {"constraints": raw_constraint_dict["constraints"]})
    existing_cost_summary = raw_constraint_dict["cost_summary"]
    column_data_flow_locations = raw_constraint_dict["column_data_flow_locations"]

    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    dq_manager = DeequDataQualityManager()
    train_data = FileLoader.load_csv(
        project_manager.get_observed_data_path(subtask_name, processed_data_label)
    )
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    spark_validation_data, spark_validation = dq_manager.spark_df_from_pandas_df(train_data)
    column_desc_dict = DeequInspectorManager().spark_df_to_column_desc_dict(spark_train, spark_train_data)

    input_variables = {
        "existing_constraints": existing_constraints,
        "column_desc_dict": column_desc_dict,
        "spark_validation": spark_validation,
        "spark_validation_data": spark_validation_data,
    }
    prismadv_post_processing = PrismaLangChainDVPostProcessing.from_config(prismadv_config)
    if prismadv_config.model.use_async:
        constraints_with_sources, fixing_results, consolidation_results, post_processing_cost_summary = asyncio.run(
            prismadv_post_processing.ainvoke(input_variables)
        )
    else:
        constraints_with_sources, fixing_results, consolidation_results, post_processing_cost_summary = prismadv_post_processing.invoke(
            input_variables)
    cost_summary = {**existing_cost_summary, **post_processing_cost_summary}
    fixing_results_ser = {
        ser_column_group_key(k): v for k, v in fixing_results.items()
    }
    consolidation_results_ser = {
        ser_column_group_key(k): v for k, v in consolidation_results.items()
    }
    res_dict = {
        "prismadv_config": prismadv_config.to_dict(),
        "constraints": constraints_with_sources.to_dict()["constraints"],
        "column_data_flow_locations": column_data_flow_locations,
        "fixing_results": fixing_results_ser,
        "consolidation_results": consolidation_results_ser,
        "cost_summary": cost_summary,
    }
    with open(post_processed_file, "w") as f:
        yaml.dump(res_dict, f, default_flow_style=False, sort_keys=False)

    spark_train.sparkContext._gateway.close()
    spark_validation.sparkContext._gateway.close()
    spark_train.stop()
    spark_validation.stop()
