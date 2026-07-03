import asyncio

import oyaml as yaml

from prismadv.data_models.config import PrismaDVConfig
from prismadv.data_models.constraints_v2 import ser_column_group_key
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.llm.langchain.models.prismadv import PrismaLangChainDV
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root


def run_prismadv(dataset_name, subtask_name, processed_data_label, prismadv_config):
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)

    dq_manager = DeequDataQualityManager()
    train_data = FileLoader.load_csv(
        project_manager.get_observed_data_path(subtask_name, processed_data_label)
    )
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    spark_validation_data, spark_validation = dq_manager.spark_df_from_pandas_df(train_data)

    script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name=subtask_name)
    for script_path in script_path_list:
        constraints_path = project_manager.get_constraints_path(
            subtask_name, processed_data_label, script_path.stem
        )
        constraints_path.mkdir(parents=True, exist_ok=True)
        constraint_output_path = constraints_path / prismadv_config.make_output_filename()
        skip = False
        for path in constraints_path.glob("prismadv--*.yaml"):
            with open(path, "r") as f:
                existing_raw_data = yaml.load(f, Loader=yaml.FullLoader)
            existing_config = PrismaDVConfig.from_dict(existing_raw_data["prismadv_config"])
            if existing_config == prismadv_config:
                if prismadv_config.io.overwrite is False:
                    print(f"Constraints file {constraint_output_path} already exists. Skipping...")
                    skip = True
                else:
                    print(f"Constraints file {constraint_output_path} already exists. Overwriting...")
                    path.unlink()
        if skip:
            continue
        source_code, assertions = FileLoader.load_py_file(script_path).extract_assertions()
        assert "# ASSERTION START" not in str(source_code)
        column_desc_dict = DeequInspectorManager().spark_df_to_column_desc_dict(spark_train, spark_train_data)
        column_desc = yaml.dump(column_desc_dict, default_flow_style=False, sort_keys=False)

        input_variables = {
            "code_script": source_code,
            "column_desc_dict": column_desc_dict,
            "columns_desc": column_desc,
            "cfg_use_dataflow": prismadv_config.model.use_dataflow,
            "spark": spark_validation,
            "data_sample": spark_validation_data,
        }
        print(f"Running PrismaDV for dataset: {dataset_name}, ")
        print(f"subtask: {subtask_name}, processed_data_label: {processed_data_label}, "
              f"script: {script_path.stem}, model: {prismadv_config.llm.model_name}")
        prismadv = PrismaLangChainDV.from_config(prismadv_config)
        if prismadv_config.model.use_async:
            constraints_with_sources, column_data_flow_locations, cost_summary = asyncio.run(
                prismadv.ainvoke(input_variables))
        else:
            constraints_with_sources, column_data_flow_locations, cost_summary = prismadv.invoke(input_variables)
        res_dict = {
            "prismadv_config": prismadv_config.to_dict(),
            "constraints": constraints_with_sources.to_dict()["constraints"],
            "column_data_flow_locations": {
                ser_column_group_key(col): loc.to_dict() for col, loc in column_data_flow_locations.items()
            },
            "cost_summary": cost_summary
        }
        with open(constraint_output_path, "w") as f:
            yaml.dump(res_dict, f, default_flow_style=False, sort_keys=False)

    spark_train.sparkContext._gateway.close()
    spark_validation.sparkContext._gateway.close()
    spark_train.stop()
    spark_validation.stop()
