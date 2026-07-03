from prismadv.dq_manager import DeequDataQualityManager
from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.loader import FileLoader
from prismadv.utils import get_project_root


def test_spark_df_to_column_desc(resources_path):
    file_path = (
            resources_path
            / "example_dataset_1"
            / "files"
            / "example_table.csv"
    )
    dq_manager = DeequDataQualityManager()
    df = FileLoader.load_csv(file_path)
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    yaml_string = DeequInspectorManager().spark_df_to_column_desc(spark_df, spark)
    assert yaml_string.startswith("UserName:\n  completeness: 1.0\n")


def test_real_dataset_profiling(dq_manager):
    import oyaml as yaml
    from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
    file_path = get_project_root() / "benchmarks" / "EIDBench-synth" / "students" / "files" / "students_data_clean.csv"
    df = FileLoader.load_csv(file_path)
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    column_desc_dict = DeequInspectorManager().spark_df_to_column_desc_dict(spark, spark_df)
    column_desc = yaml.dump(column_desc_dict, default_flow_style=False, sort_keys=False)
    print(column_desc)
