from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root


def run_deequ_dv(dataset_name, subtask_name, processed_data_label):
    dq_manager = DeequDataQualityManager()
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)

    constraint_output_path = project_manager.get_task_agnostic_constraint_path(subtask_name,
                                                                               processed_data_label) / "deequ_constraints.yaml"
    constraint_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_data = FileLoader.load_csv(
        project_manager.get_new_data_path(subtask_name, processed_data_label, clean=True) / "new_data.csv"
    )
    validation_data = FileLoader.load_csv(
        project_manager.get_new_data_path(subtask_name, processed_data_label, clean=True) / "new_data.csv"
    )
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    spark_validation_data, spark_validation = dq_manager.spark_df_from_pandas_df(validation_data)

    constraints = dq_manager.inference_constraints_for_spark_df(spark_train, spark_train_data, spark_validation,
                                                                spark_validation_data)
    constraints.save_to_yaml(constraint_output_path)

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_validation.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()
    spark_validation.stop()
