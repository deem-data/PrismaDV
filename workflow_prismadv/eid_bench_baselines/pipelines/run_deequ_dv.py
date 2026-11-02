import time

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root


def run_deequ_dv(dataset_name, subtask_name, processed_data_label):
    dq_manager = DeequDataQualityManager()
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)

    constraint_output_path = project_manager.get_task_agnostic_constraint_path(subtask_name) / "deequ_constraints.yaml"
    constraint_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Constraints are inferred from the observed half, never from the batch they
    # will be scored against. `new_data` is the evaluation set: profiling its
    # clean version would hand this row the pre-corruption distribution of its
    # own test data, which no other baseline gets.
    # Wall clock of the whole inference: data load, Spark start, profiling and suggestion.
    started = time.perf_counter()
    observed_data_path = project_manager.get_observed_data_path(subtask_name, processed_data_label)
    train_data = FileLoader.load_csv(observed_data_path)
    validation_data = FileLoader.load_csv(observed_data_path)
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    spark_validation_data, spark_validation = dq_manager.spark_df_from_pandas_df(validation_data)

    constraints = dq_manager.inference_constraints_for_spark_df(spark_train, spark_train_data, spark_validation,
                                                                spark_validation_data)
    constraints.save_to_yaml(constraint_output_path,
                             extra={"wall_clock_seconds": round(time.perf_counter() - started, 3)})

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_validation.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()
    spark_validation.stop()
