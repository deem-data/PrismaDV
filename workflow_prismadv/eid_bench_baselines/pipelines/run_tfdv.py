import tensorflow_data_validation as tfdv

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root


def run_tfdv(dataset_name, subtask_name, processed_data_label):
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)

    constraint_output_path = project_manager.get_task_agnostic_constraint_path(subtask_name,
                                                                               processed_data_label) / "schema.pbtxt"
    constraint_output_path.parent.mkdir(parents=True, exist_ok=True)
    # observed_data = FileLoader.load_csv(
    #     project_manager.get_new_data_path(subtask_name, processed_data_label, clean=True) / "new_data.csv"
    # )
    train_stats = tfdv.generate_statistics_from_csv(
        data_location=str(project_manager.get_new_data_path(subtask_name, processed_data_label,
                                                            clean=True) / "new_data.csv")
    )
    schema = tfdv.infer_schema(train_stats)
    tfdv.write_schema_text(schema, constraint_output_path)
