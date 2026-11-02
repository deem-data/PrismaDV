import json
import time

import tensorflow_data_validation as tfdv

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root


def run_tfdv(dataset_name, subtask_name, processed_data_label):
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)

    constraint_output_path = project_manager.get_task_agnostic_constraint_path(subtask_name) / "schema.pbtxt"
    constraint_output_path.parent.mkdir(parents=True, exist_ok=True)
    # The schema is inferred from the observed half, never from the batch it will
    # be scored against. `new_data` is the evaluation set: profiling its clean
    # version would hand this row the pre-corruption distribution of its own test
    # data, which no other baseline gets.
    started = time.perf_counter()
    train_stats = tfdv.generate_statistics_from_csv(
        data_location=str(project_manager.get_observed_data_path(subtask_name, processed_data_label))
    )
    schema = tfdv.infer_schema(train_stats)
    tfdv.write_schema_text(schema, constraint_output_path)
    # A .pbtxt has no metadata slot, so the inference wall clock goes in a sidecar.
    constraint_output_path.with_name("schema.timing.json").write_text(
        json.dumps({"wall_clock_seconds": round(time.perf_counter() - started, 3)}))
