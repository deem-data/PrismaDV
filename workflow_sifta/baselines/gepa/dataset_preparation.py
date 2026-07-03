"""Dataset preparation for script-level DSPy baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import dspy
import oyaml as yaml

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager


def _load_script_level_inputs(
        project_manager: ProjectManager,
        subtask_name: str,
        script_name: str,
        processed_data_label: str,
) -> dict:
    train_data = FileLoader.load_csv(
        project_manager.get_observed_data_path(subtask_name, processed_data_label)
    )
    dq_manager = DeequDataQualityManager()
    spark_df, spark_session = dq_manager.spark_df_from_pandas_df(train_data)

    try:
        column_desc_dict = DeequInspectorManager().spark_df_to_column_desc_dict(
            spark_session, spark_df
        )
        task_name = project_manager.get_task_name_from_subtask(subtask_name)
        script_path = project_manager.get_script_path(task_name, script_name)
        source_code, _ = FileLoader.load_py_file(script_path).extract_assertions()
        downstream_task_description = project_manager.get_subtask_description(
            subtask_name=subtask_name
        )
        return {
            "column_desc_dict": column_desc_dict,
            "source_code": source_code,
            "downstream_task_description": downstream_task_description,
        }
    finally:
        spark_session.sparkContext._gateway.close()
        spark_session.stop()


def prepare_script_level_dataset(
        dataset_subtasks: Dict[str, List[str]],
        script_name_list: Union[List[str], Dict[str, List[str]]],
        processed_data_label: Union[str, Dict[str, str]] = "0",
        new_processed_data_label_list: Union[List[str], Dict[str, List[str]], None] = None,
        project_root: Path = None,
) -> List[dspy.Example]:
    """
    Prepare script-level dataset examples.

    Each item includes dataset/subtask/script metadata plus the script code and
    full column descriptions.
    """

    dataset = []

    for dataset_name, subtask_list in dataset_subtasks.items():
        pm = ProjectManager(project_root=project_root, dataset_name=dataset_name)

        if isinstance(script_name_list, dict):
            scripts_to_use = script_name_list.get(dataset_name, [])
        else:
            scripts_to_use = script_name_list

        if isinstance(processed_data_label, dict):
            label = processed_data_label.get(dataset_name, "0")
        else:
            label = processed_data_label

        for subtask_name in subtask_list:
            for script_name in scripts_to_use:
                try:
                    inputs = _load_script_level_inputs(
                        project_manager=pm,
                        subtask_name=subtask_name,
                        script_name=script_name,
                        processed_data_label=label,
                    )
                    column_desc_yaml = yaml.dump(
                        inputs["column_desc_dict"],
                        default_flow_style=False,
                        sort_keys=False,
                    )

                    new_data_safety = {}
                    if new_processed_data_label_list is not None:
                        if isinstance(new_processed_data_label_list, dict):
                            labels_to_check = new_processed_data_label_list.get(dataset_name, [])
                        else:
                            labels_to_check = new_processed_data_label_list

                        for new_label in labels_to_check:
                            try:
                                exec_path = pm.get_execution_output_validation_path(
                                    subtask_name, new_label, script_name
                                ) / "basic_metrics_evaluation.json"

                                with open(exec_path, "r") as f:
                                    exec_results = yaml.load(f, Loader=yaml.FullLoader)
                                is_safe = exec_results.get("corrupted_data_is_safe", True)
                                new_data_safety[new_label] = is_safe
                            except (FileNotFoundError, KeyError):
                                new_data_safety[new_label] = True

                    example_dict = {
                        "dataset_name": dataset_name,
                        "subtask_name": subtask_name,
                        "script_name": script_name,
                        "columns_desc": column_desc_yaml,
                        "code_script": inputs["source_code"],
                        "downstream_task_description": inputs["downstream_task_description"],
                    }
                    if new_data_safety:
                        example_dict["new_data_safety"] = new_data_safety

                    example = dspy.Example(**example_dict).with_inputs(
                        "columns_desc",
                        "code_script",
                        "downstream_task_description",
                    )
                    dataset.append(example)
                except Exception as exc:
                    print(
                        f"Warning: Skipping {dataset_name}/{subtask_name}/{script_name}: {exc}"
                    )
                    continue

    return dataset
