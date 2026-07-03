"""Functions for preparing training datasets for optimization."""

from typing import Dict, List, Union

import dspy
import oyaml as yaml

from prismadv.project_manager.manager.base import ProjectManager


def prepare_single_column_training_dataset(
        project_manager: ProjectManager,
        dataset_subtasks: Dict[str, List[str]],
        script_name_list: Union[List[str], Dict[str, List[str]]],
        processed_data_label: Union[str, Dict[str, str]] = "0",
        new_processed_data_label_list: Union[List[str], Dict[str, List[str]], None] = None,
) -> List[dspy.Example]:
    """
    Prepare training dataset for single-column constraint generation.
    
    Each item in the dataset contains all information needed to call
    generate_constraints_for_column().
    
    Args:
        project_manager: ProjectManager instance (for backward compatibility, not actually used)
        dataset_subtasks: Dict mapping dataset_name to list of subtask names
            e.g., {"students": ["general_task"]}
        script_name_list: List of script names (applied to all datasets) or 
            dict mapping dataset_name to list of script names
        processed_data_label: Label for processed data (default: "0" for training data).
            Can be a string (applied to all) or dict mapping dataset_name to label.
        new_processed_data_label_list: List of new processed data labels to check safety for.
            Can be a list (applied to all datasets) or dict mapping dataset_name to list of labels.
            If None, no safety information is recorded.
        
    Returns:
        List of dicts, each containing:
            - "dataset_name": str
            - "subtask_name": str
            - "script_name": str
            - "column_name": str
            - "column_desc_dict": Dict[str, Dict] (full dict)
            - "source_code": str
            - "downstream_task_description": str
            - "new_data_safety": Dict[str, bool] (mapping label -> is_safe), if new_processed_data_label_list is provided
    """
    from workflow_sifta.optimization_fns.column_discovery import discover_columns_and_groups

    training_dataset = []

    # Iterate over all dataset/subtask/script combinations
    for dataset_name, subtask_list in dataset_subtasks.items():
        # Create project manager for this dataset
        pm = ProjectManager(dataset_name=dataset_name)

        # Get script names for this dataset
        if isinstance(script_name_list, dict):
            scripts_to_use = script_name_list.get(dataset_name, [])
        else:
            scripts_to_use = script_name_list

        # Get processed_data_label for this dataset
        if isinstance(processed_data_label, dict):
            label = processed_data_label.get(dataset_name, "0")
        else:
            label = processed_data_label

        for subtask_name in subtask_list:
            for script_name in scripts_to_use:
                try:
                    # Discover columns and groups for this script
                    discovery_result = discover_columns_and_groups(
                        project_manager=pm,
                        subtask_name=subtask_name,
                        script_name=script_name,
                        processed_data_label=label,
                    )

                    columns_to_consider = discovery_result["columns_to_consider"]
                    column_desc_dict = discovery_result["column_desc_dict"]
                    source_code = discovery_result["source_code"]
                    downstream_task_description = discovery_result["downstream_task_description"]

                    # Get new_processed_data_label_list for this dataset
                    if new_processed_data_label_list is not None:
                        if isinstance(new_processed_data_label_list, dict):
                            labels_to_check = new_processed_data_label_list.get(dataset_name, [])
                        else:
                            labels_to_check = new_processed_data_label_list

                        # Read safety information for each label
                        new_data_safety = {}
                        for new_label in labels_to_check:
                            try:
                                exec_path = pm.get_execution_output_validation_path(
                                    subtask_name, new_label, script_name
                                ) / "basic_metrics_evaluation.json"

                                with open(exec_path, "r") as f:
                                    exec_results = yaml.load(f, Loader=yaml.FullLoader)
                                # Use corrupted_data_is_safe for new test data (not clean)
                                is_safe = exec_results.get('corrupted_data_is_safe', True)
                                new_data_safety[new_label] = is_safe
                            except (FileNotFoundError, KeyError) as e:
                                # Default to True if execution results not found
                                new_data_safety[new_label] = True
                    else:
                        new_data_safety = {}

                    # Create training example for each column
                    for column_name in columns_to_consider:
                        target_column_desc = yaml.dump(
                            {column_name: column_desc_dict[column_name]},
                            default_flow_style=False,
                            sort_keys=False
                        )
                        training_example_dict = {
                            "dataset_name": dataset_name,
                            "subtask_name": subtask_name,
                            "script_name": script_name,
                            "target_column": column_name,
                            "target_column_desc": target_column_desc,
                            "code_script": source_code,
                            "downstream_task_description": downstream_task_description,
                        }
                        if new_data_safety:
                            training_example_dict["new_data_safety"] = new_data_safety
                        training_example = dspy.Example(
                            **training_example_dict
                        ).with_inputs(
                            "dataset_name",
                            "subtask_name",
                            "script_name",
                            "target_column",
                            "target_column_desc",
                            "code_script",
                            "downstream_task_description",
                        )
                        training_dataset.append(training_example)

                except Exception as e:
                    # Skip scripts that don't exist or have errors
                    print(f"Warning: Skipping {dataset_name}/{subtask_name}/{script_name}: {e}")
                    continue

    return training_dataset
