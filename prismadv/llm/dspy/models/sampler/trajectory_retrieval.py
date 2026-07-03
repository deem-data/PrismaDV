from typing import Dict, List, Union

from prismadv.data_models.trajectory import DVTrajectory, DVTrajectoryColumnGroupSuite
from prismadv.project_manager.manager.base import ProjectManager

from prismadv.llm.dspy.models.sampler.types import TrajectoryKey


def retrieve_trajectories(
    dataset_subtasks: Dict[str, List[str]],
    processed_data_label_list: Union[List[str], Dict[str, List[str]]],
    llm_name: str,
    dspy_prefix: str,
    downstream_task_type: Union[str, Dict[str, str]],
    script_name_list: Union[None, List[str], Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, List[DVTrajectory]]]:
    """
    Retrieve trajectories from multiple datasets using ProjectManager.
    
    Args:
        dataset_subtasks: Dict mapping dataset_name to list of subtask names
        processed_data_label_list: List of data labels (global) or dict (per-dataset)
        llm_name: Name of the LLM
        dspy_prefix: Prefix for dspy trajectories
        downstream_task_type: Task type (string for all or dict per-dataset)
        script_name_list: Optional list of script names to filter, or dict mapping 
            dataset_name to list of script names. If None, all available scripts are used.
        
    Returns:
        Nested dict with structure: {subtask_name: {script_name: [trajectories]}}
    """
    # Extract dataset names from the keys of dataset_subtasks
    dataset_names = list(dataset_subtasks.keys())

    # Create project managers for each dataset
    project_managers = {}
    for dataset_name in dataset_names:
        # Get task type for this dataset
        if isinstance(downstream_task_type, dict):
            task_type = downstream_task_type.get(dataset_name, "general")
        else:
            task_type = downstream_task_type

        project_managers[dataset_name] = ProjectManager(
            dataset_name=dataset_name,
            downstream_task_type=task_type,
        )

    # Collect all trajectories from all datasets
    all_trajectories = {}

    for dataset_name in dataset_names:
        pm = project_managers[dataset_name]
        subtask_name_list = dataset_subtasks[dataset_name]

        # Get labels for this dataset
        if isinstance(processed_data_label_list, dict):
            labels = processed_data_label_list[dataset_name]
        else:
            labels = processed_data_label_list

        for subtask_name in subtask_name_list:
            # Initialize subtask dict if not exists
            if subtask_name not in all_trajectories:
                all_trajectories[subtask_name] = {}

            # Get available scripts for this subtask
            available_script_path = pm.get_available_script_path_list_for_subtask(
                subtask_name
            )
            available_script_name_list = [path.stem for path in available_script_path]

            # Filter scripts if script_name_list is provided
            if script_name_list is not None:
                # Get script names for this dataset
                if isinstance(script_name_list, dict):
                    script_names_to_use = script_name_list.get(dataset_name, available_script_name_list)
                else:
                    script_names_to_use = script_name_list
                
                # Filter to only include scripts in the list
                available_script_name_list = [
                    script_name for script_name in available_script_name_list
                    if script_name in script_names_to_use
                ]

            # Collect trajectories for each script
            for script_name in available_script_name_list:
                dspy_trajectories = pm.get_dspy_trajectories(
                    subtask_name,
                    labels,
                    script_name_list=[script_name],
                    llm_name=llm_name,
                    dspy_prefix=dspy_prefix,
                )

                # Combine trajectories from different datasets
                if script_name not in all_trajectories[subtask_name]:
                    all_trajectories[subtask_name][script_name] = []
                all_trajectories[subtask_name][script_name].extend(dspy_trajectories)

    return all_trajectories


def aggregate_trajectories(
    all_trajectories: Dict[str, Dict[str, List[DVTrajectory]]]
) -> Dict[TrajectoryKey, List[DVTrajectoryColumnGroupSuite]]:
    """
    Aggregate trajectories by (dataset, subtask, script, llm) and column groups.
    
    Args:
        all_trajectories: Input trajectories grouped by subtask and script
        
    Returns:
        Aggregated trajectories with structure:
        {(dataset_name, subtask_name, script_name, llm_name): [DVTrajectoryColumnGroupSuite]}
    """
    # First, group by key and column_group
    temp_aggregated = {}

    for subtask_name, script_dict in all_trajectories.items():
        for script_name, trajectory_list in script_dict.items():
            for trajectory in trajectory_list:
                # Create unique key for this trajectory group
                key = (
                    trajectory.dataset_name,
                    subtask_name,
                    script_name,
                    trajectory.llm_name,
                )
                column_group = trajectory.column_group

                # Initialize nested dicts as needed
                if key not in temp_aggregated:
                    temp_aggregated[key] = {}
                if column_group not in temp_aggregated[key]:
                    temp_aggregated[key][column_group] = []

                temp_aggregated[key][column_group].append(trajectory)

    # Convert to DVTrajectoryColumnGroupSuite objects
    aggregated = {}
    for key, column_group_dict in temp_aggregated.items():
        aggregated[key] = [
            DVTrajectoryColumnGroupSuite(
                column_group=column_group,
                trajectories=trajectories
            )
            for column_group, trajectories in column_group_dict.items()
        ]

    return aggregated
