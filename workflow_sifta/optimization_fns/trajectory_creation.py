"""Trajectory creation functions."""

from pathlib import Path
from typing import Dict, List, Tuple

import oyaml as yaml

from prismadv.data_models import ValidationResults
from prismadv.data_models.constraints_v2 import ConstraintsWithSources
from prismadv.data_models.trajectory import DVTrajectoryColumnGroupSuite
from prismadv.loader import FileLoader
from prismadv.llm.dspy.models.sampler.trajectory_retrieval import aggregate_trajectories
from prismadv.llm.dspy.models.sampler.types import TrajectoryKey
from prismadv.project_manager.manager.base import ProjectManager


def create_trajectories_from_constraints(
    project_manager: ProjectManager,
    dataset_name: str,
    subtask_name: str,
    script_name: str,
    processed_data_label: str,
    llm_name: str,
    constraints_with_sources: ConstraintsWithSources,
    validation_results: ValidationResults,
    clean: bool = False,
) -> List:
    """
    Create trajectories from constraints and validation results.
    
    Args:
        project_manager: ProjectManager instance
        dataset_name: Name of the dataset
        subtask_name: Name of the subtask
        script_name: Name of the script
        processed_data_label: Label for processed data
        llm_name: Name of the LLM used
        constraints_with_sources: ConstraintsWithSources object
        validation_results: ValidationResults object
        clean: Whether using clean test data (default: False)
        
    Returns:
        List of DVTrajectory objects
    """
    # Get script path and load script
    task_name = project_manager._metadata_manager.get_task_name_from_subtask(subtask_name)
    script_path = project_manager._script_manager.get_script_path(task_name, script_name)
    script = FileLoader.load_py_file(script_path)
    
    # Get test data path
    test_data_path = project_manager.get_new_test_data_path(
        subtask_name, processed_data_label, clean=clean
    )
    
    # Read is_safe from execution results
    exec_path = project_manager.get_execution_output_validation_path(
        subtask_name, processed_data_label, script_name
    ) / "basic_metrics_evaluation.json"
    
    try:
        with open(exec_path, "r") as f:
            exec_results = yaml.load(f, Loader=yaml.FullLoader)
        is_safe = exec_results['clean_data_is_safe'] if clean else exec_results['corrupted_data_is_safe']
    except FileNotFoundError:
        # Default to True if execution results not found
        is_safe = True
    
    # Create trajectories using ProjectManager's format_trajectory_for_dspy method
    trajectories = project_manager.format_trajectory_for_dspy(
        dataset_name=dataset_name,
        llm_name=llm_name,
        script_path=script_path,
        script=script,
        data_path=test_data_path,
        constraints=constraints_with_sources,
        validation_results=validation_results,
        processed_data_label=processed_data_label,
        is_safe=is_safe
    )
    
    return trajectories


def aggregate_trajectories_for_sampling(
    trajectories: List,
    dataset_name: str,
    subtask_name: str,
    script_name: str,
    llm_name: str,
) -> Dict[TrajectoryKey, List[DVTrajectoryColumnGroupSuite]]:
    """
    Aggregate trajectories into the format needed for sampling.
    
    Args:
        trajectories: List of DVTrajectory objects
        dataset_name: Name of the dataset
        subtask_name: Name of the subtask
        script_name: Name of the script
        llm_name: Name of the LLM
        
    Returns:
        Dict mapping TrajectoryKey to list of DVTrajectoryColumnGroupSuite objects
    """
    # Group trajectories by subtask and script for aggregation
    # Format: {subtask_name: {script_name: [trajectories]}}
    all_trajectories = {
        subtask_name: {
            script_name: trajectories
        }
    }
    
    # Aggregate trajectories into suites
    aggregated_trajectories = aggregate_trajectories(all_trajectories)
    
    return aggregated_trajectories
