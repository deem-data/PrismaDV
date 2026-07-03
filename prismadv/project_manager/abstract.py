"""Abstract project manager with delegated responsibilities.

This module provides the AbstractProjectManager class that delegates
responsibilities to specialized manager classes while maintaining
backward compatibility with the original API.
"""

from abc import ABC
from pathlib import Path
from typing import List, Dict, Any

from prismadv.data_models import ValidationResults, ConstraintsWithSources
from prismadv.data_models.trajectory import DVTrajectory
from prismadv.project_manager.managers import (
    PathManager,
    MetadataManager,
    ScriptManager,
    DataManager,
    ConstraintManager,
    ExecutionManager,
    AnnotationManager,
    TrajectoryManager,
)


class AbstractProjectManager(ABC):
    """Abstract base class for project management with delegated responsibilities.
    
    This class maintains backward compatibility by delegating operations to
    specialized manager classes while keeping the same public API.
    """

    def __init__(self, project_root: Path = None, dataset_name: str = None):
        """Initialize the project manager and all specialized managers.
        
        Args:
            project_root: Root directory of the project. If None, uses get_project_root().
            dataset_name: Name of the dataset. Must not be None.
        """
        super().__init__()
        
        # Initialize specialized managers
        self._path_manager = PathManager(project_root, dataset_name)
        self._metadata_manager = MetadataManager(self._path_manager)
        self._script_manager = ScriptManager(self._path_manager, self._metadata_manager)
        self._data_manager = DataManager(self._path_manager, self._metadata_manager)
        self._constraint_manager = ConstraintManager(self._data_manager)
        self._execution_manager = ExecutionManager(self._data_manager)
        self._annotation_manager = AnnotationManager(self._path_manager, self._metadata_manager)
        self._trajectory_manager = TrajectoryManager(
            self._path_manager, self._metadata_manager, self._script_manager,
            self._data_manager, self._constraint_manager, self._execution_manager
        )
        
        # Expose commonly used attributes for backward compatibility
        self.project_root = self._path_manager.project_root
        self.dataset_name = self._path_manager.dataset_name
        self.dataset_path = self._path_manager.dataset_path

    # ========== Path Properties (delegated to PathManager) ==========
    
    @property
    def project_data_root(self) -> Path:
        """The default data path is the project root/data."""
        return self._path_manager.project_data_root

    @property
    def processed_data_root(self) -> Path:
        """The default processed data path is the project root/data_processed."""
        return self._path_manager.processed_data_root

    @property
    def scripts_root(self) -> Path:
        """The default scripts path is the dataset_path/scripts."""
        return self._path_manager.scripts_root

    @property
    def annotations_root(self) -> Path:
        """The default annotations path is the dataset_path/annotations."""
        return self._path_manager.annotations_root

    @property
    def files_root(self) -> Path:
        """The default files path is the dataset_path/files."""
        return self._path_manager.files_root

    @property
    def errors_root(self) -> Path:
        """The default errors path is the dataset_path/errors."""
        return self._path_manager.errors_root

    # ========== Metadata Properties and Methods (delegated to MetadataManager) ==========
    
    @property
    def downstream_processed_path_mapping(self) -> Dict[str, Any]:
        """Get the mapping from scripts to processed data paths."""
        return self._metadata_manager.downstream_processed_path_mapping

    @property
    def raw_data_info(self) -> Dict[str, Any]:
        """Get information about raw data files."""
        return self._metadata_manager.raw_data_info

    def get_task_name_from_subtask(self, subtask_name: str) -> str:
        """Get the parent task name for a given subtask."""
        return self._metadata_manager.get_task_name_from_subtask(subtask_name)

    def get_subtask_description(self, subtask_name: str) -> str:
        """Get the description of a subtask."""
        return self._metadata_manager.get_subtask_description(subtask_name)

    def get_subtask_info(self, subtask_name: str) -> Dict[str, Any]:
        """Get the full information dictionary for a subtask."""
        return self._metadata_manager.get_subtask_info(subtask_name)

    def get_available_subtasks(self) -> List[str]:
        """Get a list of all available subtasks across all tasks."""
        return self._metadata_manager.get_available_subtasks()

    # ========== Script Methods (delegated to ScriptManager) ==========
    
    def get_script_paths_with_prefix(self, task_name: str, prefix: str) -> List[Path]:
        """Get all script paths for a given task with a specific prefix."""
        return self._script_manager.get_script_paths_with_prefix(task_name, prefix)

    def get_script_path(self, task_name: str, script_name: str) -> Path:
        """Get the script path for a given task and script name."""
        return self._script_manager.get_script_path(task_name, script_name)

    def get_available_script_info(self) -> Dict[str, Dict[str, List[Path]]]:
        """Get information about all available scripts organized by task and subtask."""
        return self._script_manager.get_available_script_info()

    def get_available_script_path_list_for_subtask(self, subtask_name: str) -> List[Path]:
        """Get a list of all script paths for a given subtask."""
        return self._script_manager.get_available_script_path_list_for_subtask(subtask_name)

    # ========== Data Methods (delegated to DataManager) ==========
    
    def get_base_processed_data_path_for_subtask(self, subtask_name: str) -> Path:
        """Get the base processed data path for a subtask (without label)."""
        return self._data_manager.get_base_processed_data_path_for_subtask(subtask_name)

    def get_processed_data_path_for_subtask(self, subtask_name: str, processed_data_label: str) -> Path:
        """Get the processed data path for a subtask with a specific label."""
        return self._data_manager.get_processed_data_path_for_subtask(subtask_name, processed_data_label)

    def get_available_processed_data_labels_for_subtask(self, subtask_name: str) -> List[str]:
        """Get a list of available processed data labels for a given subtask."""
        return self._data_manager.get_available_processed_data_labels_for_subtask(subtask_name)

    def get_new_data_path(self, subtask_name: str, processed_data_label: str, clean: bool = True) -> Path:
        """Get the path to new data (clean or corrupted)."""
        return self._data_manager.get_new_data_path(subtask_name, processed_data_label, clean)

    def get_new_test_data_path(self, subtask_name: str, processed_data_label: str, clean: bool) -> Path:
        """Get the path to the new test data CSV file."""
        return self._data_manager.get_new_test_data_path(subtask_name, processed_data_label, clean)

    def get_observed_data_path(self, subtask_name: str, processed_data_label: str) -> Path:
        """Get the path to the observed data CSV file."""
        return self._data_manager.get_observed_data_path(subtask_name, processed_data_label)

    # ========== Constraint Methods (delegated to ConstraintManager) ==========
    
    def get_task_agnostic_constraint_path(self, subtask_name: str, processed_data_label: str) -> Path:
        """Get the base constraints directory path (task-agnostic)."""
        return self._constraint_manager.get_task_agnostic_constraint_path(subtask_name, processed_data_label)

    def get_constraints_path(self, subtask_name: str, processed_data_label: str, script_name: str) -> Path:
        """Get the constraints path for a specific script."""
        return self._constraint_manager.get_constraints_path(subtask_name, processed_data_label, script_name)

    def get_constraints_validation_path(self, subtask_name: str, processed_data_label: str, 
                                       script_name: str) -> Path:
        """Get the constraints validation path for a specific script."""
        return self._constraint_manager.get_constraints_validation_path(
            subtask_name, processed_data_label, script_name
        )

    def get_task_agnostic_constraints_validation_path(self, subtask_name: str, 
                                                      processed_data_label: str) -> Path:
        """Get the base constraints validation directory path (task-agnostic)."""
        return self._constraint_manager.get_task_agnostic_constraints_validation_path(
            subtask_name, processed_data_label
        )

    def get_inferred_constraints_path(self, subtask_name: str, script_name: str, 
                                     processed_data_label: str) -> Path:
        """Get the path to inferred constraints for a script."""
        return self._constraint_manager.get_inferred_constraints_path(
            subtask_name, script_name, processed_data_label
        )

    # ========== Execution Methods (delegated to ExecutionManager) ==========
    
    def get_execution_output_dir(self, subtask_name: str, processed_data_label: str, 
                                 script_name: str) -> Path:
        """Get the base execution output directory for a script."""
        return self._execution_manager.get_execution_output_dir(subtask_name, processed_data_label, script_name)

    def get_execution_output_path(self, subtask_name: str, processed_data_label: str, 
                                  script_name: str, clean: bool) -> Path:
        """Get the execution output path for clean or corrupted data."""
        return self._execution_manager.get_execution_output_path(
            subtask_name, processed_data_label, script_name, clean
        )

    def get_execution_output_validation_path(self, subtask_name: str, processed_data_label: str, 
                                            script_name: str) -> Path:
        """Get the execution output validation path for a script."""
        return self._execution_manager.get_execution_output_validation_path(
            subtask_name, processed_data_label, script_name
        )

    # ========== Annotation Methods (delegated to AnnotationManager) ==========
    
    def get_annotation_file_path(self, task_name: str, script_name: str) -> Path:
        """Get the annotation file path for a given task and script name."""
        return self._annotation_manager.get_annotation_file_path(task_name, script_name)

    def get_annotation_file_path_from_subtask(self, subtask_name: str, script_name: str, 
                                             ok_if_not_exist: bool = False) -> Path:
        """Get the annotation file path for a given subtask and script name."""
        return self._annotation_manager.get_annotation_file_path_from_subtask(
            subtask_name, script_name, ok_if_not_exist
        )

    # ========== Trajectory Methods (delegated to TrajectoryManager) ==========
    
    def get_dspy_trajectory(self, subtask_name: str, processed_data_label: str, script_name: str, 
                           llm_name: str = None, dspy_prefix: str = None) -> List[DVTrajectory]:
        """Get DSPy trajectories for a specific script and processed data label."""
        return self._trajectory_manager.get_dspy_trajectory(
            subtask_name, processed_data_label, script_name, llm_name, dspy_prefix
        )

    def get_dspy_trajectories(self, subtask_name: str, processed_data_label_list: List[str], 
                             script_name_list: List[str], llm_name: str = None, 
                             dspy_prefix: str = None) -> List[DVTrajectory]:
        """Get DSPy trajectories for multiple scripts and processed data labels."""
        return self._trajectory_manager.get_dspy_trajectories(
            subtask_name, processed_data_label_list, script_name_list, llm_name, dspy_prefix
        )

    def format_trajectory_for_dspy(self, dataset_name: str, llm_name: str, script_path: Path, 
                                  script, data_path: Path, constraints: ConstraintsWithSources,
                                  validation_results: ValidationResults, processed_data_label: str, 
                                  is_safe: bool) -> List[DVTrajectory]:
        """Format constraint and validation data into DSPy trajectory objects."""
        return self._trajectory_manager.format_trajectory_for_dspy(
            dataset_name, llm_name, script_path, script, data_path, constraints,
            validation_results, processed_data_label, is_safe
        )
