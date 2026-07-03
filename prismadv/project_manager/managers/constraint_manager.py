"""Constraint manager for handling constraint-related paths."""

from pathlib import Path


class ConstraintManager:
    """Manages constraint-related path resolution.
    
    This class handles finding and resolving paths to constraint files,
    validation results, and inferred constraints.
    """

    def __init__(self, data_manager):
        """Initialize the constraint manager.
        
        Args:
            data_manager: DataManager instance for accessing data paths.
        """
        self._data_manager = data_manager

    def get_task_agnostic_constraint_path(self, subtask_name: str, processed_data_label: str) -> Path:
        """Get the base constraints directory path (task-agnostic).
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            
        Returns:
            Path to the constraints directory.
        """
        return (self._data_manager.get_processed_data_path_for_subtask(subtask_name, processed_data_label) /
                "constraints")

    def get_constraints_path(self, subtask_name: str, processed_data_label: str, script_name: str) -> Path:
        """Get the constraints path for a specific script.
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            script_name: Name of the script.
            
        Returns:
            Path to the script-specific constraints directory.
        """
        return (self._data_manager.get_processed_data_path_for_subtask(subtask_name, processed_data_label) /
                "constraints" / f"{script_name}")

    def get_constraints_validation_path(self, subtask_name: str, processed_data_label: str,
                                        script_name: str) -> Path:
        """Get the constraints validation path for a specific script.
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            script_name: Name of the script.
            
        Returns:
            Path to the script-specific constraints validation directory.
        """
        return (self._data_manager.get_processed_data_path_for_subtask(subtask_name, processed_data_label) /
                "constraints_validation" / f"{script_name}")

    def get_task_agnostic_constraints_validation_path(self, subtask_name: str,
                                                      processed_data_label: str) -> Path:
        """Get the base constraints validation directory path (task-agnostic).
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            
        Returns:
            Path to the constraints validation directory.
        """
        return (self._data_manager.get_processed_data_path_for_subtask(subtask_name, processed_data_label) /
                "constraints_validation")

    def get_inferred_constraints_path(self, subtask_name: str, script_name: str,
                                      processed_data_label: str) -> Path:
        """Get the path to inferred constraints for a script.
        
        Args:
            subtask_name: Name of the subtask.
            script_name: Name of the script (with or without extension).
            processed_data_label: Label identifying the processed data version.
            
        Returns:
            Path to the inferred constraints directory.
        """
        processed_data_path = self._data_manager.get_processed_data_path_for_subtask(
            subtask_name, processed_data_label
        )
        inferred_path = processed_data_path / "constraints" / f"{script_name.split('.')[0]}"
        return inferred_path
