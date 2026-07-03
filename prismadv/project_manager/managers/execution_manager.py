"""Execution manager for handling execution output paths."""

from pathlib import Path


class ExecutionManager:
    """Manages execution output path resolution.
    
    This class handles finding and resolving paths to execution outputs
    and validation results.
    """

    def __init__(self, data_manager):
        """Initialize the execution manager.
        
        Args:
            data_manager: DataManager instance for accessing data paths.
        """
        self._data_manager = data_manager

    def get_execution_output_dir(self, subtask_name: str, processed_data_label: str,
                                 script_name: str) -> Path:
        """Get the base execution output directory for a script.
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            script_name: Name of the script.
            
        Returns:
            Path to the execution output directory.
        """
        return (self._data_manager.get_processed_data_path_for_subtask(subtask_name, processed_data_label) /
                "output" / f"{script_name}")

    def get_execution_output_path(self, subtask_name: str, processed_data_label: str,
                                  script_name: str, clean: bool) -> Path:
        """Get the execution output path for clean or corrupted data.
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            script_name: Name of the script.
            clean: If True, returns path for clean data results; if False, for corrupted data.
            
        Returns:
            Path to the execution output directory.
        """
        if clean:
            return self.get_execution_output_dir(subtask_name, processed_data_label, script_name) / \
                "results_on_clean_new_data"
        else:
            return self.get_execution_output_dir(subtask_name, processed_data_label, script_name) / \
                "results_on_corrupted_new_data"

    def get_execution_output_validation_path(self, subtask_name: str, processed_data_label: str,
                                             script_name: str) -> Path:
        """Get the execution output validation path for a script.
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            script_name: Name of the script.
            
        Returns:
            Path to the execution output validation directory.
        """
        return (self._data_manager.get_processed_data_path_for_subtask(subtask_name, processed_data_label) /
                "output_validation" / f"{script_name}")
