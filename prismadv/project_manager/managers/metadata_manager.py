"""Metadata manager for handling project metadata loading and parsing."""

from typing import Dict, Any, List

import oyaml as yaml


class MetadataManager:
    """Manages metadata loading and parsing for project tasks and subtasks.
    
    This class handles reading and interpreting the metadata.yaml file
    that describes task structure, subtask information, and data mappings.
    """

    def __init__(self, path_manager):
        """Initialize the metadata manager.
        
        Args:
            path_manager: PathManager instance for accessing directory paths.
        """
        self._path_manager = path_manager
        self._metadata_cache = None

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from the metadata.yaml file.
        
        Returns:
            Dictionary containing metadata information.
            
        Raises:
            FileNotFoundError: If metadata.yaml doesn't exist.
        """
        if self._metadata_cache is None:
            metadata_path = self._path_manager.scripts_root / "metadata.yaml"
            with open(metadata_path, 'r') as f:
                self._metadata_cache = yaml.safe_load(f)
        return self._metadata_cache

    @property
    def downstream_processed_path_mapping(self) -> Dict[str, Any]:
        """Get the mapping from scripts to processed data paths.
        
        Returns:
            Dictionary mapping task names to their subtask information.
        """
        metadata = self._load_metadata()
        return metadata['script_to_processed_mapping']

    @property
    def raw_data_info(self) -> Dict[str, Any]:
        """Get information about raw data files.
        
        Returns:
            Dictionary containing raw data information.
        """
        metadata = self._load_metadata()
        return metadata['raw_data_info']

    def get_task_name_from_subtask(self, subtask_name: str) -> str:
        """Get the parent task name for a given subtask.
        
        Args:
            subtask_name: Name of the subtask.
            
        Returns:
            Name of the parent task.
            
        Raises:
            ValueError: If subtask is not found in any task.
        """
        for task_name, task_info in self.downstream_processed_path_mapping.items():
            if subtask_name in task_info:
                return task_name
        raise ValueError(f"Subtask <{subtask_name}> is not available in the processed data mapping.")

    def get_subtask_info(self, subtask_name: str) -> Dict[str, Any]:
        """Get the full information dictionary for a subtask.
        
        Args:
            subtask_name: Name of the subtask.
            
        Returns:
            Dictionary containing subtask information.
            
        Raises:
            ValueError: If subtask is not found.
        """
        task_name = self.get_task_name_from_subtask(subtask_name)
        if subtask_name not in self.downstream_processed_path_mapping[task_name]:
            raise ValueError(
                f"Subtask <{subtask_name}> is not available in the processed data mapping for task <{task_name}>. "
                f"Available subtasks are: {list(self.downstream_processed_path_mapping[task_name].keys())}")
        return self.downstream_processed_path_mapping[task_name][subtask_name]

    def get_subtask_description(self, subtask_name: str) -> str:
        """Get the description of a subtask.
        
        Args:
            subtask_name: Name of the subtask.
            
        Returns:
            Description string for the subtask.
        """
        subtask_info = self.get_subtask_info(subtask_name)
        return subtask_info["description"]

    def get_available_subtasks(self) -> List[str]:
        """Get a list of all available subtasks across all tasks.
        
        Returns:
            List of subtask names.
        """
        available_tasks = self.downstream_processed_path_mapping.keys()
        available_subtasks = []
        for task in available_tasks:
            available_subtasks += self.downstream_processed_path_mapping[task].keys()
        return available_subtasks
