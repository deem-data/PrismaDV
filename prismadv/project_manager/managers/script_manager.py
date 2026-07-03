"""Script manager for handling script discovery and management."""

from pathlib import Path
from typing import List, Dict


class ScriptManager:
    """Manages script operations including discovery, path resolution, and metadata.
    
    This class handles finding and managing script files associated with tasks
    and subtasks in the project.
    """

    def __init__(self, path_manager, metadata_manager):
        """Initialize the script manager.
        
        Args:
            path_manager: PathManager instance for accessing directory paths.
            metadata_manager: MetadataManager instance for accessing metadata.
        """
        self._path_manager = path_manager
        self._metadata_manager = metadata_manager

    def get_script_paths_with_prefix(self, task_name: str, prefix: str) -> List[Path]:
        """Get all script paths for a given task with a specific prefix.
        
        Args:
            task_name: Name of the task.
            prefix: Prefix to filter script files by.
            
        Returns:
            List of Path objects for matching scripts.
        """
        script_paths = []
        task_scripts_dir = self._path_manager.scripts_root / task_name
        for script in sorted(task_scripts_dir.iterdir()):
            if script.is_file() and script.name.startswith(prefix):
                script_paths.append(script)
        return script_paths

    def get_script_path(self, task_name: str, script_name: str) -> Path:
        """Get the script path for a given task and script name.
        
        Args:
            task_name: Name of the task.
            script_name: Name of the script file (with or without .py extension).
            
        Returns:
            Path to the script file.
            
        Raises:
            ValueError: If script file doesn't exist.
        """
        if not script_name.endswith(".py"):
            script_name = f"{script_name}.py"
        script_path = self._path_manager.scripts_root / task_name / script_name
        if not script_path.exists():
            raise ValueError(f"Script file <{script_path}> does not exist.")
        return script_path

    def get_available_script_info(self) -> Dict[str, Dict[str, List[Path]]]:
        """Get information about all available scripts organized by task and subtask.
        
        Returns:
            Nested dictionary mapping task names to subtask names to script paths.
        """
        script_info_dict = {}
        for task_name, task_info in self._metadata_manager.downstream_processed_path_mapping.items():
            script_info_dict[task_name] = {}
            for subtask_name, subtask_info in task_info.items():
                scripts_paths = self.get_script_paths_with_prefix(task_name, subtask_info['script_prefix'])
                script_info_dict[task_name][subtask_name] = scripts_paths
        return script_info_dict

    def get_available_script_path_list_for_subtask(self, subtask_name: str) -> List[Path]:
        """Get a list of all script paths for a given subtask.
        
        Args:
            subtask_name: Name of the subtask.
            
        Returns:
            Sorted list of Path objects for scripts associated with the subtask.
            
        Raises:
            ValueError: If subtask is not found.
        """
        script_path_list = []
        for task_name, task_info in self._metadata_manager.downstream_processed_path_mapping.items():
            if subtask_name in task_info:
                scripts_paths = self.get_script_paths_with_prefix(
                    task_name, task_info[subtask_name]['script_prefix']
                )
                script_path_list += scripts_paths
        if not script_path_list:
            raise ValueError(f"Subtask <{subtask_name}> is not available in the processed data mapping.")
        script_path_list.sort(key=lambda x: x.name)
        return script_path_list
