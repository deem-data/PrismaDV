"""Annotation manager for handling annotation file operations."""

from pathlib import Path


class AnnotationManager:
    """Manages annotation file path resolution.
    
    This class handles finding and resolving paths to annotation files
    for scripts and tasks.
    """

    def __init__(self, path_manager, metadata_manager):
        """Initialize the annotation manager.
        
        Args:
            path_manager: PathManager instance for accessing directory paths.
            metadata_manager: MetadataManager instance for accessing metadata.
        """
        self._path_manager = path_manager
        self._metadata_manager = metadata_manager

    def get_annotation_file_path(self, task_name: str, script_name: str) -> Path:
        """Get the annotation file path for a given task and script name.
        
        Args:
            task_name: Name of the task.
            script_name: Name of the script (with or without extension).
            
        Returns:
            Path to the annotation file.
            
        Raises:
            ValueError: If annotation file doesn't exist.
        """
        if "." in script_name:
            script_name = script_name.split(".")[0]  # Remove file extension if present
        annotation_file_path = self._path_manager.annotations_root / f"{task_name}/{script_name}.yaml"
        if not annotation_file_path.exists():
            raise ValueError(f"Annotation file <{annotation_file_path}> does not exist.")
        return annotation_file_path

    def get_annotation_file_path_from_subtask(self, subtask_name: str, script_name: str,
                                              ok_if_not_exist: bool = False) -> Path:
        """Get the annotation file path for a given subtask and script name.
        
        Args:
            subtask_name: Name of the subtask.
            script_name: Name of the script (with or without extension).
            ok_if_not_exist: If True, doesn't raise error if file doesn't exist.
            
        Returns:
            Path to the annotation file.
            
        Raises:
            ValueError: If annotation file doesn't exist and ok_if_not_exist is False.
        """
        task_name = self._metadata_manager.get_task_name_from_subtask(subtask_name)
        if "." in script_name:
            script_name = script_name.split(".")[0]  # Remove file extension if present
        annotation_file_path = self._path_manager.annotations_root / f"{task_name}/{script_name}.yaml"
        if not annotation_file_path.exists() and not ok_if_not_exist:
            raise ValueError(f"Annotation file <{annotation_file_path}> does not exist.")
        return annotation_file_path
