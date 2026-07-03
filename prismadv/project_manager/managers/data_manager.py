"""Data manager for handling data path resolution."""

from pathlib import Path
from typing import List


class DataManager:
    """Manages data path resolution for processed data and test data.
    
    This class handles finding and resolving paths to processed data,
    new test data, and observed data files.
    """

    def __init__(self, path_manager, metadata_manager):
        """Initialize the data manager.
        
        Args:
            path_manager: PathManager instance for accessing directory paths.
            metadata_manager: MetadataManager instance for accessing metadata.
        """
        self._path_manager = path_manager
        self._metadata_manager = metadata_manager

    def get_base_processed_data_path_for_subtask(self, subtask_name: str) -> Path:
        """Get the base processed data path for a subtask (without label).
        
        Args:
            subtask_name: Name of the subtask.
            
        Returns:
            Path to the base processed data directory.
            
        Raises:
            ValueError: If subtask is not found.
        """
        downstream_processed_path_dict = {}
        for task_type, task_info in self._metadata_manager.downstream_processed_path_mapping.items():
            for subtask, subtask_info in task_info.items():
                downstream_processed_path_dict[subtask] = subtask_info["processed_folder_name"]
        if subtask_name not in downstream_processed_path_dict.keys():
            raise ValueError(f"Subtask <{subtask_name}> is not available in the processed data mapping.")
        return (self._path_manager.processed_data_root /
                f"{self._path_manager.dataset_name}" /
                f"{downstream_processed_path_dict[subtask_name]}")

    def get_processed_data_path_for_subtask(self, subtask_name: str, processed_data_label: str) -> Path:
        """Get the processed data path for a subtask with a specific label.
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            
        Returns:
            Path to the processed data directory.
            
        Raises:
            ValueError: If subtask is not found or path doesn't exist.
        """
        downstream_processed_path_dict = {}
        for task_type, task_info in self._metadata_manager.downstream_processed_path_mapping.items():
            for subtask, subtask_info in task_info.items():
                downstream_processed_path_dict[subtask] = subtask_info["processed_folder_name"]
        if subtask_name not in downstream_processed_path_dict.keys():
            raise ValueError(f"Subtask <{subtask_name}> is not available in the processed data mapping.")

        processed_data_path = (self._path_manager.processed_data_root /
                               f"{self._path_manager.dataset_name}" /
                               f"{downstream_processed_path_dict[subtask_name]}" /
                               f"{processed_data_label}")
        if not processed_data_path.exists():
            raise ValueError(
                f"Processed data path <{processed_data_path}> does not exist.\n"
                f"Please check if the processed data has been generated for subtask <{subtask_name}> "
                f"with label <{processed_data_label}>.")
        return processed_data_path

    def get_available_processed_data_labels_for_subtask(self, subtask_name: str) -> List[str]:
        """Get a list of available processed data labels for a given subtask.
        
        Args:
            subtask_name: Name of the subtask.
            
        Returns:
            Sorted list of processed data labels.
            
        Raises:
            ValueError: If subtask is not found or processed data path doesn't exist.
        """
        task_name = self._metadata_manager.get_task_name_from_subtask(subtask_name)
        subtask_info = self._metadata_manager.downstream_processed_path_mapping[task_name][subtask_name]
        processed_data_path = (self._path_manager.processed_data_root /
                               f"{self._path_manager.dataset_name}" /
                               f"{subtask_info['processed_folder_name']}")
        if not processed_data_path.exists():
            raise ValueError(
                f"Processed data path <{processed_data_path}> does not exist.\n"
                f"Please check if the processed data has been generated for subtask <{subtask_name}>.")
        processed_data_labels = [item.name for item in processed_data_path.iterdir() if item.is_dir()]
        processed_data_labels = sorted(processed_data_labels)
        return processed_data_labels

    def get_new_data_path(self, subtask_name: str, processed_data_label: str, clean: bool = True) -> Path:
        """Get the path to new data (clean or corrupted).
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            clean: If True, returns path to clean data; if False, returns path to corrupted data.
            
        Returns:
            Path to the new data directory.
        """
        if clean:
            return (self.get_processed_data_path_for_subtask(subtask_name, processed_data_label) /
                    "files_with_clean_new_data")
        else:
            return (self.get_processed_data_path_for_subtask(subtask_name, processed_data_label) /
                    "files_with_corrupted_new_data")

    def get_new_test_data_path(self, subtask_name: str, processed_data_label: str, clean: bool) -> Path:
        """Get the path to the new test data CSV file.
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            clean: If True, returns path to clean data; if False, returns path to corrupted data.
            
        Returns:
            Path to the new test data CSV file.
        """
        task_name = self._metadata_manager.get_task_name_from_subtask(subtask_name)
        new_data_file_name = (self._metadata_manager.downstream_processed_path_mapping[task_name]
        [subtask_name]["new_data_file_name"])
        return self.get_new_data_path(subtask_name, processed_data_label, clean) / f"{new_data_file_name}.csv"

    def get_observed_data_path(self, subtask_name: str, processed_data_label: str) -> Path:
        """Get the path to the observed data CSV file.
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            
        Returns:
            Path to the observed data CSV file.
        """
        task_name = self._metadata_manager.get_task_name_from_subtask(subtask_name)
        observed_data_file_name = (self._metadata_manager.downstream_processed_path_mapping[task_name]
        [subtask_name]["observed_data_file_name"])
        return (self.get_new_data_path(subtask_name, processed_data_label, clean=True) /
                f"{observed_data_file_name}.csv")
