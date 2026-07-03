from pathlib import Path
from typing import Union

from prismadv.project_manager.abstract import AbstractProjectManager


class ProjectManager(AbstractProjectManager):

    def __init__(self, project_root: Union[Path, str] = None, dataset_name: str = None,
                 downstream_task_type: str = None):
        super().__init__(project_root=project_root, dataset_name=dataset_name)

        self._available_datasets = [item.name for item in self.project_data_root.iterdir()]

        self.__available_tasks = []
        if not self.dataset_path.exists():
            raise ValueError(
                f"Dataset {dataset_name} is not available in the project data path. Available datasets are: {self._available_datasets}")
        if self.scripts_root.exists():
            self.__available_tasks = [item.name for item in self.scripts_root.iterdir()]
        if downstream_task_type is not None and downstream_task_type not in self.__available_tasks:
            raise ValueError(
                f"Downstream task type {downstream_task_type} is not available in the scripts path. Available tasks are: {self.__available_tasks}")
        elif downstream_task_type is None:
            self.downstream_task_types = [item.name for item in self.scripts_root.iterdir() if item.is_dir()]
        elif isinstance(downstream_task_type, str):
            self.downstream_task_types = [downstream_task_type]
            self.downstream_task_type_paths = [self.scripts_root / downstream_task_type]
        else:
            raise ValueError("Invalid downstream task types, downstream_task_type should be a string")

    def get_available_script_table_pairs(self):
        """
        Get a list of available script and dataset pairs.
        Inputs:
        - None
        Outputs:
        - Pandas DataFrame with columns: "dataset_name", "subtask_name", "script_name", "raw_data_path", "script_path", "script_type"
        """
        print(self.dataset_name)
        print(self.downstream_task_types)
        print(self.downstream_task_type_paths)
