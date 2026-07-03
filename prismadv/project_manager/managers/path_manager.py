"""Path manager for handling project directory structure."""

from pathlib import Path

from prismadv.utils import get_project_root


class PathManager:
    """Manages core path resolution for project directories.
    
    This class handles the basic directory structure of the project,
    including data, processed data, scripts, annotations, files, and errors paths.
    """

    def __init__(self, project_root: Path = None, dataset_name: str = None):
        """Initialize the path manager.
        
        Args:
            project_root: Root directory of the project. If None, uses get_project_root().
            dataset_name: Name of the dataset. Must not be None.
            
        Raises:
            ValueError: If dataset_name is None.
        """
        if project_root is None:
            self.project_root = get_project_root()
        else:
            self.project_root = project_root

        if dataset_name is None:
            raise ValueError('dataset_name cannot be None')

        self.dataset_name = dataset_name
        self.dataset_path = self.project_data_root / dataset_name

    @property
    def project_data_root(self) -> Path:
        """The default data path is the project root/data.
        
        Returns:
            Path to the data directory.
        """
        return self.project_root / "benchmarks" / "EIDBench-synth"

    @property
    def processed_data_root(self) -> Path:
        """The default processed data path is the project root/data_processed.
        
        Returns:
            Path to the processed data directory.
        """
        return self.project_root / "data_processed"

    @property
    def scripts_root(self) -> Path:
        """The default scripts path is the dataset_path/scripts.
        
        Returns:
            Path to the scripts directory for this dataset.
        """
        return self.dataset_path / "scripts"

    @property
    def annotations_root(self) -> Path:
        """The default annotations path is the dataset_path/annotations.
        
        Returns:
            Path to the annotations directory for this dataset.
        """
        return self.dataset_path / "annotations"

    @property
    def files_root(self) -> Path:
        """The default files path is the dataset_path/files.
        
        Returns:
            Path to the files directory for this dataset.
        """
        return self.dataset_path / "files"

    @property
    def errors_root(self) -> Path:
        """The default errors path is the dataset_path/errors.
        
        Returns:
            Path to the errors directory for this dataset.
        """
        return self.dataset_path / "errors"
