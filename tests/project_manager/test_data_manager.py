"""Unit tests for DataManager."""

import pytest
from pathlib import Path
from unittest.mock import Mock
from prismadv.project_manager.managers import DataManager


@pytest.fixture
def mock_managers(tmp_path):
    """Create mock PathManager and MetadataManager."""
    path_manager = Mock()
    path_manager.dataset_name = "test_dataset"
    path_manager.processed_data_root = tmp_path / "data_processed"
    path_manager.processed_data_root.mkdir(parents=True, exist_ok=True)
    
    metadata_manager = Mock()
    metadata_manager.downstream_processed_path_mapping = {
        'task1': {
            'subtask1': {
                'processed_folder_name': 'processed1',
                'new_data_file_name': 'test_data',
                'observed_data_file_name': 'observed_data'
            }
        }
    }
    metadata_manager.get_task_name_from_subtask = Mock(return_value='task1')
    
    return path_manager, metadata_manager


def test_data_manager_initialization(mock_managers):
    """Test DataManager initialization."""
    path_manager, metadata_manager = mock_managers
    dm = DataManager(path_manager, metadata_manager)
    
    assert dm._path_manager == path_manager
    assert dm._metadata_manager == metadata_manager


def test_get_base_processed_data_path_for_subtask(mock_managers):
    """Test getting base processed data path."""
    path_manager, metadata_manager = mock_managers
    dm = DataManager(path_manager, metadata_manager)
    
    path = dm.get_base_processed_data_path_for_subtask("subtask1")
    expected = path_manager.processed_data_root / "test_dataset" / "processed1"
    assert path == expected


def test_get_processed_data_path_for_subtask(mock_managers):
    """Test getting processed data path with label."""
    path_manager, metadata_manager = mock_managers
    dm = DataManager(path_manager, metadata_manager)
    
    # Create the directory so it exists
    processed_dir = path_manager.processed_data_root / "test_dataset" / "processed1" / "0"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    path = dm.get_processed_data_path_for_subtask("subtask1", "0")
    assert path == processed_dir


def test_get_processed_data_path_not_found(mock_managers):
    """Test error when processed data path doesn't exist."""
    path_manager, metadata_manager = mock_managers
    dm = DataManager(path_manager, metadata_manager)
    
    with pytest.raises(ValueError, match="Processed data path .* does not exist"):
        dm.get_processed_data_path_for_subtask("subtask1", "999")


def test_get_new_data_path(mock_managers):
    """Test getting new data path (clean/corrupted)."""
    path_manager, metadata_manager = mock_managers
    dm = DataManager(path_manager, metadata_manager)
    
    # Create the directory
    processed_dir = path_manager.processed_data_root / "test_dataset" / "processed1" / "0"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    clean_path = dm.get_new_data_path("subtask1", "0", clean=True)
    assert clean_path.name == "files_with_clean_new_data"
    
    corrupted_path = dm.get_new_data_path("subtask1", "0", clean=False)
    assert corrupted_path.name == "files_with_corrupted_new_data"


def test_get_new_test_data_path(mock_managers):
    """Test getting new test data CSV path."""
    path_manager, metadata_manager = mock_managers
    dm = DataManager(path_manager, metadata_manager)
    
    # Create the directory
    processed_dir = path_manager.processed_data_root / "test_dataset" / "processed1" / "0"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    path = dm.get_new_test_data_path("subtask1", "0", clean=True)
    assert path.name == "test_data.csv"
    assert "files_with_clean_new_data" in str(path)

