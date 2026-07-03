"""Unit tests for MetadataManager."""

import pytest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch
from prismadv.project_manager.managers import MetadataManager


@pytest.fixture
def mock_path_manager(tmp_path):
    """Create a mock PathManager."""
    pm = Mock()
    pm.scripts_root = tmp_path / "scripts"
    pm.scripts_root.mkdir(parents=True, exist_ok=True)
    return pm


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        'script_to_processed_mapping': {
            'task1': {
                'subtask1': {
                    'description': 'Test subtask 1',
                    'script_prefix': 'test_',
                    'processed_folder_name': 'processed1'
                },
                'subtask2': {
                    'description': 'Test subtask 2',
                    'script_prefix': 'test_',
                    'processed_folder_name': 'processed2'
                }
            },
            'task2': {
                'subtask3': {
                    'description': 'Test subtask 3',
                    'script_prefix': 'task2_',
                    'processed_folder_name': 'processed3'
                }
            }
        },
        'raw_data_info': {
            'file1': 'info1'
        }
    }


def test_metadata_manager_initialization(mock_path_manager):
    """Test MetadataManager initialization."""
    mm = MetadataManager(mock_path_manager)
    assert mm._path_manager == mock_path_manager
    assert mm._metadata_cache is None


def test_metadata_manager_loads_metadata(mock_path_manager, sample_metadata):
    """Test that MetadataManager loads metadata from YAML."""
    metadata_file = mock_path_manager.scripts_root / "metadata.yaml"
    
    with patch('builtins.open', mock_open(read_data='script_to_processed_mapping: {}')):
        with patch('oyaml.safe_load', return_value=sample_metadata):
            mm = MetadataManager(mock_path_manager)
            mapping = mm.downstream_processed_path_mapping
            
            assert mapping == sample_metadata['script_to_processed_mapping']


def test_get_task_name_from_subtask(mock_path_manager, sample_metadata):
    """Test getting task name from subtask."""
    with patch('builtins.open', mock_open()):
        with patch('oyaml.safe_load', return_value=sample_metadata):
            mm = MetadataManager(mock_path_manager)
            
            assert mm.get_task_name_from_subtask('subtask1') == 'task1'
            assert mm.get_task_name_from_subtask('subtask3') == 'task2'


def test_get_task_name_from_invalid_subtask(mock_path_manager, sample_metadata):
    """Test error when subtask doesn't exist."""
    with patch('builtins.open', mock_open()):
        with patch('oyaml.safe_load', return_value=sample_metadata):
            mm = MetadataManager(mock_path_manager)
            
            with pytest.raises(ValueError, match="Subtask <invalid> is not available"):
                mm.get_task_name_from_subtask('invalid')


def test_get_subtask_description(mock_path_manager, sample_metadata):
    """Test getting subtask description."""
    with patch('builtins.open', mock_open()):
        with patch('oyaml.safe_load', return_value=sample_metadata):
            mm = MetadataManager(mock_path_manager)
            
            assert mm.get_subtask_description('subtask1') == 'Test subtask 1'
            assert mm.get_subtask_description('subtask3') == 'Test subtask 3'


def test_get_available_subtasks(mock_path_manager, sample_metadata):
    """Test getting all available subtasks."""
    with patch('builtins.open', mock_open()):
        with patch('oyaml.safe_load', return_value=sample_metadata):
            mm = MetadataManager(mock_path_manager)
            
            subtasks = mm.get_available_subtasks()
            assert 'subtask1' in subtasks
            assert 'subtask2' in subtasks
            assert 'subtask3' in subtasks
            assert len(subtasks) == 3

