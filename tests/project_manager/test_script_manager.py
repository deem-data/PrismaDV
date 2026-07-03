"""Unit tests for ScriptManager."""

import pytest
from pathlib import Path
from unittest.mock import Mock
from prismadv.project_manager.managers import ScriptManager


@pytest.fixture
def mock_managers(tmp_path):
    """Create mock PathManager and MetadataManager."""
    path_manager = Mock()
    path_manager.scripts_root = tmp_path / "scripts"
    path_manager.scripts_root.mkdir(parents=True, exist_ok=True)
    
    metadata_manager = Mock()
    metadata_manager.downstream_processed_path_mapping = {
        'task1': {
            'subtask1': {
                'script_prefix': 'test_',
                'processed_folder_name': 'processed1'
            }
        }
    }
    
    return path_manager, metadata_manager


def test_script_manager_initialization(mock_managers):
    """Test ScriptManager initialization."""
    path_manager, metadata_manager = mock_managers
    sm = ScriptManager(path_manager, metadata_manager)
    
    assert sm._path_manager == path_manager
    assert sm._metadata_manager == metadata_manager


def test_get_script_paths_with_prefix(mock_managers):
    """Test getting script paths with prefix."""
    path_manager, metadata_manager = mock_managers
    
    # Create test scripts
    task_dir = path_manager.scripts_root / "task1"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "test_script1.py").touch()
    (task_dir / "test_script2.py").touch()
    (task_dir / "other_script.py").touch()
    
    sm = ScriptManager(path_manager, metadata_manager)
    scripts = sm.get_script_paths_with_prefix("task1", "test_")
    
    assert len(scripts) == 2
    assert all(s.name.startswith("test_") for s in scripts)


def test_get_script_path(mock_managers):
    """Test getting a specific script path."""
    path_manager, metadata_manager = mock_managers
    
    # Create test script
    task_dir = path_manager.scripts_root / "task1"
    task_dir.mkdir(parents=True, exist_ok=True)
    script_file = task_dir / "test_script.py"
    script_file.touch()
    
    sm = ScriptManager(path_manager, metadata_manager)
    
    # Test with .py extension
    path = sm.get_script_path("task1", "test_script.py")
    assert path == script_file
    
    # Test without .py extension
    path = sm.get_script_path("task1", "test_script")
    assert path == script_file


def test_get_script_path_not_found(mock_managers):
    """Test error when script doesn't exist."""
    path_manager, metadata_manager = mock_managers
    
    task_dir = path_manager.scripts_root / "task1"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    sm = ScriptManager(path_manager, metadata_manager)
    
    with pytest.raises(ValueError, match="Script file .* does not exist"):
        sm.get_script_path("task1", "nonexistent.py")


def test_get_available_script_path_list_for_subtask(mock_managers):
    """Test getting all scripts for a subtask."""
    path_manager, metadata_manager = mock_managers
    
    # Create test scripts
    task_dir = path_manager.scripts_root / "task1"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "test_script1.py").touch()
    (task_dir / "test_script2.py").touch()
    
    sm = ScriptManager(path_manager, metadata_manager)
    scripts = sm.get_available_script_path_list_for_subtask("subtask1")
    
    assert len(scripts) == 2
    # Check that they're sorted
    assert scripts[0].name <= scripts[1].name

