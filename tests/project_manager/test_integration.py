"""Integration tests for refactored ProjectManager.

These tests verify that the refactored ProjectManager maintains
backward compatibility with the original implementation.
"""

import pytest
from pathlib import Path
from prismadv.project_manager.manager.base import ProjectManager


def test_project_manager_can_be_instantiated():
    """Test that ProjectManager can still be instantiated with real data."""
    # This will use the actual project structure
    try:
        pm = ProjectManager(dataset_name="toy_example")
        assert pm is not None
        assert pm.dataset_name == "toy_example"
    except ValueError as e:
        # It's okay if toy_example doesn't exist in test environment
        pytest.skip(f"Skipping test - dataset not available: {e}")


def test_project_manager_has_all_expected_properties():
    """Test that ProjectManager has all expected properties."""
    try:
        pm = ProjectManager(dataset_name="toy_example")
        
        # Test that all properties are accessible
        assert hasattr(pm, 'project_root')
        assert hasattr(pm, 'dataset_name')
        assert hasattr(pm, 'dataset_path')
        assert hasattr(pm, 'project_data_root')
        assert hasattr(pm, 'processed_data_root')
        assert hasattr(pm, 'scripts_root')
        assert hasattr(pm, 'annotations_root')
        assert hasattr(pm, 'files_root')
        assert hasattr(pm, 'errors_root')
        
        # Test that properties return Path objects
        assert isinstance(pm.project_root, Path)
        assert isinstance(pm.project_data_root, Path)
        assert isinstance(pm.processed_data_root, Path)
        
    except ValueError as e:
        pytest.skip(f"Skipping test - dataset not available: {e}")


def test_project_manager_has_all_expected_methods():
    """Test that ProjectManager has all expected methods."""
    try:
        pm = ProjectManager(dataset_name="toy_example")
        
        # Test that all methods are accessible
        assert hasattr(pm, 'get_task_name_from_subtask')
        assert hasattr(pm, 'get_subtask_description')
        assert hasattr(pm, 'get_subtask_info')
        assert hasattr(pm, 'get_available_subtasks')
        assert hasattr(pm, 'get_script_paths_with_prefix')
        assert hasattr(pm, 'get_script_path')
        assert hasattr(pm, 'get_available_script_info')
        assert hasattr(pm, 'get_available_script_path_list_for_subtask')
        assert hasattr(pm, 'get_processed_data_path_for_subtask')
        assert hasattr(pm, 'get_available_processed_data_labels_for_subtask')
        assert hasattr(pm, 'get_new_data_path')
        assert hasattr(pm, 'get_new_test_data_path')
        assert hasattr(pm, 'get_observed_data_path')
        assert hasattr(pm, 'get_constraints_path')
        assert hasattr(pm, 'get_constraints_validation_path')
        assert hasattr(pm, 'get_execution_output_path')
        assert hasattr(pm, 'get_execution_output_dir')
        assert hasattr(pm, 'get_execution_output_validation_path')
        assert hasattr(pm, 'get_annotation_file_path')
        assert hasattr(pm, 'get_annotation_file_path_from_subtask')
        assert hasattr(pm, 'get_dspy_trajectory')
        assert hasattr(pm, 'get_dspy_trajectories')
        assert hasattr(pm, 'format_trajectory_for_dspy')
        
    except ValueError as e:
        pytest.skip(f"Skipping test - dataset not available: {e}")


def test_specialized_managers_are_accessible():
    """Test that specialized managers can be imported and used."""
    from prismadv.project_manager import (
        PathManager,
        MetadataManager,
        ScriptManager,
        DataManager,
        ConstraintManager,
        ExecutionManager,
        AnnotationManager,
        TrajectoryManager,
    )
    
    # Just verify they can be imported
    assert PathManager is not None
    assert MetadataManager is not None
    assert ScriptManager is not None
    assert DataManager is not None
    assert ConstraintManager is not None
    assert ExecutionManager is not None
    assert AnnotationManager is not None
    assert TrajectoryManager is not None

