"""Unit tests for PathManager."""

from pathlib import Path

import pytest

from prismadv.project_manager.managers import PathManager


def test_path_manager_initialization(tmp_path):
    """Test PathManager initialization with custom project root."""
    dataset_name = "test_dataset"
    pm = PathManager(project_root=tmp_path, dataset_name=dataset_name)

    assert pm.project_root == tmp_path
    assert pm.dataset_name == dataset_name
    assert pm.dataset_path == tmp_path / "benchmarks" / "EIDBench-synth" / dataset_name


def test_path_manager_requires_dataset_name(tmp_path):
    """Test that PathManager raises error when dataset_name is None."""
    with pytest.raises(ValueError, match="dataset_name cannot be None"):
        PathManager(project_root=tmp_path, dataset_name=None)


def test_path_manager_properties(tmp_path):
    """Test PathManager directory properties."""
    dataset_name = "test_dataset"
    pm = PathManager(project_root=tmp_path, dataset_name=dataset_name)

    assert pm.project_data_root == tmp_path / "benchmarks" / "EIDBench-synth"
    assert pm.processed_data_root == tmp_path / "data_processed"
    assert pm.scripts_root == tmp_path / "benchmarks" / "EIDBench-synth" / dataset_name / "scripts"
    assert pm.annotations_root == tmp_path / "benchmarks" / "EIDBench-synth" / dataset_name / "annotations"
    assert pm.files_root == tmp_path / "benchmarks" / "EIDBench-synth" / dataset_name / "files"
    assert pm.errors_root == tmp_path / "benchmarks" / "EIDBench-synth" / dataset_name / "errors"


def test_path_manager_uses_default_project_root():
    """Test PathManager uses get_project_root() when project_root is None."""
    dataset_name = "test_dataset"
    pm = PathManager(project_root=None, dataset_name=dataset_name)

    # Should not raise an error and should have a valid project_root
    assert pm.project_root is not None
    assert isinstance(pm.project_root, Path)
