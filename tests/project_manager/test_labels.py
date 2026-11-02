"""Unit tests for the processed-data label helpers."""

import pytest

from prismadv.project_manager import config_labels, config_path, is_clean_label, is_label, label_sort_key


def test_label_names():
    assert is_clean_label("0") and is_clean_label(0)
    assert is_label("0") and is_label("12-3")
    for name in ("37", "constraints", "12_3", "12-", "-3", "1-2-3"):
        assert not is_label(name)


def test_label_order_is_numeric_by_group_then_n():
    labels = ["10-1", "2-10", "0", "2-2", "1-1"]
    assert sorted(labels, key=label_sort_key) == ["0", "1-1", "2-2", "2-10", "10-1"]
    with pytest.raises(ValueError):
        label_sort_key("37")


def test_config_path_and_listing(tmp_path):
    errors = tmp_path / "errors"
    for k, n in [(1, 1), (1, 2), (12, 3)]:
        (errors / f"error-group-{k}").mkdir(parents=True, exist_ok=True)
        (errors / f"error-group-{k}" / f"general_task_{n}.yaml").write_text("[]")
    (errors / "error-group-1" / "other_task_9.yaml").write_text("[]")  # another subtask
    (errors / "notes").mkdir()

    assert config_path(errors, "general_task", "12-3") == errors / "error-group-12" / "general_task_3.yaml"
    assert config_labels(errors, "general_task") == ["1-1", "1-2", "12-3"]
    with pytest.raises(ValueError):
        config_path(errors, "general_task", "0")
