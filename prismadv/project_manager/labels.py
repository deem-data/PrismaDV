from __future__ import annotations

import re
from pathlib import Path

CLEAN_LABEL = "0"
_CORRUPTED_LABEL = re.compile(r"^(\d+)-(\d+)$")
_GROUP_DIR = re.compile(r"^error-group-(\d+)$")
_CONFIG_FILE = re.compile(r"^(?P<subtask>.+)_(?P<n>\d+)\.ya?ml$")


def is_clean_label(processed_data_label: str | int) -> bool:
    """True for the uncorrupted baseline label."""
    return str(processed_data_label) == CLEAN_LABEL


def is_label(name: str) -> bool:
    """True for a folder name that is a processed-data label (`0` or `<k>-<n>`)."""
    return is_clean_label(name) or _CORRUPTED_LABEL.match(name) is not None


def label_sort_key(processed_data_label: str) -> tuple[int, int]:
    """Clean label first, then by group and by number in the group."""
    if is_clean_label(processed_data_label):
        return (0, 0)
    match = _CORRUPTED_LABEL.match(processed_data_label)
    if match is None:
        raise ValueError(f"'{processed_data_label}' is not a processed-data label (expected 0 or <group>-<n>)")
    return (int(match.group(1)), int(match.group(2)))


def config_path(errors_root: Path, subtask_name: str, processed_data_label: str) -> Path:
    """`errors/error-group-<k>/<subtask>_<n>.yaml` of a corrupted label."""
    if is_clean_label(processed_data_label):
        raise ValueError("the clean label has no error config")
    k, n = label_sort_key(processed_data_label)
    return errors_root / f"error-group-{k}" / f"{subtask_name}_{n}.yaml"


def config_labels(errors_root: Path, subtask_name: str) -> list[str]:
    """Every corrupted label that has a config for this subtask, in label order."""
    labels = []
    for group_dir in errors_root.iterdir():
        group = _GROUP_DIR.match(group_dir.name)
        if not group_dir.is_dir() or group is None:
            continue
        for config in group_dir.iterdir():
            match = _CONFIG_FILE.match(config.name)
            if match is not None and match.group("subtask") == subtask_name:
                labels.append(f"{int(group.group(1))}-{int(match.group('n'))}")
    return sorted(labels, key=label_sort_key)


def corruption_variants(processed_data_label: str | int) -> list[bool]:
    """The `clean` flags a pipeline should run for this label.

    Every corrupted label runs both branches, because a row compares the
    pipeline's behaviour on clean input against its behaviour on corrupted
    input. The clean label has no corrupted input to run against, so it runs the
    clean branch alone.
    """
    return [True] if is_clean_label(processed_data_label) else [True, False]
