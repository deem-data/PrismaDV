"""Shared helpers for EIDBench-real workflow scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import oyaml as yaml

from prismadv.project_manager import MultiTableProjectManager, label_sort_key
from prismadv.utils import get_project_root


REAL_ETL_PROCESSED_DIR = "eid_bench_real"
VALID_VARIANTS = {"observed", "clean", "corrupted"}
_GROUP_DIR = re.compile(r"^error-group-(\d+)$")
_CONFIG_FILE = re.compile(r"^project_(\d+)\.yaml$")


def resolve_project_root(project_root: Path | str | None = None) -> Path:
    return get_project_root() if project_root is None else Path(project_root)


def eid_bench_real_processed_root(project_root: Path | str | None = None) -> Path:
    return resolve_project_root(project_root) / "data_processed" / REAL_ETL_PROCESSED_DIR


def example_processed_root(example_id: str, project_root: Path | str | None = None) -> Path:
    return eid_bench_real_processed_root(project_root) / example_id


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def variant_component(variant: str, corruption_label: str | None = None) -> Path:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VALID_VARIANTS)}")
    if variant == "corrupted":
        if not corruption_label:
            raise ValueError("corruption_label is required for corrupted artifacts")
        return Path("corrupted") / corruption_label
    if corruption_label is not None:
        raise ValueError("corruption_label is only valid for corrupted artifacts")
    return Path(variant)


def constraints_dir(
    example_id: str,
    script_id: str,
    project_root: Path | str | None = None,
    *,
    create: bool = False,
) -> Path:
    path = example_processed_root(example_id, project_root) / "constraints" / script_id
    return ensure_dir(path) if create else path


def constraint_artifact_path(
    example_id: str,
    script_id: str,
    filename: str,
    project_root: Path | str | None = None,
) -> Path:
    return constraints_dir(example_id, script_id, project_root) / filename


def constraints_validation_dir(
    example_id: str,
    script_id: str,
    variant: str,
    project_root: Path | str | None = None,
    *,
    corruption_label: str | None = None,
    create: bool = False,
) -> Path:
    path = (
        example_processed_root(example_id, project_root)
        / "constraints_validation"
        / script_id
        / variant_component(variant, corruption_label)
    )
    return ensure_dir(path) if create else path


def execution_dir(
    example_id: str,
    script_id: str,
    variant: str,
    project_root: Path | str | None = None,
    *,
    corruption_label: str | None = None,
    create: bool = False,
) -> Path:
    path = (
        example_processed_root(example_id, project_root)
        / "execution"
        / script_id
        / variant_component(variant, corruption_label)
    )
    return ensure_dir(path) if create else path


def outcomes_dir(
    example_id: str,
    project_root: Path | str | None = None,
    *,
    create: bool = False,
) -> Path:
    path = example_processed_root(example_id, project_root) / "outcomes"
    return ensure_dir(path) if create else path


def expected_outcomes_path(example_id: str, project_root: Path | str | None = None) -> Path:
    return outcomes_dir(example_id, project_root) / "expected_outcomes.yaml"


def error_config_dir(project_manager: MultiTableProjectManager) -> Path:
    return project_manager.example_path / "errors"


def config_label(path: Path) -> str:
    """`errors/error-group-<k>/project_<n>.yaml` -> label `<k>-<n>`, as in EIDBench-synth.

    The group `k` is the error type of the config (the same group numbers as EIDBench-synth) and `n` counts from 1
    in each group.
    """
    group = _GROUP_DIR.match(path.parent.name)
    number = _CONFIG_FILE.match(path.name)
    if group is None or number is None:
        raise ValueError(f"EIDBench-real error config must be errors/error-group-<k>/project_<n>.yaml: {path}")
    return f"{int(group.group(1))}-{int(number.group(1))}"


def config_paths_in(config_dir: Path) -> list[Path]:
    """Every error config under `config_dir`, in label order."""
    if not config_dir.exists():
        return []
    return sorted(config_dir.glob("error-group-*/*.yaml"), key=lambda path: label_sort_key(config_label(path)))


def error_config_paths(project_manager: MultiTableProjectManager) -> list[Path]:
    return config_paths_in(error_config_dir(project_manager))


def load_error_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"EIDBench-real error config must be a mapping: {path}")
    config = dict(config)
    config["label"] = config_label(path)
    return config


def load_error_configs(project_manager: MultiTableProjectManager) -> dict[str, dict[str, Any]]:
    configs = {}
    for path in error_config_paths(project_manager):
        config = load_error_config(path)
        label = config["label"]
        if label in configs:
            raise ValueError(f"duplicate EIDBench-real corruption label <{label}> in {error_config_dir(project_manager)}")
        configs[label] = config
    return configs


def corruption_labels(project_manager: MultiTableProjectManager) -> list[str]:
    return sorted(load_error_configs(project_manager), key=label_sort_key)
