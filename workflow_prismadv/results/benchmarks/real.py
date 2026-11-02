"""EIDBench-real: a unit is a label (one corrupted bundle of an application); its ground truth is the hand label
expected_current_outcome in benchmarks/EIDBench-real/<app>/errors/error-group-<k>/project_<n>.yaml (label <k>-<n>).
"""
from __future__ import annotations

import re
from pathlib import Path

from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_real_experiments.utils import config_label, config_paths_in
from workflow_prismadv.results.alarm import load
from workflow_prismadv.utils.replicates import parse_replicate

NAME = "real"
DATASETS = ["healthy_diet_dashboard", "bank_marketing_analysis", "omop_cdm_databricks"]


def base(ds: str, root: Path | None = None) -> Path:
    return (root or get_project_root()) / "data_processed/eid_bench_real" / ds


def units(ds: str, root: Path | None = None) -> list[tuple[str, bool]]:
    """(label, safe) for every label of the application."""
    root = root or get_project_root()
    paths = config_paths_in(root / "benchmarks/EIDBench-real" / ds / "errors")
    return [(config_label(p), load(p)["expected_current_outcome"] == "safe") for p in paths]


def artifact(ds: str, prefix: str, model: str, rep: int, root: Path | None = None) -> Path | None:
    project = base(ds, root) / "constraints/project"
    pattern = rf"{re.escape(prefix)}--{re.escape(model)}(--r\d+)?(--\d{{8}}_\d{{6}})?"
    found = [p for p in project.glob(f"{prefix}--{model}*.yaml")
             if re.fullmatch(pattern, p.stem) and parse_replicate(p.stem) == rep]
    return max(found, key=lambda p: p.stem) if found else None


def result(ds: str, label: str, stem: str, root: Path | None = None) -> Path:
    return base(ds, root) / "constraints_validation/project/corrupted" / label / f"validation_results__{stem}.yaml"
