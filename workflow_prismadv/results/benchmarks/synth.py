"""EIDBench-synth: a unit is a (label, task) pair; it is safe when the task runs correctly on the corrupted batch.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from prismadv.utils import get_project_root

NAME = "synth"
DATASETS = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]


def general(ds: str, root: Path | None = None) -> Path:
    return (root or get_project_root()) / "data_processed" / ds / "general"


def tasks(ds: str, root: Path | None = None) -> list[str]:
    root = root or get_project_root()
    return sorted(p.stem for p in (root / "benchmarks/EIDBench-synth" / ds / "scripts/general").glob("*.py"))


def labels(ds: str, root: Path | None = None) -> list[str]:
    root = root or get_project_root()
    out = []
    for p in (root / "benchmarks/EIDBench-synth" / ds / "errors").glob("error-group-*/general_task_*.yaml"):
        k = re.search(r"error-group-(\d+)", str(p)).group(1)
        out.append(f"{int(k)}-{int(p.stem.split('_')[-1])}")
    return sorted(out, key=lambda n: tuple(map(int, n.split("-"))))


def units(ds: str, root: Path | None = None) -> list[tuple[str, str, bool]]:
    """(label, task, safe) for every unit of the dataset."""
    out = []
    for n in labels(ds, root):
        for t in tasks(ds, root):
            gt = general(ds, root) / n / "output_validation" / t / "basic_metrics_evaluation.json"
            if gt.exists():
                out.append((n, t, bool(json.loads(gt.read_text())["corrupted_data_is_safe"])))
    return out


def task_result(ds: str, label: str, task: str, stem: str, root: Path | None = None) -> Path:
    """Result of a per-task constraint artifact <stem> on the label's corrupted batch."""
    return (general(ds, root) / label / "constraints_validation" / task /
            f"validation_results_on_corrupted_test_data__{stem}.yaml")


def label_result(ds: str, label: str, stem: str, root: Path | None = None) -> Path:
    """Result of a task-agnostic method, one per label (it counts for every task of the label)."""
    return general(ds, root) / label / "constraints_validation" / f"validation_results_on_corrupted_test_data__{stem}.yaml"
