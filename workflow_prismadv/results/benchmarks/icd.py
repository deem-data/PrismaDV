"""ICDBench: one dataset of 63 cases; each case gives two units, its pass batch (safe; the method alarms when it
does not pass it) and its reject batch (unsafe; the method alarms when it rejects it).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from prismadv.utils import get_project_root

NAME = "icd"
DATASET = "ICDBench"


def units(root: Path | None = None) -> list[str]:
    """Every unit of ICDBench: <case>/pass and <case>/reject for each case directory in data_processed/icd_bench."""
    cases = sorted(p.name for p in ((root or get_project_root()) / "data_processed/icd_bench").iterdir() if p.is_dir())
    return [f"{c}/{kind}" for c in cases for kind in ("pass", "reject")]


def outcome_table(stem: str, rep: int, root: Path | None = None) -> Path:
    return (root or get_project_root()) / "data_processed/icd_bench_outcomes" / f"{stem}--r{rep}.tsv"


def rows(path: Path, arm: str, rep: int) -> list[tuple]:
    df = pd.read_csv(path, sep="\t")
    ok = lambda c: df[c].astype(str) == "True"
    out = []
    for case, passed, rejected in zip(df["case"], ok("passes_pass_batch"), ok("rejects_reject_batch")):
        out.append((NAME, DATASET, f"{case}/pass", arm, rep, True, not passed))
        out.append((NAME, DATASET, f"{case}/reject", arm, rep, False, bool(rejected)))
    return out
