"""Optimization cost of GEPA[P], GEPA[S] and SIFTA"""
from __future__ import annotations

import json

import oyaml as yaml
import pandas as pd

from prismadv.utils import get_project_root
from workflow_sifta._0_splits import DATASETS, SPLIT_SEEDS, load_split, test_units
from workflow_sifta._4_scoring import LLM, newest_artifact

PROGRAMS = {"gepa_column": "GEPA[P]", "gepa_script": "GEPA[S]", "sifta": "SIFTA"}
FIELDS = ["input", "output", "reasoning", "calls", "usd"]


def run_dir(ds: str, split: int, program: str) -> str:
    dirs = set()
    for scripts, _ in test_units(load_split(ds, split)).values():
        for t in scripts:
            a = newest_artifact(ds, t, program, LLM, split)
            if a is not None:
                dirs.add(yaml.safe_load(a.read_text())["sifta_generation"]["run_dir"])
    if len(dirs) != 1:
        raise ValueError(f"{program} {ds} split {split}: runs {sorted(dirs)}")
    return dirs.pop()


def records() -> pd.DataFrame:
    rows = []
    for program in PROGRAMS:
        for ds in DATASETS:
            for s in SPLIT_SEEDS:
                d = run_dir(ds, s, program)
                u = json.loads((get_project_root() / d / "summary.json").read_text())["llm_usage"]["total"]
                rows.append(dict(program=program, dataset=ds, split=s, run=d, input=u["prompt_tokens"],
                                 output=u["completion_tokens"] - u["reasoning_tokens"],
                                 reasoning=u["reasoning_tokens"], calls=u["requests"] - u["cache_served_requests"],
                                 usd=u["cost_spent"]))
    return pd.DataFrame(rows)


def summary(rec: pd.DataFrame) -> pd.DataFrame:
    """Mean over the datasets per split, then mean and sample SD over the splits."""
    g = rec.groupby(["program", "split"])[FIELDS].mean().groupby("program")
    return g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
