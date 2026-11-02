"""PrismaDV ablation on EIDBench-real (tab:ablation-real), gpt-6-sol (medium), replicates r1-r3.

  full                         prismadv--<model>
  w/o correlation discovery    prismadv--<model>, without its multi-column constraints (a key "[...]" or code that
                               names two or more columns of the table)
  w/o dataflow analysis        prismadv_wo_dataflow--<model>
  w/o assumption inference     prismadv_wo_assumption--<model>
  w/o data-code graph          prismadv_wo_dataflow_wo_assumption--<model>
"""
from __future__ import annotations

import re

import pandas as pd

from workflow_prismadv.results.alarm import NOT_RUN, load
from workflow_prismadv.results.benchmarks import real
from workflow_prismadv.results.methods._common import REPLICATES, frame, real_arms

MODEL = "gpt-6-sol-medium"
ROWS = {"full": "prismaDV [gpt-6]", "wo-corr": "w/o correlation discovery", "wo-flow": "w/o dataflow analysis",
        "wo-assume": "w/o assumption inference", "wo-graph": "w/o data-code assumption graph"}
RUNS = {"full": "prismadv", "wo-flow": "prismadv_wo_dataflow", "wo-assume": "prismadv_wo_assumption",
        "wo-graph": "prismadv_wo_dataflow_wo_assumption"}


def table_columns(ds: str) -> dict[str, list[str]]:
    profiles = load(real.base(ds) / "constraints/project/prismadv_inputs.yaml")["table_profiles"]
    return {t: [c["name"] for c in p.get("columns", [])] for t, p in profiles.items()}


def single_column(key: str, code: str, columns: list[str]) -> bool:
    if key.lstrip().startswith("["):
        return False
    return sum(1 for c in columns if re.search(rf"[`'\"]{re.escape(c)}[`'\"]", code)) < 2


def alarm_single_column(doc: dict, columns: dict[str, list[str]]) -> bool:
    return any(e.get("status") is False and e.get("level", "error") == "error"
               and NOT_RUN not in (e.get("reason_if_failed") or "")
               and single_column(key, e.get("suggestion") or "", columns.get(table, []))
               for table, cols in (doc.get("tables") or {}).items() for key, body in (cols or {}).items()
               for e in (body or {}).get("code") or [])


def without_correlation() -> pd.DataFrame:
    rows = []
    for ds in real.DATASETS:
        columns = table_columns(ds)
        for r in REPLICATES:
            art = real.artifact(ds, "prismadv", MODEL, r)
            for n, safe in real.units(ds):
                f = real.result(ds, n, art.stem)
                rows.append((real.NAME, ds, n, "wo-corr", r, safe, alarm_single_column(load(f), columns)))
    return frame(rows)


def outcomes() -> pd.DataFrame:
    runs = real_arms({k: (p, MODEL) for k, p in RUNS.items()})
    return pd.concat([runs, without_correlation()], ignore_index=True)
