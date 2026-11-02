"""Per-dataset breakdown of tab:combined-results for EIDBench-synth and EIDBench-real (tab:per-dataset-results): one
row per method, one column per dataset, the synth datasets first, then the real ones. A cell is the macro metric of
that dataset, mean and sample SD over the replicates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from workflow_prismadv.results import methods
from workflow_prismadv.results.benchmarks import real, synth
from workflow_prismadv.results.metrics import METRICS, dataset_summary, summary

DATASETS = {"synth": synth.DATASETS, "real": real.DATASETS}
MEAN = "mean"
# Column labels in the paper header.
LABELS = {"IPL_win_prediction": "IPL", "healthy_diet_dashboard": "nutrition", "bank_marketing_analysis": "marketing",
          "omop_cdm_databricks": "healthcare"}


def cell(row: pd.Series | None, metric: str, rank: int | None, wrap: str | None = None) -> str:
    if row is None:
        return "-"
    v = f"{100 * row[f'{metric}_mean']:.1f}"
    if rank == 0:
        v = rf"\textbf{{{v}}}"
    elif rank == 1:
        v = rf"\underline{{{v}}}"
    if row["reps"] >= 2:
        v += rf"{{\scriptsize$\pm${100 * row[f'{metric}_sd']:.1f}}}"
    return rf"\{wrap}{{{v}}}" if wrap else v


def columns(ds: pd.DataFrame, s: pd.DataFrame, benches: list[str], mean: bool) -> dict[tuple, pd.DataFrame]:
    """{(benchmark, dataset or MEAN): rows indexed by arm}, the benchmarks in the given order."""
    cols = {}
    for b in benches:
        if b not in set(s.index.get_level_values(0)):
            continue
        present = set(ds.loc[b].index.get_level_values("dataset"))
        cols.update({(b, d): ds.loc[b].xs(d, level="dataset") for d in DATASETS[b] if d in present})
        if mean:
            cols[(b, MEAN)] = s.loc[b]
    return cols


def latex_rows(ds: pd.DataFrame, s: pd.DataFrame, benches: list[str], keys: list[str], metric: str,
               mean: bool = False, wrap: str | None = None) -> list[str]:
    cols = columns(ds, s, benches, mean)
    ranks = {c: {k: i for i, k in enumerate(t[f"{metric}_mean"].sort_values(ascending=False).index[:2])}
             for c, t in cols.items()}
    by_key = {r.key: r for r in methods.ROWS}
    out = ["% columns: " + ", ".join(f"{b}:{LABELS.get(d, d)}" for b, d in cols)]
    for gi, group in enumerate(g for g in methods.GROUPS if any(k in keys for k in g)):
        if gi:
            out.append(r"\midrule")
        for key in (k for k in group if k in keys):
            cells = [cell(t.loc[key] if key in t.index else None, metric, ranks[c].get(key), wrap)
                     for c, t in cols.items()]
            out.append(rf"\texttt{{{by_key[key].name}}} & " + " & ".join(cells) + r" \\")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", action="append", choices=sorted(DATASETS))
    ap.add_argument("--family", action="append", choices=sorted(methods.BY_FAMILY))
    ap.add_argument("--metric", choices=METRICS, default="f1")
    ap.add_argument("--mean", action="store_true", help="add the mean over the datasets of each benchmark")
    ap.add_argument("--wrap", help="wrap every cell in \\WRAP{...}, e.g. rnone")
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--summary", type=Path)
    a = ap.parse_args(argv)
    benches = [b for b in DATASETS if b in a.benchmark] if a.benchmark else list(DATASETS)
    families = [methods.BY_FAMILY[f] for f in a.family] if a.family else methods.FAMILIES
    keys = [r.key for f in families for r in f.ROWS]
    if a.cache and a.cache.exists():
        outcomes = pd.read_csv(a.cache, sep="\t")
        outcomes = outcomes[outcomes.arm.isin(keys) & outcomes.benchmark.isin(benches)]
        methods.check_complete(outcomes)
    else:
        outcomes = methods.outcomes(families, benches)
        if a.cache:
            outcomes.to_csv(a.cache, sep="\t", index=False)
    ds, s = dataset_summary(outcomes), summary(outcomes)
    if a.summary:
        ds.to_csv(a.summary, sep="\t")
    print(f"% macro {a.metric} per dataset")
    print("\n".join(latex_rows(ds, s, benches, keys, a.metric, a.mean, a.wrap)))
    for b in benches:
        if b in set(s.index.get_level_values(0)):
            print(f"% {b}: " + ", ".join(f"{k} {v} rep(s)" for k, v in s.loc[b].reps.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
