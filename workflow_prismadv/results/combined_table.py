"""Rows of tab:combined-results and its two appendix variants: precision, recall and F1 per method on ICDBench,
EIDBench-synth and EIDBench-real (workflow_prismadv/results/__init__.py says how they are computed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from workflow_prismadv.results import methods
from workflow_prismadv.results.metrics import KINDS, METRICS, ci_half_width, summary

INTERVALS = ("sd", "ci95")


def number(v: float) -> tuple[str, str]:
    """(padding, digits) of a percentage, padded to the width of 100.0."""
    digits = f"{v:.1f}"
    return r"\phantom{" + "0" * (5 - len(digits)) + "}" if len(digits) < 5 else "", digits


def cell(s: pd.DataFrame, bench: str, key: str, metric: str, ranks: dict, interval: str = "sd",
         wrap: str | None = None) -> str:
    if (bench, key) not in s.index:
        return "-"
    row = s.loc[(bench, key)]
    pad, digits = number(100 * row[f"{metric}_mean"])
    if metric == "f1" and key == ranks[bench][0]:
        v = rf"\textbf{{{pad}{digits}}}"
    elif metric == "f1" and key == ranks[bench][1]:
        v = rf"{pad}\underline{{{digits}}}"
    else:
        v = f"{pad}{digits}"
    if row["reps"] >= 2:
        sd = row[f"{metric}_sd"]
        width = sd if interval == "sd" else ci_half_width(sd, int(row["reps"]))
        v += rf"{{\scriptsize$\pm${100 * width:.1f}}}"
    return rf"\{wrap}{{{v}}}" if wrap else v


def latex_rows(s: pd.DataFrame, keys: list[str], interval: str = "sd", wrap: str | None = None) -> list[str]:
    benches = set(s.index.get_level_values(0))
    ranks = {b: (s.loc[b].f1_mean.sort_values(ascending=False).index.tolist() + [None, None]) if b in benches
             else [None, None] for b in methods.BENCHMARKS}
    by_key = {r.key: r for r in methods.ROWS}
    mark = lambda x: r"\checkmark" if x else "-"
    out = []
    for gi, group in enumerate(g for g in methods.GROUPS if any(k in keys for k in g)):
        if gi:
            out.append(r"\midrule")
        for key in (k for k in group if k in keys):
            r = by_key[key]
            cells = [cell(s, b, key, m, ranks, interval, wrap) for b in methods.BENCHMARKS for m in METRICS]
            out.append(rf"\texttt{{{r.name}}} & {mark(r.task_aware)} & {mark(r.generates_test)} & "
                       + " & ".join(cells) + r" \\")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=KINDS, default="macro")
    ap.add_argument("--interval", choices=INTERVALS, default="sd")
    ap.add_argument("--wrap", help="wrap every cell in \\WRAP{...}, e.g. rnone")
    ap.add_argument("--family", action="append", choices=sorted(methods.BY_FAMILY))
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--summary", type=Path)
    a = ap.parse_args(argv)
    families = [methods.BY_FAMILY[f] for f in a.family] if a.family else methods.FAMILIES
    keys = [r.key for f in families for r in f.ROWS]
    if a.cache and a.cache.exists():
        outcomes = pd.read_csv(a.cache, sep="\t")
        outcomes = outcomes[outcomes.arm.isin(keys)]
        methods.check_complete(outcomes)
    else:
        outcomes = methods.outcomes(families)
        if a.cache:
            outcomes.to_csv(a.cache, sep="\t", index=False)
    s = summary(outcomes, a.metric)
    if a.summary:
        s.to_csv(a.summary, sep="\t")
    print("\n".join(latex_rows(s, keys, a.interval, a.wrap)))
    for b in methods.BENCHMARKS:
        if b in set(s.index.get_level_values(0)):
            reps = s.loc[b].reps
            print(f"% {b}: " + ", ".join(f"{k} {v} rep(s)" for k, v in reps.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
