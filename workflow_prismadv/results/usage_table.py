"""Token usage of the LLM-based methods (tab:token-usage): input tokens, output tokens without the reasoning part, and
reasoning tokens.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from workflow_prismadv.results import methods, usage

# (header, summary field) of the token columns; "cached" is the cache share
COLUMNS = [("Input", "input"), ("Output", "output_text"), ("Reasoning", "reasoning")]
CACHED = ("Cached", "cached")


def k(row: pd.Series, field: str) -> tuple[str, str]:
    """(thousands, +- SD) of a token field."""
    sd = "" if row["reps"] < 2 else rf"{{\scriptsize$\pm${row[f'{field}_sd'] / 1000:.1f}}}"
    return f"{row[f'{field}_mean'] / 1000:.1f}", sd


def latex_rows(s: pd.DataFrame, benches: list[str], cached: bool = False) -> list[str]:
    names = {r.key: r.name for r in methods.ROWS}
    keys = [key for group in methods.GROUPS for key in group if key in usage.ARMS]
    cols = [(b, c) for b in benches for c in (COLUMNS[:1] + [CACHED] + COLUMNS[1:] if cached else COLUMNS)]
    raw = {key: [(rf"{100 * s.loc[(b, key), 'cached_share']:.0f}\%", "") if f == "cached" else k(s.loc[(b, key)], f)
                 if (b, key) in s.index else ("-", "") for b, (_, f) in cols] for key in keys}
    width = [max(len(raw[key][i][0]) for key in keys) for i in range(len(cols))]
    pad = lambda v, w: (r"\phantom{" + "0" * (w - len(v)) + "}" if len(v) < w else "") + v
    out = ["% columns: " + ", ".join(f"{b}:{h}" for b, (h, _) in cols)]
    groups = [[key for key in g if key in keys] for g in methods.GROUPS]
    for gi, group in enumerate(g for g in groups if g):
        if gi:
            out.append(r"\midrule")
        for key in group:
            cells = [v if v.endswith(r"\%") or v == "-" else pad(v, w) + sd for (v, sd), w in zip(raw[key], width)]
            out.append(rf"\texttt{{{names[key]}}} & " + " & ".join(cells) + r" \\")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", action="append", choices=usage.BENCHMARKS)
    ap.add_argument("--cached", action="store_true", help="add the prompt-cache share of the input tokens")
    ap.add_argument("--records", type=Path)
    ap.add_argument("--summary", type=Path)
    a = ap.parse_args(argv)
    benches = [b for b in usage.BENCHMARKS if b in a.benchmark] if a.benchmark else ["synth"]
    rec = usage.records(benches)
    s = usage.summary(rec)
    if a.records:
        rec.to_csv(a.records, index=False)
    if a.summary:
        s.to_csv(a.summary)
    print("\n".join(latex_rows(s, benches, a.cached)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
