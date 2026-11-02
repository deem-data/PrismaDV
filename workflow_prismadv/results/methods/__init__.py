"""Method families of tab:combined-results, in table order. Each family module lists its rows (ROWS) and has one
reader per benchmark, synth(), real() and icd(), returning outcome rows; shared readers are in _common.py."""
from __future__ import annotations

import pandas as pd

from workflow_prismadv.results.benchmarks import icd, real, synth
from workflow_prismadv.results.methods import autotest, codex, novelty, prismadv, profilers, prompting

FAMILIES = [novelty, autotest, profilers, prompting, codex, prismadv]
BY_FAMILY = {f.FAMILY: f for f in FAMILIES}
ROWS = [r for f in FAMILIES for r in f.ROWS]
# Rows between two \midrule in the table.
GROUPS = [["ocsvm", "iforest", "stats-nov", "autotest"], ["deequ", "tfdv"],
          ["single-g38", "single-6sol"], ["few-g38", "few-6sol"], ["codex"],
          ["P-g38", "P-6sol"]]
BENCHMARKS = ("icd", "synth", "real")
assert [k for g in GROUPS for k in g] == [r.key for r in ROWS]


def expected_units(benchmark: str) -> dict[str, set[str]]:
    """{dataset: units} of a benchmark, from its ground truth."""
    if benchmark == "synth":
        return {ds: {f"{n}/{t}" for n, t, _ in synth.units(ds)} for ds in synth.DATASETS}
    if benchmark == "real":
        return {ds: {n for n, _ in real.units(ds)} for ds in real.DATASETS}
    return {icd.DATASET: set(icd.units())}


def check_complete(out: pd.DataFrame) -> None:
    """Every (benchmark, arm, rep) with outcomes must cover every unit of every dataset of its benchmark; otherwise
    the per-dataset metric or the mean over datasets would silently use less data."""
    problems = []
    for b in out["benchmark"].unique():
        expected = expected_units(b)
        for (arm, rep), g in out[out["benchmark"] == b].groupby(["arm", "rep"]):
            got = g.groupby("dataset")["unit"].agg(set)
            for ds, units in expected.items():
                missing = units - got.get(ds, set())
                if missing:
                    problems.append(f"{b} {arm} r{rep} {ds}: {len(missing)}/{len(units)} units missing")
    if problems:
        raise ValueError("incomplete outcomes:\n" + "\n".join(problems))


def outcomes(families=FAMILIES, benchmarks=BENCHMARKS) -> pd.DataFrame:
    parts = [getattr(f, b)() for f in families for b in benchmarks]
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    check_complete(out)
    return out
