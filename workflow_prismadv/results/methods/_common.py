"""Readers shared by the method families. A family module states its rows and where their files are; these
functions turn that into outcome rows (workflow_prismadv.results.benchmarks.COLUMNS)."""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import pandas as pd

from workflow_prismadv.results.alarm import alarm, load
from workflow_prismadv.results.benchmarks import COLUMNS, icd, real, synth
from workflow_prismadv.utils.replicates import replicate_of_artifact

REPLICATES = (1, 2, 3)


@dataclass(frozen=True)
class Row:
    key: str
    name: str               # \texttt{...} label in tab:combined-results
    task_aware: bool
    generates_test: bool


def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------- EIDBench-synth

def _synth_per_task_dataset(args):
    ds, globs = args
    stems = {}                                  # (row key, task) -> {replicate: artifact stem}
    for t in synth.tasks(ds):
        for key, glob in globs.items():
            reps = {}
            for p in sorted((synth.general(ds) / "constraints" / t).glob(f"{glob}.yaml")):
                r = replicate_of_artifact(p, load(p))
                if r in REPLICATES:
                    reps[r] = p.stem            # sorted by name, so the newest timestamp wins
            stems[(key, t)] = reps
    rows = []
    for n, t, safe in _units(ds):
        for key in globs:
            for r, stem in stems[(key, t)].items():
                f = synth.task_result(ds, n, t, stem)
                if f.exists():
                    rows.append((synth.NAME, ds, f"{n}/{t}", key, r, safe, alarm(load(f))))
    return rows


_UNITS = {}


def _units(ds):
    if ds not in _UNITS:
        _UNITS[ds] = synth.units(ds)
    return _UNITS[ds]


def synth_per_task(globs: dict[str, str]) -> pd.DataFrame:
    """Methods with one constraint artifact per task: {row key: artifact glob in general/constraints/<task>/}."""
    with Pool(len(synth.DATASETS)) as pool:
        parts = pool.map(_synth_per_task_dataset, [(ds, globs) for ds in synth.DATASETS])
    return frame([r for p in parts for r in p])


def synth_per_label(stems: dict[str, str]) -> pd.DataFrame:
    """Task-agnostic methods with one result per label: {row key: result stem}; the verdict counts for every task."""
    rows = []
    for ds in synth.DATASETS:
        verdicts = {}
        for n, t, safe in _units(ds):
            for key, stem in stems.items():
                if (n, key) not in verdicts:
                    f = synth.label_result(ds, n, stem)
                    verdicts[(n, key)] = alarm(load(f)) if f.exists() else None
                if verdicts[(n, key)] is not None:
                    rows.append((synth.NAME, ds, f"{n}/{t}", key, 1, safe, verdicts[(n, key)]))
    return frame(rows)


# ---------------------------------------------------------------- EIDBench-real

def real_arms(arms: dict[str, tuple[str, str]], reps=REPLICATES) -> pd.DataFrame:
    """{row key: (artifact prefix, model)}. A label without a result gives no row; methods.check_complete raises
    on the gap."""
    rows = []
    for ds in real.DATASETS:
        units = real.units(ds)
        for key, (prefix, model) in arms.items():
            for r in reps:
                art = real.artifact(ds, prefix, model, r)
                if art is None:
                    continue
                for n, safe in units:
                    f = real.result(ds, n, art.stem)
                    if f.exists():
                        rows.append((real.NAME, ds, n, key, r, safe, alarm(load(f))))
    return frame(rows)


# ---------------------------------------------------------------- ICDBench

def icd_arms(stems: dict[str, str], reps=REPLICATES) -> pd.DataFrame:
    """{row key: outcome-table stem}; a missing table means no outcomes for that replicate."""
    rows = []
    for key, stem in stems.items():
        for r in reps:
            f = icd.outcome_table(stem, r)
            if f.exists():
                rows += icd.rows(f, key, r)
    return frame(rows)
