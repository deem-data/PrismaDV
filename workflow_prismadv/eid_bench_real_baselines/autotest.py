#!/usr/bin/env python3
"""AutoTest task-agnostic baseline for EIDBench-real."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import oyaml as yaml
import pandas as pd

from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils
from workflow_prismadv.eid_bench_baselines.pipelines.run_autotest import (
    AUTOTEST_STAGING,
    autotest_detect_csvs,
    autotest_table_timings,
)

PROJECT_SCRIPT_ID = "project"
BASELINE_METHOD = "autotest"
MODEL_TAG = "autotest"
DISTINCT_CAP = int(os.environ.get("AUTOTEST_DISTINCT_CAP", "5000"))
# 1000 rows for every app: full tables are intractable (omop observations has 489k rows), and
# AutoTest's SDCs were mined on columns with at most 1000 distinct values.
ROW_CAP = int(os.environ.get("AUTOTEST_ROW_CAP", "1000"))
SDC_NAME = "rt_train"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example-id", required=True)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--distinct-cap", type=int, default=DISTINCT_CAP)
    p.add_argument("--row-cap", type=int, default=ROW_CAP)
    p.add_argument("--label", action="append", default=None,
                   help="Limit to specific corruption label(s). Default: all.")
    p.add_argument("--run-name", default=None,
                   help="Staging directory under AUTOTEST_STAGING; jobs running at the same time need different "
                        "names (default: eid_real_<example-id>). A resubmit with the same name resumes.")
    p.add_argument("--timestamp", default=None,
                   help="Artifact timestamp (YYYYmmdd_HHMMSS); give the same one to every job of one app so "
                        "all labels share one artifact stem (default: now).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def corrupted_labels(example_path: Path, only=None):
    labels = [utils.config_label(p) for p in utils.config_paths_in(example_path / "errors")]
    if only:
        labels = [l for l in labels if l in set(only)]
    return labels


def prepare_table_csv(df: pd.DataFrame, out_path: Path, distinct_cap: int, row_cap: int):
    """Write a table as a CSV for AutoTest."""
    keep = [c for c in df.columns if df[c].astype(str).nunique(dropna=True) <= distinct_cap]
    out = df[keep]
    if row_cap and len(out) > row_cap:
        out = out.sample(n=row_cap, random_state=0)
    out.to_csv(out_path, index=False)
    return keep, len(out)


def main(argv=None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    pm = MultiTableProjectManager(project_root=project_root, example_id=args.example_id)
    table_names = pm.get_table_names()
    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{BASELINE_METHOD}--{MODEL_TAG}--{ts}"
    run_name = args.run_name or f"eid_real_{args.example_id}"

    constraints_dir = utils.constraints_dir(args.example_id, PROJECT_SCRIPT_ID, project_root, create=True)
    artifact_path = constraints_dir / f"{stem}.yaml"
    write = not (artifact_path.exists() and not args.overwrite)
    if write:
        yaml.safe_dump({
            "baseline_method": BASELINE_METHOD,
            "model_name": MODEL_TAG,
            "reference": "Chen et al., Auto-Test, SIGMOD 2025 (arXiv:2504.10762)",
            "sdc_set": SDC_NAME,
            "distinct_cap": args.distinct_cap,
            "row_cap": args.row_cap,
            "note": ("Task- and dataset-agnostic SDCs applied directly to corrupted "
                     "bundles; validation_results written by this script (AutoTest does "
                     "not emit executable constraints for the shared Deequ validator)."),
        }, artifact_path.open("w"), sort_keys=False)
    print(f"artifact: {artifact_path}")

    labels = corrupted_labels(pm.example_path, only=args.label)
    print(f"example_id={args.example_id} tables={table_names} labels={len(labels)}")

    # A corrupted bundle copies its untouched tables from the clean data, so most (label, table) pairs repeat a
    # table seen before. Detection is deterministic (and the row sample is seeded), so each distinct prepared
    # table goes to AutoTest once, keyed by its content hash, and every label that holds it shares the result.
    inputs = AUTOTEST_STAGING / run_name / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    id_to_csv = {}
    label_tables = {}  # label -> [(table, csv_id, n columns kept)]
    with tempfile.TemporaryDirectory() as tmp:
        for label in labels:
            for table in table_names:
                tpath = pm.get_table_path(table, "corrupted", corruption_label=label)
                if not Path(tpath).exists():
                    continue
                df = pd.read_csv(tpath, dtype=str)
                tmp_csv = Path(tmp) / "table.csv"
                kept, _ = prepare_table_csv(df, tmp_csv, args.distinct_cap, args.row_cap)
                csv_id = f"{table}__{hashlib.md5(tmp_csv.read_bytes()).hexdigest()[:16]}"
                out_csv = inputs / f"{csv_id}.csv"
                if not out_csv.exists():
                    shutil.move(tmp_csv, out_csv)
                id_to_csv[csv_id] = out_csv
                label_tables.setdefault(label, []).append((table, csv_id, len(kept)))
    print(f"{sum(len(v) for v in label_tables.values())} (label, table) pairs -> {len(id_to_csv)} distinct tables")

    detections = autotest_detect_csvs(id_to_csv, sdc_name=SDC_NAME, run_name=run_name, skip_missing=True)
    timings = autotest_table_timings(id_to_csv, run_name)

    for label in labels:
        entries = label_tables.get(label, [])
        missing = [t for t, i, _ in entries if i not in detections]
        if missing:
            # No result rather than a silent pass: an empty detection would read as "nothing flagged".
            print(f"  {label}: AutoTest failed on table(s) {missing}; no result written")
            continue
        failed_error = sum(len(detections[i]) for _, i, _ in entries)
        total_cols = sum(n for _, _, n in entries)
        out_dir = utils.constraints_validation_dir(
            args.example_id, PROJECT_SCRIPT_ID, "corrupted", project_root,
            corruption_label=label, create=True)
        out_path = out_dir / f"validation_results__{stem}.yaml"
        result = {
            "example_id": args.example_id,
            "script_id": PROJECT_SCRIPT_ID,
            "variant": "corrupted",
            "corruption_label": label,
            "method": BASELINE_METHOD,
            "summary": {
                "passed_warning": 0,
                "failed_warning": 0,
                "passed_error": max(total_cols - failed_error, 0),
                "failed_error": failed_error,
                "non_compilable": 0,
                "total": total_cols,
            },
            "flagged_by_table": {t: len(detections[i]) for t, i, _ in entries},
            # Detection time of each table's distinct content; a table shared with other labels was timed once.
            "table_ids": {t: i for t, i, _ in entries},
            "validation_seconds_by_table": {t: timings.get(i, {}).get("elapsed_s") for t, i, _ in entries},
        }
        yaml.safe_dump(result, out_path.open("w"), sort_keys=False)
        print(f"  {label}: failed_error={failed_error} (cols checked={total_cols}) -> {out_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
