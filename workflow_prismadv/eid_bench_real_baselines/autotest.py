#!/usr/bin/env python3
"""AutoTest task-agnostic baseline for EIDBench-real."""
from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path

import oyaml as yaml
import pandas as pd

from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils
from workflow_prismadv.eid_bench_baselines.pipelines.run_autotest import autotest_detect_csvs

PROJECT_SCRIPT_ID = "project"
BASELINE_METHOD = "autotest"
MODEL_TAG = "autotest"
DISTINCT_CAP = int(os.environ.get("AUTOTEST_DISTINCT_CAP", "5000"))
ROW_CAP = int(os.environ.get("AUTOTEST_ROW_CAP", "5000"))
SDC_NAME = "rt_train"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example-id", required=True)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--distinct-cap", type=int, default=DISTINCT_CAP)
    p.add_argument("--row-cap", type=int, default=ROW_CAP)
    p.add_argument("--label", action="append", default=None,
                   help="Limit to specific corruption label(s). Default: all.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def corrupted_labels(example_path: Path, only=None):
    errors_dir = example_path / "errors"
    labels = sorted(p.stem for p in errors_dir.glob("*.yaml"))
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{BASELINE_METHOD}--{MODEL_TAG}--{ts}"

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
            "note": ("Task- and dataset-agnostic SDCs applied directly to corrupted "
                     "bundles; validation_results written by this script (AutoTest does "
                     "not emit executable constraints for the shared Deequ validator)."),
        }, artifact_path.open("w"), sort_keys=False)
    print(f"artifact: {artifact_path}")

    labels = corrupted_labels(pm.example_path, only=args.label)
    print(f"example_id={args.example_id} tables={table_names} labels={len(labels)}")

    staging = Path(tempfile.mkdtemp(prefix=f"autotest_real_{args.example_id}_"))
    id_to_csv = {}
    id_meta = {}
    for label in labels:
        for table in table_names:
            tpath = pm.get_table_path(table, "corrupted", corruption_label=label)
            if not Path(tpath).exists():
                continue
            df = pd.read_csv(tpath, dtype=str)
            csv_id = f"{label}__{table}"
            out_csv = staging / f"{csv_id}.csv"
            kept, _ = prepare_table_csv(df, out_csv, args.distinct_cap, args.row_cap)
            id_to_csv[csv_id] = out_csv
            id_meta[csv_id] = (label, table, len(kept))

    detections = autotest_detect_csvs(
        id_to_csv, sdc_name=SDC_NAME,
        run_name=f"eid_real_{args.example_id}",
    )

    per_label = {}
    for csv_id, (label, table, ncols) in id_meta.items():
        flagged = len(detections.get(csv_id, []))
        agg = per_label.setdefault(label, [0, 0])
        agg[0] += flagged
        agg[1] += ncols

    for label in labels:
        failed_error, total_cols = per_label.get(label, [0, 0])
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
        }
        yaml.safe_dump(result, out_path.open("w"), sort_keys=False)
        print(f"  {label}: failed_error={failed_error} (cols checked={total_cols}) -> {out_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
