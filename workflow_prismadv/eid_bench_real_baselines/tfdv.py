#!/usr/bin/env python3
"""TensorFlow Data Validation task-agnostic baseline for EIDBench-real."""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import oyaml as yaml
import pandas as pd
import tensorflow_data_validation as tfdv

from prismadv.project_manager.manager.multi_table import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_SCRIPT_ID = "project"
BASELINE_METHOD = "tfdv"
MODEL_TAG = "tfdv"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example-id", required=True)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--label", action="append", default=None,
                   help="Limit to specific corruption label(s). Default: all.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def infer_clean_schemas(pm: MultiTableProjectManager, table_names: list[str]) -> dict:
    """Infer one TFDV schema per table from its clean data."""
    schemas = {}
    for table in table_names:
        clean_df = pd.read_csv(pm.get_table_path(table, "clean"))
        stats = tfdv.generate_statistics_from_dataframe(clean_df)
        schemas[table] = tfdv.infer_schema(stats)
    return schemas


def detect_table(schema, corrupted_path: Path) -> tuple[int, int]:
    """Validate a corrupted table against its clean schema."""
    n_features = len(schema.feature)
    corrupted_df = pd.read_csv(corrupted_path)
    stats = tfdv.generate_statistics_from_dataframe(corrupted_df)
    anomalies = tfdv.validate_statistics(stats, schema)
    return n_features, len(anomalies.anomaly_info)


def main(argv=None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    pm = MultiTableProjectManager(project_root=project_root, example_id=args.example_id)
    table_names = pm.get_table_names()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{BASELINE_METHOD}--{MODEL_TAG}--{ts}"

    constraints_dir = utils.constraints_dir(args.example_id, PROJECT_SCRIPT_ID, project_root, create=True)
    artifact_path = constraints_dir / f"{stem}.yaml"
    if not (artifact_path.exists() and not args.overwrite):
        yaml.safe_dump({
            "baseline_method": BASELINE_METHOD,
            "model_name": MODEL_TAG,
            "reference": "TensorFlow Data Validation (schema inference + validate_statistics)",
            "note": ("Task- and dataset-agnostic schema inferred from the clean data and "
                     "validated against each corrupted bundle; validation_results written by "
                     "this script (TFDV does not emit executable constraints for the shared "
                     "Deequ validator)."),
        }, artifact_path.open("w"), sort_keys=False)
    print(f"artifact: {artifact_path}")

    labels = utils.corruption_labels(pm)
    if args.label:
        wanted = set(args.label)
        labels = [l for l in labels if l in wanted]
    print(f"example_id={args.example_id} tables={table_names} labels={len(labels)}")

    schemas = infer_clean_schemas(pm, table_names)

    for label in labels:
        failed_error = 0
        total_cols = 0
        for table in table_names:
            corrupted_path = pm.get_table_path(table, "corrupted", corruption_label=label)
            if not Path(corrupted_path).exists():
                continue
            n_features, flagged = detect_table(schemas[table], corrupted_path)
            total_cols += n_features
            failed_error += flagged

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
