#!/usr/bin/env python3
"""Generate EIDBench-real corrupted bundles from benchmark-local error YAMLs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import oyaml as yaml
import pandas as pd

from prismadv.error_injection.managers import MultiTableErrorInjectionManager, TableCorruptionSpec
from prismadv.project_manager import MultiTableProjectManager
from prismadv.utils import get_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", default="omop_cdm_synthea")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing one YAML file per corruption. Defaults to <example>/errors.",
    )
    parser.add_argument(
        "--clean-tables-dir",
        type=Path,
        default=None,
        help="Directory containing clean EIDBench-real table CSVs. Defaults to manifest clean_tables_dir.",
    )
    parser.add_argument(
        "--corrupted-root",
        type=Path,
        default=None,
        help="Directory where corruption label folders are written. Defaults to manifest corrupted_root_dir.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Only generate the selected corruption label. May be provided more than once.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Maximum rows to mutate when a YAML config uses severity: cli_fraction.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.01,
        help="Approximate row fraction when a YAML config uses severity: cli_fraction.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete existing corruption outputs first.")
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config


def load_yaml_configs(config_dir: Path, labels: set[str] | None = None) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(config_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"no corruption YAML files found in {config_dir}")

    configs = []
    for path in paths:
        config = load_yaml_config(path)
        # Identity is the file stem (e.g. project_0), eid-synth id style.
        config["label"] = path.stem
        if labels is not None and config["label"] not in labels:
            continue
        configs.append((path, config))

    if labels is not None:
        found = {config["label"] for _, config in configs}
        missing = sorted(labels - found)
        if missing:
            raise ValueError(f"requested corruption labels not found in {config_dir}: {missing}")
    return configs


def mutation_count(row_count: int, fraction: float, max_rows: int) -> int:
    if row_count <= 0:
        return 0
    return max(1, min(max_rows, int(row_count * fraction)))


def mutation_severity(row_count: int, fraction: float, max_rows: int) -> float:
    count = mutation_count(row_count, fraction, max_rows)
    if count == 0:
        return 0.0
    return min(1.0, count / row_count)


def row_counts(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {table_name: int(len(table)) for table_name, table in tables.items()}


def changed_cell_count(
    clean_tables: dict[str, pd.DataFrame],
    corrupted_tables: dict[str, pd.DataFrame],
    table_name: str,
    column_name: str,
) -> int | None:
    clean_table = clean_tables[table_name]
    corrupted_table = corrupted_tables[table_name]
    if column_name not in clean_table.columns or column_name not in corrupted_table.columns:
        return None
    if len(clean_table) != len(corrupted_table):
        return None
    clean_column = clean_table[column_name]
    corrupted_column = corrupted_table[column_name]
    same_values = clean_column.eq(corrupted_column) | (clean_column.isna() & corrupted_column.isna())
    return int((~same_values).sum())


def patient_orphan_count(tables: dict[str, pd.DataFrame]) -> int:
    patient_ids = set(tables["patients"]["Id"].astype(str))
    return int((~tables["encounters"]["PATIENT"].astype(str).isin(patient_ids)).sum())


def negative_medication_cost_count(tables: dict[str, pd.DataFrame]) -> int:
    costs = pd.to_numeric(tables["medications"]["TOTALCOST"], errors="coerce")
    return int((costs < 0).sum())


def medication_encounter_orphan_count(tables: dict[str, pd.DataFrame]) -> int:
    encounter_ids = set(tables["encounters"]["Id"].astype(str))
    medication_encounters = tables["medications"]["ENCOUNTER"].astype(str)
    return int((~medication_encounters.isin(encounter_ids)).sum())


def duplicate_patient_id_count(tables: dict[str, pd.DataFrame]) -> int:
    return int(tables["patients"]["Id"].duplicated(keep=False).sum())


def unexpected_encounter_class_count(
    clean_tables: dict[str, pd.DataFrame],
    corrupted_tables: dict[str, pd.DataFrame],
) -> int:
    clean_classes = set(clean_tables["encounters"]["ENCOUNTERCLASS"].dropna().astype(str))
    corrupted_classes = corrupted_tables["encounters"]["ENCOUNTERCLASS"].dropna().astype(str)
    return int((~corrupted_classes.isin(clean_classes)).sum())


def invalid_medication_cost_type_count(tables: dict[str, pd.DataFrame]) -> int:
    cost_values = tables["medications"]["TOTALCOST"]
    numeric_costs = pd.to_numeric(cost_values, errors="coerce")
    return int((cost_values.notna() & numeric_costs.isna()).sum())


def unexpected_value_tuple_count(
    clean_tables: dict[str, pd.DataFrame],
    corrupted_tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> int:
    table_name = config["table"]
    columns = config["corruption"].get("columns", [])
    if not columns:
        return 0
    if not all(column in clean_tables[table_name].columns for column in columns):
        return 0
    if not all(column in corrupted_tables[table_name].columns for column in columns):
        return 0

    clean_tuples = set(clean_tables[table_name][columns].dropna().itertuples(index=False, name=None))
    corrupted_tuples = corrupted_tables[table_name][columns].dropna().itertuples(index=False, name=None)
    return int(sum(tuple_value not in clean_tuples for tuple_value in corrupted_tuples))


def future_date_count(
    clean_tables: dict[str, pd.DataFrame],
    corrupted_tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> int:
    table_name = config["table"]
    columns = config["corruption"].get("columns", [])
    count = 0
    for column in columns:
        if column not in clean_tables[table_name].columns or column not in corrupted_tables[table_name].columns:
            continue
        clean_values = pd.to_datetime(clean_tables[table_name][column], errors="coerce", utc=True)
        corrupted_values = pd.to_datetime(corrupted_tables[table_name][column], errors="coerce", utc=True)
        clean_max = clean_values.max()
        if pd.isna(clean_max):
            continue
        count += int((corrupted_values > clean_max).sum())
    return count


def temporal_order_violation_count(
    corrupted_tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> int:
    table_name = config["table"]
    columns = config["corruption"].get("columns", [])
    if len(columns) < 2:
        return 0
    start_column, stop_column = columns[:2]
    if start_column not in corrupted_tables[table_name].columns or stop_column not in corrupted_tables[table_name].columns:
        return 0
    starts = pd.to_datetime(corrupted_tables[table_name][start_column], errors="coerce", utc=True)
    stops = pd.to_datetime(corrupted_tables[table_name][stop_column], errors="coerce", utc=True)
    return int((starts.notna() & stops.notna() & (stops <= starts)).sum())


def missing_columns(tables: dict[str, pd.DataFrame], table_name: str, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in tables[table_name].columns]


def resolve_reference_values(
    tables: dict[str, pd.DataFrame],
    reference_values: dict[str, Any],
) -> dict[str, list[Any]]:
    resolved = {}
    for column_name, reference_spec in reference_values.items():
        if isinstance(reference_spec, dict) and {"table", "column"} <= set(reference_spec):
            resolved[column_name] = tables[reference_spec["table"]][reference_spec["column"]].dropna().tolist()
        else:
            resolved[column_name] = reference_spec
    return resolved


def resolve_corruption_params(
    config: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    table_name: str,
    fraction: float,
    max_rows: int,
) -> dict[str, Any]:
    params = dict(config.get("params", {}))
    if params.get("severity") == "cli_fraction":
        params["severity"] = mutation_severity(len(tables[table_name]), fraction, max_rows)
    if "reference_values" in params:
        params["reference_values"] = resolve_reference_values(tables, params["reference_values"])
    return params


def build_table_spec(
    manager: MultiTableErrorInjectionManager,
    config: dict[str, Any],
    fraction: float,
    max_rows: int,
) -> TableCorruptionSpec:
    corruption_config = config["corruption"]
    corruption_class_name = corruption_config["class"]
    corruption_class = manager.corruption_classes[corruption_class_name]
    params = resolve_corruption_params(corruption_config, manager.clean_tables, config["table"], fraction, max_rows)
    corruption = corruption_class(columns=corruption_config.get("columns"), **params)
    return TableCorruptionSpec(table_name=config["table"], corruptions=[corruption])


def metric_value(
    metric_name: str,
    clean_tables: dict[str, pd.DataFrame],
    corrupted_tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> Any:
    if metric_name == "orphan_patient_rows":
        return patient_orphan_count(corrupted_tables)
    if metric_name == "negative_medication_cost_rows":
        return negative_medication_cost_count(corrupted_tables)
    if metric_name == "orphan_medication_encounter_rows":
        return medication_encounter_orphan_count(corrupted_tables)
    if metric_name == "duplicate_patient_id_rows":
        return duplicate_patient_id_count(corrupted_tables)
    if metric_name == "unexpected_encounter_class_rows":
        return unexpected_encounter_class_count(clean_tables, corrupted_tables)
    if metric_name == "invalid_medication_cost_type_rows":
        return invalid_medication_cost_type_count(corrupted_tables)
    if metric_name == "missing_columns":
        return missing_columns(corrupted_tables, config["table"], config["corruption"].get("columns", []))
    if metric_name == "unexpected_value_tuple_rows":
        return unexpected_value_tuple_count(clean_tables, corrupted_tables, config)
    if metric_name == "future_date_rows":
        return future_date_count(clean_tables, corrupted_tables, config)
    if metric_name == "temporal_order_violation_rows":
        return temporal_order_violation_count(corrupted_tables, config)
    raise ValueError(f"unknown report metric: {metric_name}")


def build_report(
    clean_tables: dict[str, pd.DataFrame],
    corrupted_tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Any]:
    columns = config["corruption"].get("columns", [])
    report = {
        "label": config["label"],
        "table": config["table"],
        "columns": columns,
        "expected_current_outcome": config.get("expected_current_outcome"),
        "row_counts": row_counts(corrupted_tables),
    }
    added_rows = len(corrupted_tables[config["table"]]) - len(clean_tables[config["table"]])
    if added_rows:
        report["added_rows"] = int(added_rows)
    if len(columns) == 1:
        changed_count = changed_cell_count(clean_tables, corrupted_tables, config["table"], columns[0])
        if changed_count is not None:
            report["mutated_rows"] = changed_count
    for metric_name in config.get("report_metrics", []):
        report[metric_name] = metric_value(metric_name, clean_tables, corrupted_tables, config)
    return report


def write_corruption(
    output_root: Path,
    config_path: Path,
    manager: MultiTableErrorInjectionManager,
    report: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if manager.post_corruption_tables is None:
        raise RuntimeError("corruption manager did not produce tables")

    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        existing_report = output_root / "corruption_report.json"
        if existing_report.exists():
            report = json.loads(existing_report.read_text(encoding="utf-8"))
        report["skipped_existing_output"] = True
        return report

    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)

    manager.save_data(
        output_root,
        overwrite=False,
        include_input_copy=True,
        report=report,
    )
    shutil.copyfile(config_path, output_root / "corruption_source.yaml")
    report["skipped_existing_output"] = False
    return report


def main() -> int:
    args = parse_args()
    project_root = get_project_root() if args.project_root is None else args.project_root.resolve()
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=args.example_id)
    clean_tables_dir = (
        args.clean_tables_dir.resolve()
        if args.clean_tables_dir is not None
        else project_manager.resolve_path(project_manager.input_layout["clean_tables_dir"])
    )
    corrupted_root = (
        args.corrupted_root.resolve()
        if args.corrupted_root is not None
        else project_manager.get_corrupted_root_dir()
    )
    config_dir = (
        args.config_dir.resolve()
        if args.config_dir is not None
        else project_manager.example_path / "errors"
    )
    labels = set(args.label) if args.label else None
    table_names = project_manager.get_table_names()

    base_manager = MultiTableErrorInjectionManager(clean_tables_dir, table_names)
    clean_tables = base_manager.clean_tables

    reports = []
    for config_path, config in load_yaml_configs(config_dir, labels):
        manager = MultiTableErrorInjectionManager(clean_tables_dir, table_names)
        table_spec = build_table_spec(manager, config, args.fraction, args.max_rows)
        manager.error_injection([table_spec])
        if manager.post_corruption_tables is None:
            raise RuntimeError(f"{config['label']} corruption did not produce tables")
        report = build_report(clean_tables, manager.post_corruption_tables, config)
        reports.append(
            write_corruption(
                corrupted_root / config["label"],
                config_path,
                manager,
                report,
                args.overwrite,
            )
        )

    summary = {
        "example_id": args.example_id,
        "clean_tables_dir": str(clean_tables_dir),
        "config_dir": str(config_dir),
        "corrupted_root": str(corrupted_root),
        "corruptions": reports,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
