import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import oyaml as yaml

from tests.eid_bench_real.test_project_manager import write_example
from workflow_prismadv.eid_bench_real_experiments import _0_error_injection


def write_clean_tables(example_root: Path) -> None:
    clean_tables = example_root / "files" / "clean" / "tables"
    pd.DataFrame(
        [
            {"Id": "p1", "name": "Ada"},
            {"Id": "p2", "name": "Lin"},
        ]
    ).to_csv(clean_tables / "patients.csv", index=False)
    pd.DataFrame(
        [
            {"Id": "e1", "PATIENT": "p1"},
            {"Id": "e2", "PATIENT": "p2"},
        ]
    ).to_csv(clean_tables / "encounters.csv", index=False)


def write_error_config(
    example_root: Path,
    label: str,
    *,
    table: str = "patients",
    columns: list[str] | None = None,
) -> None:
    errors_dir = example_root / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    columns = ["name"] if columns is None else columns
    config = {
        "label": label,
        "description": f"Drop {columns} from {table}.",
        "table": table,
        "corruption": {
            "class": "ColumnDropping",
            "columns": columns,
        },
        "report_metrics": ["missing_columns"],
        "expected_current_outcome": {
            "patient_summary": "execution_failure",
        },
    }
    (errors_dir / f"{label}.yaml").write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")


def run_error_injection(tmp_path: Path, monkeypatch, *extra_args: str) -> None:
    argv = [
        "_0_error_injection.py",
        "--project-root",
        str(tmp_path),
        "--example-id",
        "example_a",
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert _0_error_injection.main() == 0


def test_error_injection_reads_benchmark_local_errors_by_default(tmp_path, monkeypatch):
    example_root = write_example(tmp_path)
    write_clean_tables(example_root)
    write_error_config(example_root, "drop_patient_name")

    run_error_injection(tmp_path, monkeypatch, "--label", "drop_patient_name")

    output_root = example_root / "files" / "corrupted" / "drop_patient_name"
    patients_table = pd.read_csv(output_root / "tables" / "patients.csv")
    patients_input = pd.read_csv(output_root / "input" / "patients.csv")
    encounters_table = pd.read_csv(output_root / "tables" / "encounters.csv")
    report = json.loads((output_root / "corruption_report.json").read_text(encoding="utf-8"))

    assert "name" not in patients_table.columns
    assert "name" not in patients_input.columns
    assert list(encounters_table.columns) == ["Id", "PATIENT"]
    assert (output_root / "corruption_source.yaml").exists()
    assert report["label"] == "drop_patient_name"
    assert report["table"] == "patients"
    assert report["columns"] == ["name"]
    assert report["missing_columns"] == ["name"]
    assert report["expected_current_outcome"] == {"patient_summary": "execution_failure"}


def test_error_injection_label_filters_configs(tmp_path, monkeypatch):
    example_root = write_example(tmp_path)
    write_clean_tables(example_root)
    write_error_config(example_root, "drop_patient_name")
    write_error_config(example_root, "drop_encounter_patient", table="encounters", columns=["PATIENT"])

    run_error_injection(tmp_path, monkeypatch, "--label", "drop_encounter_patient")

    corrupted_root = example_root / "files" / "corrupted"
    assert (corrupted_root / "drop_encounter_patient" / "tables" / "encounters.csv").exists()
    assert not (corrupted_root / "drop_patient_name").exists()
    report = json.loads(
        (corrupted_root / "drop_encounter_patient" / "corruption_report.json").read_text(encoding="utf-8")
    )
    assert report["label"] == "drop_encounter_patient"
    assert report["missing_columns"] == ["PATIENT"]


def test_error_injection_reports_unknown_label(tmp_path, monkeypatch):
    example_root = write_example(tmp_path)
    write_clean_tables(example_root)
    write_error_config(example_root, "drop_patient_name")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "_0_error_injection.py",
            "--project-root",
            str(tmp_path),
            "--example-id",
            "example_a",
            "--label",
            "not_declared",
        ],
    )

    with pytest.raises(ValueError, match="requested corruption labels not found"):
        _0_error_injection.main()


def test_error_injection_reports_unexpected_value_tuples(tmp_path, monkeypatch):
    example_root = write_example(tmp_path)
    write_clean_tables(example_root)
    errors_dir = example_root / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "label": "patient_id_name_mismatch",
        "description": "Pair valid patient IDs with the wrong names.",
        "table": "patients",
        "corruption": {
            "class": "FunctionalDependencyViolation",
            "columns": ["Id", "name"],
            "params": {
                "key_columns": ["Id"],
                "dependent_columns": ["name"],
                "severity": 1.0,
                "random_state": 11,
            },
        },
        "report_metrics": ["unexpected_value_tuple_rows"],
        "expected_current_outcome": {
            "patient_summary": "safe",
        },
    }
    (errors_dir / "patient_id_name_mismatch.yaml").write_text(
        yaml.dump(config, sort_keys=False),
        encoding="utf-8",
    )

    run_error_injection(tmp_path, monkeypatch, "--label", "patient_id_name_mismatch")

    output_root = example_root / "files" / "corrupted" / "patient_id_name_mismatch"
    report = json.loads((output_root / "corruption_report.json").read_text(encoding="utf-8"))
    assert report["unexpected_value_tuple_rows"] == 2


def test_error_injection_fails_clearly_for_missing_clean_table(tmp_path, monkeypatch):
    example_root = write_example(tmp_path)
    write_clean_tables(example_root)
    (example_root / "files" / "clean" / "tables" / "encounters.csv").unlink()
    write_error_config(example_root, "drop_patient_name")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "_0_error_injection.py",
            "--project-root",
            str(tmp_path),
            "--example-id",
            "example_a",
            "--label",
            "drop_patient_name",
        ],
    )

    with pytest.raises(FileNotFoundError, match="missing clean table"):
        _0_error_injection.main()
