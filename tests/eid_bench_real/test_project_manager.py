from pathlib import Path

import pytest

from prismadv.project_manager import MultiTableProjectManager


def write_example(tmp_path: Path, example_id: str = "example_a") -> Path:
    example_root = tmp_path / "benchmarks" / "EIDBench-real" / example_id
    for path in [
        example_root / "files" / "observed" / "tables",
        example_root / "files" / "clean" / "tables",
        example_root / "files" / "clean" / "input",
        example_root / "files" / "corrupted" / "missing_key" / "tables",
        example_root / "files" / "corrupted" / "missing_key" / "input",
        example_root / "adapter",
        example_root / "expected" / "patient_summary",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    (example_root / "manifest.yaml").write_text(
        """
example_id: example_a
display_name: Example A
source_repo: https://example.com/repo.git
source_commit: null
domain: healthcare
primary_runtime: pyspark
primary_language: python
task_type: etl
clean_should_pass: true
corruption_mode: materialized
notes_on_repairs: Local adapter only.
input_layout:
  observed_tables_dir: files/observed/tables
  clean_input_dir: files/clean/input
  clean_tables_dir: files/clean/tables
  corrupted_root_dir: files/corrupted
expected_outputs:
  patient_summary:
    clean_summary_path: expected/patient_summary/clean_summary.json
tables:
  patients:
    format: csv
    observed_path: files/observed/tables/patients.csv
    clean_path: files/clean/tables/patients.csv
    primary_key: Id
  encounters:
    format: csv
    observed_path: files/observed/tables/encounters.csv
    clean_path: files/clean/tables/encounters.csv
    corrupted_paths:
      missing_key: files/corrupted/missing_key/tables/encounters_bad.csv
scripts:
  patient_summary:
    entrypoint: adapter/run_script.py
    reads: [patients, encounters]
    writes: [summary.json]
    validates_with: adapter/validator.py
    timeout_seconds: 60
corruptions:
  missing_key:
    table: encounters
    column: PATIENT
    expected_effect: validation_failure
    rationale: Breaks patient join coverage.
""".lstrip()
    )
    return example_root


def test_project_manager_resolves_manifest_paths(tmp_path):
    example_root = write_example(tmp_path)

    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")

    assert manager.example_path == example_root
    assert manager.get_table_names() == ["encounters", "patients"]
    assert manager.get_script_ids() == ["patient_summary"]
    assert manager.get_table_path("patients", "observed") == (
        example_root / "files" / "observed" / "tables" / "patients.csv"
    )
    assert manager.get_table_path("patients", "clean") == (
        example_root / "files" / "clean" / "tables" / "patients.csv"
    )
    assert manager.get_table_path("encounters", "corrupted", "missing_key") == (
        example_root / "files" / "corrupted" / "missing_key" / "tables" / "encounters_bad.csv"
    )
    assert manager.get_clean_input_dir() == example_root / "files" / "clean" / "input"
    assert manager.get_corrupted_input_dir("missing_key") == (
        example_root / "files" / "corrupted" / "missing_key" / "input"
    )
    assert manager.get_script_entrypoint("patient_summary") == example_root / "adapter" / "run_script.py"
    assert manager.get_expected_clean_summary_path("patient_summary") == (
        example_root / "expected" / "patient_summary" / "clean_summary.json"
    )


def test_project_manager_infers_corrupted_table_path_when_not_declared(tmp_path):
    example_root = write_example(tmp_path)

    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")

    assert manager.get_table_path("patients", "corrupted", "missing_key") == (
        example_root / "files" / "corrupted" / "missing_key" / "tables" / "patients.csv"
    )


def test_project_manager_requires_corruption_label(tmp_path):
    write_example(tmp_path)
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")

    with pytest.raises(ValueError, match="corruption_label is required"):
        manager.get_table_path("patients", "corrupted")


def test_project_manager_validates_required_manifest_fields(tmp_path):
    example_root = write_example(tmp_path)
    (example_root / "manifest.yaml").write_text("example_id: example_a\n")

    with pytest.raises(ValueError, match="missing required fields"):
        MultiTableProjectManager(project_root=tmp_path, example_id="example_a")


def test_project_manager_lists_available_examples(tmp_path):
    write_example(tmp_path)

    assert MultiTableProjectManager.available_examples(project_root=tmp_path) == ["example_a"]
