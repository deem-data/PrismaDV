from pathlib import Path

import pytest

from prismadv.project_manager import MultiTableProjectManager
from tests.eid_bench_real.test_project_manager import write_example
from workflow_prismadv.eid_bench_real_experiments import utils


def write_error_config(example_root: Path, label: str, table: str = "patients") -> None:
    errors_dir = example_root / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    (errors_dir / f"{label}.yaml").write_text(
        f"""
label: {label}
description: Test config for {label}.
table: {table}
corruption:
  class: ColumnDropping
  columns: [name]
expected_current_outcome:
  patient_summary: execution_failure
""".lstrip(),
        encoding="utf-8",
    )


def test_processed_paths_are_stable(tmp_path):
    assert utils.eid_bench_real_processed_root(tmp_path) == tmp_path / "data_processed" / "eid_bench_real"
    assert utils.example_processed_root("example_a", tmp_path) == (
        tmp_path / "data_processed" / "eid_bench_real" / "example_a"
    )
    assert utils.constraints_dir("example_a", "patient_summary", tmp_path) == (
        tmp_path / "data_processed" / "eid_bench_real" / "example_a" / "constraints" / "patient_summary"
    )
    assert utils.constraint_artifact_path("example_a", "patient_summary", "prismadv--test.yaml", tmp_path) == (
        tmp_path
        / "data_processed"
        / "eid_bench_real"
        / "example_a"
        / "constraints"
        / "patient_summary"
        / "prismadv--test.yaml"
    )
    assert utils.constraints_validation_dir("example_a", "patient_summary", "clean", tmp_path) == (
        tmp_path
        / "data_processed"
        / "eid_bench_real"
        / "example_a"
        / "constraints_validation"
        / "patient_summary"
        / "clean"
    )
    assert utils.constraints_validation_dir(
        "example_a",
        "patient_summary",
        "corrupted",
        tmp_path,
        corruption_label="missing_key",
    ) == (
        tmp_path
        / "data_processed"
        / "eid_bench_real"
        / "example_a"
        / "constraints_validation"
        / "patient_summary"
        / "corrupted"
        / "missing_key"
    )
    assert utils.execution_dir("example_a", "patient_summary", "observed", tmp_path) == (
        tmp_path
        / "data_processed"
        / "eid_bench_real"
        / "example_a"
        / "execution"
        / "patient_summary"
        / "observed"
    )
    assert utils.expected_outcomes_path("example_a", tmp_path) == (
        tmp_path / "data_processed" / "eid_bench_real" / "example_a" / "outcomes" / "expected_outcomes.yaml"
    )


def test_processed_paths_can_create_directories(tmp_path):
    path = utils.constraints_validation_dir(
        "example_a",
        "patient_summary",
        "corrupted",
        tmp_path,
        corruption_label="missing_key",
        create=True,
    )

    assert path.exists()
    assert path.is_dir()


def test_variant_component_validates_corruption_labels():
    assert utils.variant_component("clean") == Path("clean")
    assert utils.variant_component("observed") == Path("observed")
    assert utils.variant_component("corrupted", "missing_key") == Path("corrupted") / "missing_key"

    with pytest.raises(ValueError, match="corruption_label is required"):
        utils.variant_component("corrupted")
    with pytest.raises(ValueError, match="only valid for corrupted"):
        utils.variant_component("clean", "missing_key")
    with pytest.raises(ValueError, match="variant must be one of"):
        utils.variant_component("bad_variant")


def test_load_error_configs_from_benchmark_local_errors(tmp_path):
    example_root = write_example(tmp_path)
    write_error_config(example_root, "drop_patient_name")
    write_error_config(example_root, "drop_encounter_name", table="encounters")
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")

    configs = utils.load_error_configs(manager)

    assert sorted(configs) == ["drop_encounter_name", "drop_patient_name"]
    assert utils.corruption_labels(manager) == ["drop_encounter_name", "drop_patient_name"]
    assert configs["drop_patient_name"]["expected_current_outcome"] == {
        "patient_summary": "execution_failure",
    }


def test_load_error_config_derives_label_from_filename(tmp_path):
    # eid-synth id style: identity is the file stem, no internal label field.
    example_root = write_example(tmp_path)
    errors_dir = example_root / "errors"
    errors_dir.mkdir()
    (errors_dir / "project_0.yaml").write_text(
        """
description: No internal label field.
table: patients
corruption:
  class: ColumnDropping
  columns: [name]
""".lstrip(),
        encoding="utf-8",
    )
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")

    configs = utils.load_error_configs(manager)

    assert sorted(configs) == ["project_0"]
    assert configs["project_0"]["label"] == "project_0"
