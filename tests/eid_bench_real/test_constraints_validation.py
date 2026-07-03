from pathlib import Path

import oyaml as yaml

from prismadv.data_models import ConstraintsWithSources
from prismadv.data_models.constraints_v2 import (
    AssumptionEntry,
    CodeEntry,
    ColumnConstraintsWithSources,
)
from prismadv.project_manager import MultiTableProjectManager
from tests.eid_bench_real.test_multi_table_loader import write_tables
from tests.eid_bench_real.test_project_manager import write_example
from workflow_prismadv.eid_bench_real_experiments import _4_constraints_validation as validation


class FakeGateway:
    def close(self):
        pass


class FakeSparkContext:
    _gateway = FakeGateway()


class FakeSpark:
    sparkContext = FakeSparkContext()

    def stop(self):
        pass


class FakeDQManager:
    def __init__(self):
        self.spark = FakeSpark()

    def spark_df_from_pandas_df(self, pandas_df, schema=None, spark_session=None):
        return pandas_df, self.spark

    def validate_constraints_with_reasons(self, spark, spark_df, code_list_for_constraints, isolated_check=True):
        results = []
        for code in code_list_for_constraints:
            if code == '.isComplete("PATIENT")' and spark_df["PATIENT"].isna().any():
                results.append((False, "PATIENT contains null values"))
            else:
                results.append((True, ""))
        return results


def write_constraint_artifact(path: Path) -> None:
    constraints = ConstraintsWithSources()
    constraints.data_map["patients.Id"] = ColumnConstraintsWithSources(
        assumptions=[AssumptionEntry(text="patients.Id should be complete.")],
        code=[CodeEntry(suggestion='.isComplete("Id")', validity=True, level="error")],
        table_name="patients",
        column_group="Id",
    )
    constraints.data_map["encounters.PATIENT"] = ColumnConstraintsWithSources(
        assumptions=[AssumptionEntry(text="encounters.PATIENT should be complete.")],
        code=[CodeEntry(suggestion='.isComplete("PATIENT")', validity=True, level="error")],
        table_name="encounters",
        column_group="PATIENT",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(constraints.to_dict(), sort_keys=False), encoding="utf-8")


def test_validation_targets_include_clean_and_selected_corruptions(tmp_path):
    write_example(tmp_path)
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")

    assert validation.validation_targets(manager, "all", ["missing_key"]) == [
        ("clean", None),
        ("corrupted", "missing_key"),
    ]


def test_java_security_manager_compat_flag_is_added_once(monkeypatch):
    monkeypatch.setenv("JAVA_TOOL_OPTIONS", "-Xmx2g")

    validation.ensure_java_security_manager_compat()
    validation.ensure_java_security_manager_compat()

    assert validation.JAVA_SECURITY_MANAGER_FLAG in validation.os.environ["JAVA_TOOL_OPTIONS"]
    assert validation.os.environ["JAVA_TOOL_OPTIONS"].count(validation.JAVA_SECURITY_MANAGER_FLAG) == 1


def test_latest_constraint_path_includes_single_shot_real_etl_artifacts(tmp_path):
    output_dir = (
        tmp_path
        / "data_processed"
        / "eid_bench_real"
        / "example_a"
        / "constraints"
        / "project"
    )
    output_dir.mkdir(parents=True)
    prismadv_path = output_dir / "prismadv_real_etl--fake.yaml"
    single_shot_path = output_dir / "single_shot_real_etl--fake.yaml"
    prismadv_path.write_text("constraints: {}\n", encoding="utf-8")
    single_shot_path.write_text("constraints: {}\n", encoding="utf-8")
    prismadv_path.touch()
    single_shot_path.touch()

    assert validation.latest_constraint_path("example_a", "project", tmp_path) == single_shot_path


def test_validate_constraints_on_bundle_groups_by_table(tmp_path):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")
    constraint_path = tmp_path / "constraints.yaml"
    write_constraint_artifact(constraint_path)
    constraints = validation.load_constraints(constraint_path)

    clean_result = validation.validate_constraints_on_bundle(
        manager,
        constraints,
        script_id="patient_summary",
        variant="clean",
        dq_manager=FakeDQManager(),
    )
    corrupted_result = validation.validate_constraints_on_bundle(
        manager,
        constraints,
        script_id="patient_summary",
        variant="corrupted",
        corruption_label="missing_key",
        dq_manager=FakeDQManager(),
    )

    assert clean_result["summary"]["passed_error"] == 2
    assert clean_result["summary"]["failed_error"] == 0
    assert corrupted_result["summary_by_table"]["encounters"]["failed_error"] == 1
    assert corrupted_result["tables"]["encounters"]["PATIENT"]["code"][0]["reason_if_failed"] == (
        "PATIENT contains null values"
    )


def test_run_validation_writes_clean_and_corrupted_artifacts(tmp_path, monkeypatch):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    constraint_path = (
        tmp_path
        / "data_processed"
        / "eid_bench_real"
        / "example_a"
        / "constraints"
        / "patient_summary"
        / "prismadv_real_etl--fake.yaml"
    )
    write_constraint_artifact(constraint_path)

    monkeypatch.setattr(validation, "DeequDataQualityManager", FakeDQManager)

    output_paths = validation.run_validation(
        project_root=tmp_path,
        example_id="example_a",
        script_id="patient_summary",
        constraint_path=constraint_path,
        target="all",
        labels=["missing_key"],
        overwrite=True,
    )

    assert len(output_paths) == 2
    assert output_paths[0] == (
        tmp_path
        / "data_processed"
        / "eid_bench_real"
        / "example_a"
        / "constraints_validation"
        / "patient_summary"
        / "clean"
        / "validation_results__prismadv_real_etl--fake.yaml"
    )
    assert output_paths[1] == (
        tmp_path
        / "data_processed"
        / "eid_bench_real"
        / "example_a"
        / "constraints_validation"
        / "patient_summary"
        / "corrupted"
        / "missing_key"
        / "validation_results__prismadv_real_etl--fake.yaml"
    )
    raw = yaml.safe_load(output_paths[1].read_text(encoding="utf-8"))
    assert raw["variant"] == "corrupted"
    assert raw["corruption_label"] == "missing_key"
    assert raw["summary"]["failed_error"] == 1
