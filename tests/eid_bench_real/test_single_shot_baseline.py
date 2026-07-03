from pathlib import Path

import oyaml as yaml

from tests.eid_bench_real.test_multi_table_loader import write_tables
from tests.eid_bench_real.test_prepare_prismadv_inputs import (
    add_adapter_manifest_files,
    write_adapter_files,
)
from tests.eid_bench_real.test_project_manager import write_example
from workflow_prismadv.eid_bench_real_baselines import single_shot


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
        return [(True, "") for _ in code_list_for_constraints]


def write_eid_bench_real_project(tmp_path: Path) -> Path:
    example_root = write_example(tmp_path)
    write_tables(example_root)
    write_adapter_files(example_root)
    add_adapter_manifest_files(example_root)
    return example_root


def test_single_shot_real_etl_generates_project_level_constraints(tmp_path):
    write_eid_bench_real_project(tmp_path)
    captured_input = {}

    def fake_inferer(input_variables):
        captured_input.update(input_variables)
        return [
            {
                "table_name": "patients",
                "column_name": "Id",
                "code_for_constraint": '.isComplete("Id")',
            },
            {
                "table_name": "encounters",
                "column_name": "PATIENT",
                "code_for_constraint": '.isComplete("PATIENT")',
            },
            {
                "table_name": "missing",
                "column_name": "Id",
                "code_for_constraint": '.isComplete("Id")',
            },
        ]

    output_path = single_shot.generate_single_shot_constraints(
        project_root=tmp_path,
        example_id="example_a",
        model_name="gpt-5-mini",
        overwrite=True,
        constraint_inferer=fake_inferer,
        dq_manager=FakeDQManager(),
    )

    assert output_path == (
        tmp_path
        / "data_processed"
        / "eid_bench_real"
        / "example_a"
        / "constraints"
        / "project"
        / output_path.name
    )
    assert output_path.name.startswith("single_shot_real_etl--gpt-5-mini--")
    assert "Project scripts:" in captured_input["downstream_task_description"]
    assert "patients:" in captured_input["tables_desc"]
    assert "encounters:" in captured_input["data_sample"]

    raw = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert raw["baseline_method"] == "single_shot_real_etl"
    assert raw["input_artifact"].endswith("constraints/project/prismadv_inputs.yaml")
    assert sorted(raw["constraints"]) == ["encounters.PATIENT", "patients.Id"]
    assert raw["constraints"]["patients.Id"]["table_name"] == "patients"
    assert raw["constraints"]["patients.Id"]["column_group"] == "Id"
    assert raw["constraints"]["patients.Id"]["code"][0]["validity"] is True


def test_single_shot_real_etl_reuses_existing_model_artifact_without_overwrite(tmp_path):
    write_eid_bench_real_project(tmp_path)
    calls = 0

    def fake_inferer(input_variables):
        nonlocal calls
        calls += 1
        return [
            {
                "table_name": "patients",
                "column_name": "Id",
                "code_for_constraint": '.isComplete("Id")',
            }
        ]

    first = single_shot.generate_single_shot_constraints(
        project_root=tmp_path,
        example_id="example_a",
        model_name="gpt-5-mini",
        overwrite=True,
        constraint_inferer=fake_inferer,
        dq_manager=FakeDQManager(),
    )
    second = single_shot.generate_single_shot_constraints(
        project_root=tmp_path,
        example_id="example_a",
        model_name="gpt-5-mini",
        overwrite=False,
        constraint_inferer=fake_inferer,
        dq_manager=FakeDQManager(),
    )

    assert first == second
    assert calls == 1

