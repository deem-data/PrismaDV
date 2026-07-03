from pathlib import Path

import oyaml as yaml

from prismadv.data_models.constraints_v2 import (
    AssumptionEntry,
    CodeEntry,
    ColumnConstraintsWithSources,
    ConstraintsWithSources,
    SourceLocation,
    SourceLocations,
)
from tests.eid_bench_real.test_multi_table_loader import write_tables
from tests.eid_bench_real.test_prepare_prismadv_inputs import (
    add_adapter_manifest_files,
    write_adapter_files,
)
from tests.eid_bench_real.test_project_manager import write_example
from workflow_prismadv.eid_bench_real_experiments import _3_0_prepare_prismadv_inputs as input_prep
from workflow_prismadv.eid_bench_real_experiments import _3_1_constraints_generation as generation


class FakeMultiTableGenerator:
    config = None
    input_variables = None

    @classmethod
    def from_config(cls, config):
        cls.config = config
        return cls()

    def invoke(self, input_variables):
        type(self).input_variables = input_variables
        constraints = ConstraintsWithSources()
        constraints.data_map["patients.Id"] = ColumnConstraintsWithSources(
            assumptions=[
                AssumptionEntry(
                    text="patients.Id should be complete.",
                    sources=[SourceLocation(file="adapter/case.py", start_line=1, end_line=2)],
                )
            ],
            code=[CodeEntry(suggestion='.isComplete("Id")', validity=True, level="error")],
            table_name="patients",
            column_group="Id",
        )
        return (
            constraints,
            {
                "patients.Id": SourceLocations(
                    [SourceLocation(file="adapter/case.py", start_line=1, end_line=2)]
                )
            },
            {"table_column_access_detection": {"total_tokens": 0, "cost": 0}},
        )


def prepare_inputs(tmp_path: Path) -> Path:
    example_root = write_example(tmp_path)
    write_tables(example_root)
    write_adapter_files(example_root)
    add_adapter_manifest_files(example_root)
    input_prep.main(
        [
            "--project-root",
            str(tmp_path),
            "--example-id",
            "example_a",
            "--script-id",
            "patient_summary",
            "--variant",
            "clean",
            "--overwrite",
        ]
    )
    return (
        tmp_path
        / "data_processed"
        / "eid_bench_real"
        / "example_a"
        / "constraints"
        / "patient_summary"
        / "prismadv_inputs.yaml"
    )


def test_generate_constraints_writes_eid_bench_real_prismadv_artifact_without_llm_or_spark(tmp_path, monkeypatch):
    input_path = prepare_inputs(tmp_path)
    monkeypatch.setattr(
        generation,
        "build_table_validation_inputs",
        lambda project_manager, prismadv_inputs: ({}, {}, None),
    )

    output_path = generation.generate_constraints(
        project_root=tmp_path,
        example_id="example_a",
        script_id="patient_summary",
        input_path=input_path,
        model_name="gpt-5-mini",
        use_async=False,
        use_dataflow=True,
        overwrite=True,
        generator_cls=FakeMultiTableGenerator,
    )

    assert output_path.name.startswith("prismadv_real_etl--gpt-5-mini--")
    raw = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert raw["input_artifact"] == str(input_path)
    assert raw["constraints"]["patients.Id"]["table_name"] == "patients"
    assert raw["constraints"]["patients.Id"]["column_group"] == "Id"
    assert raw["column_data_flow_locations"]["patients.Id"] == [
        {"file": "adapter/case.py", "start_line": 1, "end_line": 2}
    ]
    assert "post_processing" not in raw
    assert FakeMultiTableGenerator.config.model.correlation_detection is True
    assert FakeMultiTableGenerator.config.model.with_assumptions is True
    assert FakeMultiTableGenerator.input_variables["context"]["script"]["reads"] == ["patients", "encounters"]


def test_generate_constraints_propagates_ablation_flags(tmp_path, monkeypatch):
    input_path = prepare_inputs(tmp_path)
    monkeypatch.setattr(
        generation,
        "build_table_validation_inputs",
        lambda project_manager, prismadv_inputs: ({}, {}, None),
    )

    generation.generate_constraints(
        project_root=tmp_path,
        example_id="example_a",
        script_id="patient_summary",
        input_path=input_path,
        model_name="gpt-5-mini",
        use_async=False,
        use_dataflow=False,
        correlation_detection=False,
        with_assumptions=False,
        overwrite=True,
        generator_cls=FakeMultiTableGenerator,
    )

    assert FakeMultiTableGenerator.config.model.use_dataflow is False
    assert FakeMultiTableGenerator.config.model.correlation_detection is False
    assert FakeMultiTableGenerator.config.model.with_assumptions is False
