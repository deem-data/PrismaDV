from pathlib import Path

import oyaml as yaml

from prismadv.project_manager import MultiTableProjectManager
from tests.eid_bench_real.test_multi_table_loader import write_tables
from tests.eid_bench_real.test_project_manager import write_example
from workflow_prismadv.eid_bench_real_experiments import _3_0_prepare_prismadv_inputs as input_prep


def write_adapter_files(example_root: Path) -> None:
    (example_root / "pyproject.toml").write_text(
        "[project]\nname = \"example-a\"\n",
        encoding="utf-8",
    )
    (example_root / "adapter" / "run_script.py").write_text(
        "from adapter.case import run\n\nrun()\n",
        encoding="utf-8",
    )
    (example_root / "adapter" / "case.py").write_text(
        "def run():\n    return 'patient_summary'\n",
        encoding="utf-8",
    )
    (example_root / "adapter" / "validator.py").write_text(
        "def validate():\n    return True\n",
        encoding="utf-8",
    )


def add_adapter_manifest_files(example_root: Path) -> None:
    manifest_path = example_root / "manifest.yaml"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_text = manifest_text.replace(
        "notes_on_repairs: Local adapter only.\n",
        (
            "notes_on_repairs: Local adapter only.\n"
            "benchmark_adapter_files:\n"
            "  - adapter/case.py\n"
            "  - adapter/run_script.py\n"
            "  - adapter/validator.py\n"
            "  - pyproject.toml\n"
        ),
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")


def test_build_prismadv_inputs_profiles_only_script_read_tables(tmp_path):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    write_adapter_files(example_root)
    add_adapter_manifest_files(example_root)
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")

    inputs = input_prep.build_prismadv_inputs(manager, "patient_summary", variant="clean")

    assert inputs["context"]["example_id"] == "example_a"
    assert inputs["context"]["script"]["reads"] == ["patients", "encounters"]
    assert sorted(inputs["table_profiles"]) == ["encounters", "patients"]
    assert inputs["table_profiles"]["patients"]["table_name"] == "patients"
    assert inputs["table_profiles"]["patients"]["row_count"] == 3
    assert inputs["table_profiles"]["patients"]["primary_key"] == "Id"
    assert [column["name"] for column in inputs["table_profiles"]["patients"]["columns"]] == ["Id", "name"]


def test_build_prismadv_inputs_project_mode_profiles_all_project_read_tables(tmp_path):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    write_adapter_files(example_root)
    add_adapter_manifest_files(example_root)
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")

    inputs = input_prep.build_prismadv_inputs(manager, "project", variant="clean")

    assert inputs["context"]["script_id"] == "project"
    assert inputs["context"]["script"]["entrypoint"] is None
    assert inputs["context"]["script"]["reads"] == ["encounters", "patients"]
    assert inputs["context"]["script"]["scripts"] == [
        {
            "script_id": "patient_summary",
            "entrypoint": "adapter/run_script.py",
            "reads": ["patients", "encounters"],
            "writes": ["summary.json"],
            "depends_on": [],
            "timeout_seconds": 60,
        }
    ]
    assert sorted(inputs["table_profiles"]) == ["encounters", "patients"]


def test_code_context_excludes_validator_by_default(tmp_path):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    write_adapter_files(example_root)
    add_adapter_manifest_files(example_root)
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")

    paths = input_prep.resolve_code_context_paths(manager, "patient_summary")

    assert [str(path.relative_to(example_root)) for path in paths] == [
        "adapter/run_script.py",
        "adapter/case.py",
        "pyproject.toml",
    ]


def test_code_context_paths_hide_adapter_prefix(tmp_path):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    write_adapter_files(example_root)
    add_adapter_manifest_files(example_root)
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")

    paths = input_prep.resolve_code_context_paths(manager, "patient_summary")
    code_context = input_prep.build_code_context(manager, paths)

    assert [file_info["path"] for file_info in code_context["files"]] == [
        "run_script.py",
        "case.py",
        "pyproject.toml",
    ]
    assert "# File: run_script.py\n      0001: from adapter.case import run" in code_context[
        "combined_with_line_numbers"
    ]
    assert "# File: case.py\n      0001: def run():" in code_context["combined_with_line_numbers"]


def test_write_prismadv_input_artifacts_uses_constraints_directory(tmp_path):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    write_adapter_files(example_root)
    add_adapter_manifest_files(example_root)

    result = input_prep.main(
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

    assert result == 0
    output_dir = tmp_path / "data_processed" / "eid_bench_real" / "example_a" / "constraints" / "patient_summary"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "eid_bench_real_context.yaml",
        "prismadv_inputs.yaml",
        "table_profiles.yaml",
    ]
    table_profiles = yaml.safe_load((output_dir / "table_profiles.yaml").read_text(encoding="utf-8"))
    assert sorted(table_profiles) == ["encounters", "patients"]
    full_inputs = yaml.safe_load((output_dir / "prismadv_inputs.yaml").read_text(encoding="utf-8"))
    assert "validator.py" not in full_inputs["code_context"]["combined"]
    assert "# File: run_script.py" in full_inputs["code_context"]["combined"]
    assert "# File: pyproject.toml" in full_inputs["code_context"]["combined"]
    assert "combined_with_line_numbers" in full_inputs["code_context"]
    assert "      0001: from adapter.case import run" in full_inputs["code_context"]["combined_with_line_numbers"]


def test_write_prismadv_input_artifacts_supports_project_mode(tmp_path):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    write_adapter_files(example_root)
    add_adapter_manifest_files(example_root)

    result = input_prep.main(
        [
            "--project-root",
            str(tmp_path),
            "--example-id",
            "example_a",
            "--script-id",
            "project",
            "--variant",
            "clean",
            "--overwrite",
        ]
    )

    assert result == 0
    output_dir = tmp_path / "data_processed" / "eid_bench_real" / "example_a" / "constraints" / "project"
    full_inputs = yaml.safe_load((output_dir / "prismadv_inputs.yaml").read_text(encoding="utf-8"))
    assert full_inputs["context"]["script_id"] == "project"
    assert full_inputs["context"]["script"]["reads"] == ["encounters", "patients"]
