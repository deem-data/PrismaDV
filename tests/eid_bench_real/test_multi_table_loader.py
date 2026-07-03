from pathlib import Path

import pandas as pd
import pytest

from prismadv.loader import MultiTableLoader
from prismadv.project_manager import MultiTableProjectManager
from tests.eid_bench_real.test_project_manager import write_example


def write_tables(example_root: Path) -> None:
    (example_root / "files" / "observed" / "tables" / "patients.csv").write_text(
        "Id,name\np1,Ada\np2,Lin\n"
    )
    (example_root / "files" / "observed" / "tables" / "encounters.csv").write_text(
        "Id,PATIENT\nenc1,p1\nenc2,p2\n"
    )
    (example_root / "files" / "clean" / "tables" / "patients.csv").write_text(
        "Id,name\np1,Ada\np2,Lin\np3,Grace\n"
    )
    (example_root / "files" / "clean" / "tables" / "encounters.csv").write_text(
        "Id,PATIENT\nenc1,p1\nenc2,p2\nenc3,p3\n"
    )
    (example_root / "files" / "corrupted" / "missing_key" / "tables" / "patients.csv").write_text(
        "Id,name\np1,Ada\np2,Lin\np3,Grace\n"
    )
    (example_root / "files" / "corrupted" / "missing_key" / "tables" / "encounters_bad.csv").write_text(
        "Id,PATIENT\nenc1,p1\nenc2,\nenc3,p3\n"
    )


def test_multi_table_loader_loads_named_pandas_tables(tmp_path):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")
    loader = MultiTableLoader(manager)

    bundle = loader.load_pandas("observed")

    assert bundle.example_id == "example_a"
    assert bundle.variant == "observed"
    assert bundle.corruption_label is None
    assert sorted(bundle.tables.keys()) == ["encounters", "patients"]
    pd.testing.assert_frame_equal(
        bundle.tables["patients"],
        pd.DataFrame({"Id": ["p1", "p2"], "name": ["Ada", "Lin"]}),
    )
    assert bundle.paths["patients"] == example_root / "files" / "observed" / "tables" / "patients.csv"


def test_multi_table_loader_loads_selected_corrupted_table(tmp_path):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")
    loader = MultiTableLoader(manager)

    bundle = loader.load_pandas(
        variant="corrupted",
        corruption_label="missing_key",
        table_names=["encounters"],
    )

    assert list(bundle.tables.keys()) == ["encounters"]
    assert bundle.corruption_label == "missing_key"
    assert bundle.tables["encounters"]["PATIENT"].isna().sum() == 1
    assert bundle.paths["encounters"] == (
        example_root / "files" / "corrupted" / "missing_key" / "tables" / "encounters_bad.csv"
    )


def test_multi_table_loader_rejects_non_csv_pandas_table(tmp_path):
    example_root = write_example(tmp_path)
    write_tables(example_root)
    manifest_path = example_root / "manifest.yaml"
    manifest_text = manifest_path.read_text().replace("format: csv", "format: json", 1)
    manifest_path.write_text(manifest_text)

    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")
    loader = MultiTableLoader(manager)

    with pytest.raises(NotImplementedError, match="supports CSV tables only"):
        loader.load_pandas("observed", table_names=["patients"])


def test_multi_table_loader_reports_missing_table_file(tmp_path):
    write_example(tmp_path)
    manager = MultiTableProjectManager(project_root=tmp_path, example_id="example_a")
    loader = MultiTableLoader(manager)

    with pytest.raises(FileNotFoundError, match="table file not found"):
        loader.load_pandas("observed", table_names=["patients"])
