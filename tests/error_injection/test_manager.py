from pathlib import Path

import pandas as pd

from prismadv.error_injection.corrupts import ForeignKeyViolation
from prismadv.error_injection.managers import MultiTableErrorInjectionManager, TableCorruptionSpec


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_manager_applies_table_scoped_corruptions(tmp_path):
    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    write_csv(clean_root / "patients.csv", [{"Id": "p1"}, {"Id": "p2"}])
    write_csv(
        clean_root / "encounters.csv",
        [
            {"Id": "e1", "PATIENT": "p1"},
            {"Id": "e2", "PATIENT": "p2"},
            {"Id": "e3", "PATIENT": "p1"},
            {"Id": "e4", "PATIENT": "p2"},
        ],
    )

    manager = MultiTableErrorInjectionManager(clean_root, ["patients", "encounters"])
    manager.error_injection(
        [
            TableCorruptionSpec(
                table_name="encounters",
                corruptions=[
                    ForeignKeyViolation(
                        columns=["PATIENT"],
                        reference_values={"PATIENT": ["p1", "p2"]},
                        severity=0.5,
                        random_state=7,
                    )
                ],
            )
        ]
    )

    assert manager.post_corruption_tables is not None
    corrupted_encounters = manager.post_corruption_tables["encounters"]
    assert len(corrupted_encounters) == 4
    assert (~corrupted_encounters["PATIENT"].isin({"p1", "p2"})).sum() == 2

    output_root = tmp_path / "corrupted"
    manager.save_data(output_root, report={"label": "missing_patient_join_key"})

    assert (output_root / "tables" / "encounters.csv").exists()
    assert (output_root / "input" / "encounters.csv").exists()
    config = (output_root / "error_injection_config.yaml").read_text()
    assert "Table: encounters" in config
    assert "ForeignKeyViolation" in config
    assert (output_root / "corruption_report.json").read_text().startswith("{")
