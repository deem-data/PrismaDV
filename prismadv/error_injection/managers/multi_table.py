import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import oyaml as yaml
import pandas as pd

from prismadv.error_injection.abstract_corruption import DataCorruption
from prismadv.error_injection.abstract_error_injection_manager import AbstractErrorInjectionManager


@dataclass
class TableCorruptionSpec:
    table_name: str
    corruptions: list[DataCorruption]


class MultiTableErrorInjectionManager(AbstractErrorInjectionManager):
    """Apply existing tabular corruptions to a multi-table table set."""

    def __init__(self, clean_tables_dir: Path, table_names: list[str] | None = None):
        self.clean_tables_dir = Path(clean_tables_dir)
        self.table_names = table_names
        self.clean_tables = self.load_data()
        self.post_corruption_tables: dict[str, pd.DataFrame] | None = None
        self.table_specs: list[TableCorruptionSpec] = []

    def load_data(self) -> dict[str, pd.DataFrame]:
        if not self.clean_tables_dir.exists():
            raise FileNotFoundError(f"clean tables directory does not exist: {self.clean_tables_dir}")

        table_names = self.table_names
        if table_names is None:
            table_names = sorted(path.stem for path in self.clean_tables_dir.glob("*.csv"))
        if not table_names:
            raise ValueError(f"no table CSVs found in {self.clean_tables_dir}")

        tables = {}
        for table_name in table_names:
            path = self.clean_tables_dir / f"{table_name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"missing clean table: {path}")
            tables[table_name] = pd.read_csv(path)
        return tables

    def error_injection(self, corrupts: list[TableCorruptionSpec]) -> None:
        post_corruption_tables = {
            table_name: table.copy(deep=True) for table_name, table in self.clean_tables.items()
        }
        for table_spec in corrupts:
            if table_spec.table_name not in post_corruption_tables:
                raise ValueError(f"unknown table for corruption: {table_spec.table_name}")
            table = post_corruption_tables[table_spec.table_name]
            for corruption in table_spec.corruptions:
                table = corruption.transform(table)
            post_corruption_tables[table_spec.table_name] = table

        self.post_corruption_tables = post_corruption_tables
        self.table_specs = corrupts

    def save_data(
        self,
        output_root: Path,
        *,
        overwrite: bool = False,
        include_input_copy: bool = True,
        report: dict | None = None,
    ) -> None:
        if self.post_corruption_tables is None:
            raise ValueError("call error_injection before save_data")

        output_root = Path(output_root)
        if output_root.exists():
            if overwrite:
                shutil.rmtree(output_root)
            elif any(output_root.iterdir()):
                raise FileExistsError(f"output directory already exists: {output_root}")

        tables_root = output_root / "tables"
        tables_root.mkdir(parents=True, exist_ok=True)
        input_root = output_root / "input"
        if include_input_copy:
            input_root.mkdir(parents=True, exist_ok=True)

        for table_name, table in self.post_corruption_tables.items():
            table.to_csv(tables_root / f"{table_name}.csv", index=False)
            if include_input_copy:
                table.to_csv(input_root / f"{table_name}.csv", index=False)

        self.save_error_injection_config(output_root, self.table_specs)
        if report is not None:
            (output_root / "corruption_report.json").write_text(json.dumps(report, indent=2) + "\n")

    def save_error_injection_config(self, output_root: Path, corrupts: list[TableCorruptionSpec]) -> None:
        config_path = Path(output_root) / "error_injection_config.yaml"
        config_path.write_text(self.table_corrupts_to_yaml(corrupts))

    def table_corrupts_to_yaml(self, corrupts: list[TableCorruptionSpec]) -> str:
        config = []
        for table_spec in corrupts:
            config.append(
                {
                    "Table": table_spec.table_name,
                    "Corruptions": [corruption.to_dict() for corruption in table_spec.corruptions],
                }
            )
        return yaml.dump(config, default_flow_style=False)
