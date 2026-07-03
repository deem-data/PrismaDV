"""Load EIDBench-real named table bundles."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class TableSet:
    example_id: str
    variant: str
    corruption_label: Optional[str]
    tables: Dict[str, pd.DataFrame]
    paths: Dict[str, Path]


class MultiTableLoader:
    """Load named EIDBench-real tables as pandas or Spark dataframes."""

    def __init__(self, project_manager: Any):
        self.project_manager = project_manager

    def load_pandas(
        self,
        variant: str,
        corruption_label: Optional[str] = None,
        table_names: Optional[Iterable[str]] = None,
        **read_csv_kwargs,
    ) -> TableSet:
        paths = self.project_manager.get_table_paths(
            variant=variant,
            corruption_label=corruption_label,
            table_names=table_names,
        )
        tables = {
            table_name: self._load_pandas_table(table_name, path, **read_csv_kwargs)
            for table_name, path in paths.items()
        }
        return TableSet(
            example_id=self.project_manager.example_id,
            variant=variant,
            corruption_label=corruption_label,
            tables=tables,
            paths=paths,
        )

    def load_spark(
        self,
        spark,
        variant: str,
        corruption_label: Optional[str] = None,
        table_names: Optional[Iterable[str]] = None,
        **csv_options,
    ) -> Dict[str, object]:
        paths = self.project_manager.get_table_paths(
            variant=variant,
            corruption_label=corruption_label,
            table_names=table_names,
        )
        dataframes = {}
        for table_name, path in paths.items():
            spec = self.project_manager.get_table_spec(table_name)
            if spec["format"] != "csv":
                raise NotImplementedError(
                    f"Spark loading currently supports CSV tables only; table <{table_name}> has format "
                    f"<{spec['format']}>"
                )
            options = {"header": True, "inferSchema": True}
            options.update(csv_options)
            dataframes[table_name] = spark.read.options(**options).csv(str(path))
        return dataframes

    def _load_pandas_table(self, table_name: str, path: Path, **read_csv_kwargs) -> pd.DataFrame:
        spec = self.project_manager.get_table_spec(table_name)
        if spec["format"] != "csv":
            raise NotImplementedError(
                f"Pandas loading currently supports CSV tables only; table <{table_name}> has format "
                f"<{spec['format']}>"
            )
        if not path.exists():
            raise FileNotFoundError(f"EIDBench-real table file not found: {path}")
        return pd.read_csv(path, **read_csv_kwargs)
