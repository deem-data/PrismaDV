"""Project manager for EIDBench-real examples."""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import oyaml as yaml

from prismadv.utils import get_project_root


class MultiTableProjectManager:
    """Resolve EIDBench-real manifests and benchmark-standard paths."""

    REQUIRED_FIELDS = {
        "example_id",
        "display_name",
        "source_repo",
        "domain",
        "primary_runtime",
        "primary_language",
        "task_type",
        "tables",
        "scripts",
        "input_layout",
        "expected_outputs",
        "clean_should_pass",
        "corruption_mode",
        "notes_on_repairs",
    }

    def __init__(
        self,
        project_root: Optional[Union[Path, str]] = None,
        example_id: Optional[str] = None,
        benchmark_root: Optional[Union[Path, str]] = None,
    ):
        if project_root is None:
            self.project_root = get_project_root()
        else:
            self.project_root = Path(project_root)

        self.benchmark_root = (
            Path(benchmark_root)
            if benchmark_root is not None
            else self.project_root / "benchmarks" / "EIDBench-real"
        )

        if example_id is None:
            raise ValueError("example_id cannot be None")

        self.example_id = example_id
        self.example_path = self.benchmark_root / example_id
        self.manifest_path = self.example_path / "manifest.yaml"

        if not self.example_path.exists():
            raise ValueError(f"EIDBench-real example <{example_id}> does not exist at {self.example_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"EIDBench-real manifest not found: {self.manifest_path}")

        self._manifest = self._load_manifest()
        self._validate_manifest()

    @property
    def manifest(self) -> Dict[str, Any]:
        return self._manifest

    @property
    def tables(self) -> Dict[str, Dict[str, Any]]:
        return self.manifest["tables"]

    @property
    def scripts(self) -> Dict[str, Dict[str, Any]]:
        return self.manifest["scripts"]

    @property
    def input_layout(self) -> Dict[str, str]:
        return self.manifest["input_layout"]

    @classmethod
    def available_examples(cls, project_root: Optional[Union[Path, str]] = None) -> list[str]:
        root = get_project_root() if project_root is None else Path(project_root)
        benchmark_root = root / "benchmarks" / "EIDBench-real"
        if not benchmark_root.exists():
            return []
        return sorted(
            item.name
            for item in benchmark_root.iterdir()
            if item.is_dir() and (item / "manifest.yaml").exists()
        )

    def get_table_names(self) -> list[str]:
        return sorted(self.tables.keys())

    def get_script_ids(self) -> list[str]:
        return sorted(self.scripts.keys())

    def get_table_spec(self, table_name: str) -> Dict[str, Any]:
        if table_name not in self.tables:
            raise KeyError(f"Table <{table_name}> is not declared in {self.manifest_path}")
        return self.tables[table_name]

    def get_script_spec(self, script_id: str) -> Dict[str, Any]:
        if script_id not in self.scripts:
            raise KeyError(f"Script <{script_id}> is not declared in {self.manifest_path}")
        return self.scripts[script_id]

    def resolve_path(self, path: Union[Path, str]) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.example_path / path

    def get_table_path(self, table_name: str, variant: str, corruption_label: Optional[str] = None) -> Path:
        spec = self.get_table_spec(table_name)
        if variant == "observed":
            return self.resolve_path(spec["observed_path"])
        if variant == "clean":
            return self.resolve_path(spec["clean_path"])
        if variant == "corrupted":
            if corruption_label is None:
                raise ValueError("corruption_label is required for corrupted table paths")
            corrupted_paths = spec.get("corrupted_paths", {})
            if corruption_label in corrupted_paths:
                return self.resolve_path(corrupted_paths[corruption_label])
            return self.get_corrupted_tables_dir(corruption_label) / f"{table_name}.{spec['format']}"
        raise ValueError("variant must be one of: observed, clean, corrupted")

    def get_table_paths(
        self,
        variant: str,
        corruption_label: Optional[str] = None,
        table_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Path]:
        names = list(table_names) if table_names is not None else self.get_table_names()
        return {
            table_name: self.get_table_path(table_name, variant, corruption_label)
            for table_name in names
        }

    def get_clean_input_dir(self) -> Path:
        return self.resolve_path(self.input_layout["clean_input_dir"])

    def get_corrupted_input_dir(self, corruption_label: str) -> Path:
        return self.get_corrupted_root_dir() / corruption_label / "input"

    def get_corrupted_tables_dir(self, corruption_label: str) -> Path:
        return self.get_corrupted_root_dir() / corruption_label / "tables"

    def get_corrupted_root_dir(self) -> Path:
        return self.resolve_path(self.input_layout["corrupted_root_dir"])

    def get_script_entrypoint(self, script_id: str) -> Path:
        return self.resolve_path(self.get_script_spec(script_id)["entrypoint"])

    def get_expected_clean_summary_path(self, script_id: str) -> Path:
        if script_id not in self.manifest["expected_outputs"]:
            raise KeyError(f"Expected output for script <{script_id}> is not declared")
        spec = self.manifest["expected_outputs"][script_id]
        return self.resolve_path(spec["clean_summary_path"])

    def _load_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, "r") as f:
            manifest = yaml.safe_load(f)
        if not isinstance(manifest, dict):
            raise ValueError(f"EIDBench-real manifest must be a mapping: {self.manifest_path}")
        return manifest

    def _validate_manifest(self) -> None:
        missing = sorted(self.REQUIRED_FIELDS - self.manifest.keys())
        if missing:
            raise ValueError(f"EIDBench-real manifest is missing required fields: {missing}")
        if self.manifest["example_id"] != self.example_id:
            raise ValueError(
                f"Manifest example_id <{self.manifest['example_id']}> does not match manager example_id "
                f"<{self.example_id}>"
            )
        if not isinstance(self.tables, dict) or not self.tables:
            raise ValueError("EIDBench-real manifest must declare at least one table")
        if not isinstance(self.scripts, dict) or not self.scripts:
            raise ValueError("EIDBench-real manifest must declare at least one script")

        for field in ("observed_tables_dir", "clean_input_dir", "clean_tables_dir", "corrupted_root_dir"):
            if field not in self.input_layout:
                raise ValueError(f"input_layout is missing required field <{field}>")

        for table_name, table_spec in self.tables.items():
            for field in ("format", "observed_path", "clean_path"):
                if field not in table_spec:
                    raise ValueError(f"Table <{table_name}> is missing required field <{field}>")

        for script_id, script_spec in self.scripts.items():
            for field in ("entrypoint", "reads", "writes"):
                if field not in script_spec:
                    raise ValueError(f"Script <{script_id}> is missing required field <{field}>")
            undeclared_tables = sorted(set(script_spec["reads"]) - self.tables.keys())
            if undeclared_tables:
                raise ValueError(f"Script <{script_id}> reads undeclared tables: {undeclared_tables}")
            expected_outputs = self.manifest["expected_outputs"]
            if script_id not in expected_outputs:
                raise ValueError(f"expected_outputs is missing script <{script_id}>")
            if "clean_summary_path" not in expected_outputs[script_id]:
                raise ValueError(f"expected_outputs.<{script_id}> is missing clean_summary_path")
