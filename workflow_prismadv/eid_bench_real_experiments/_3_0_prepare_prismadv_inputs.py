#!/usr/bin/env python3
"""Prepare EIDBench-real inputs for PrismaDV generation.

This skeleton intentionally stops before LLM-based PrismaDV generation. It
materializes the multi-file code context and table-qualified data profiles that
the next EIDBench-real PrismaDV generation pass will consume.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import oyaml as yaml
import pandas as pd
from pandas.api.types import is_numeric_dtype

from prismadv.loader import MultiTableLoader
from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils


DEFAULT_EXAMPLE_ID = "omop_cdm_synthea"
DEFAULT_SCRIPT_ID = "project"
DEFAULT_PROFILE_SAMPLE_VALUES = 5
PROJECT_SCRIPT_ID = "project"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", default=DEFAULT_EXAMPLE_ID)
    parser.add_argument("--script-id", default=DEFAULT_SCRIPT_ID)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--variant",
        choices=sorted(utils.VALID_VARIANTS),
        default="clean",
        help="Table bundle variant to profile.",
    )
    parser.add_argument(
        "--corruption-label",
        default=None,
        help="Required when --variant corrupted.",
    )
    parser.add_argument(
        "--sample-values",
        type=int,
        default=DEFAULT_PROFILE_SAMPLE_VALUES,
        help="Maximum non-null distinct sample values to record per column.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def yaml_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sample_values(series: pd.Series, limit: int) -> list[Any]:
    if limit <= 0:
        return []
    values = []
    seen = set()
    for value in series.dropna():
        safe_value = yaml_safe_value(value)
        marker = repr(safe_value)
        if marker in seen:
            continue
        seen.add(marker)
        values.append(safe_value)
        if len(values) >= limit:
            break
    return values


def numeric_summary(series: pd.Series) -> dict[str, Any] | None:
    if not is_numeric_dtype(series):
        return None
    non_null = series.dropna()
    if non_null.empty:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": yaml_safe_value(non_null.min()),
        "max": yaml_safe_value(non_null.max()),
        "mean": yaml_safe_value(non_null.mean()),
    }


def profile_table(
    table_name: str,
    dataframe: pd.DataFrame,
    path: Path,
    table_spec: dict[str, Any],
    *,
    sample_value_limit: int = DEFAULT_PROFILE_SAMPLE_VALUES,
) -> dict[str, Any]:
    row_count = int(len(dataframe))
    columns = []
    for column_name in dataframe.columns:
        series = dataframe[column_name]
        null_count = int(series.isna().sum())
        column_profile = {
            "name": column_name,
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_fraction": float(null_count / row_count) if row_count else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": sample_values(series, sample_value_limit),
        }
        numeric = numeric_summary(series)
        if numeric is not None:
            column_profile["numeric"] = numeric
        columns.append(column_profile)

    return {
        "table_name": table_name,
        "path": str(path),
        "format": table_spec["format"],
        "row_count": row_count,
        "column_count": int(len(dataframe.columns)),
        "columns": columns,
        "primary_key": table_spec.get("primary_key"),
        "foreign_keys": table_spec.get("foreign_keys", []),
    }


def resolve_code_context_paths(
    project_manager: MultiTableProjectManager,
    script_id: str,
) -> list[Path]:
    script_ids = project_manager.get_script_ids() if is_project_script_id(script_id) else [script_id]
    candidate_paths = [project_manager.get_script_entrypoint(current_script_id) for current_script_id in script_ids]
    for relative_path in project_manager.manifest.get("benchmark_adapter_files", []):
        candidate_paths.append(project_manager.resolve_path(relative_path))

    resolved_paths = []
    seen = set()
    for path in candidate_paths:
        path = path.resolve()
        if is_validator_context_file(project_manager, path):
            continue
        if path in seen:
            continue
        if not path.exists():
            raise FileNotFoundError(f"EIDBench-real code context file not found: {path}")
        seen.add(path)
        resolved_paths.append(path)
    return resolved_paths


def is_project_script_id(script_id: str) -> bool:
    return script_id == PROJECT_SCRIPT_ID


def build_code_context(
    project_manager: MultiTableProjectManager,
    paths: Iterable[Path],
) -> dict[str, Any]:
    files = []
    combined_parts = []
    numbered_parts = []
    for path in paths:
        relative_path = display_code_context_path(project_manager, path)
        text = path.read_text(encoding="utf-8")
        files.append(
            {
                "path": str(relative_path),
                "line_count": len(text.splitlines()),
                "char_count": len(text),
                "content": text,
            }
        )
        combined_parts.append(f"# File: {relative_path}\n{text.rstrip()}\n")
        numbered_parts.append(render_code_file_with_line_numbers(relative_path, text))
    return {
        "files": files,
        "combined": "\n".join(combined_parts).rstrip() + "\n",
        "combined_with_line_numbers": "\n\n".join(numbered_parts).rstrip() + "\n",
    }


def render_code_file_with_line_numbers(path: Path, text: str) -> str:
    rendered = [f"# File: {path}"]
    for line_number, line in enumerate(text.rstrip().splitlines(), start=1):
        rendered.append(f"      {line_number:04}: {line}")
    return "\n".join(rendered)


def is_validator_context_file(project_manager: MultiTableProjectManager, path: Path) -> bool:
    try:
        relative_path = path.relative_to(project_manager.example_path.resolve())
    except ValueError:
        relative_path = path
    return relative_path.name == "validator.py"


def display_code_context_path(project_manager: MultiTableProjectManager, path: Path) -> Path:
    relative_path = path.resolve().relative_to(project_manager.example_path.resolve())
    if relative_path.parts and relative_path.parts[0] == "adapter":
        return Path(*relative_path.parts[1:])
    return relative_path


def build_context_artifact(
    project_manager: MultiTableProjectManager,
    script_id: str,
    variant: str,
    corruption_label: str | None,
    code_context: dict[str, Any],
) -> dict[str, Any]:
    script_context = build_script_context(project_manager, script_id)
    return {
        "example_id": project_manager.example_id,
        "display_name": project_manager.manifest.get("display_name"),
        "domain": project_manager.manifest.get("domain"),
        "primary_runtime": project_manager.manifest.get("primary_runtime"),
        "primary_language": project_manager.manifest.get("primary_language"),
        "task_type": project_manager.manifest.get("task_type"),
        "source_repo": project_manager.manifest.get("source_repo"),
        "source_commit": project_manager.manifest.get("source_commit"),
        "script_id": script_id,
        "variant": variant,
        "corruption_label": corruption_label,
        "script": script_context,
        "code_context_files": [
            {
                "path": file_info["path"],
                "line_count": file_info["line_count"],
                "char_count": file_info["char_count"],
            }
            for file_info in code_context["files"]
        ],
    }


def build_script_context(project_manager: MultiTableProjectManager, script_id: str) -> dict[str, Any]:
    if not is_project_script_id(script_id):
        script_spec = project_manager.get_script_spec(script_id)
        return {
            "entrypoint": script_spec["entrypoint"],
            "reads": list(script_spec["reads"]),
            "writes": list(script_spec["writes"]),
            "timeout_seconds": script_spec.get("timeout_seconds"),
        }

    scripts = []
    reads = []
    writes = []
    for current_script_id in project_manager.get_script_ids():
        script_spec = project_manager.get_script_spec(current_script_id)
        current_reads = list(script_spec["reads"])
        current_writes = list(script_spec["writes"])
        scripts.append(
            {
                "script_id": current_script_id,
                "entrypoint": script_spec["entrypoint"],
                "reads": current_reads,
                "writes": current_writes,
                "depends_on": list(script_spec.get("depends_on", [])),
                "timeout_seconds": script_spec.get("timeout_seconds"),
            }
        )
        reads.extend(current_reads)
        writes.extend(current_writes)

    return {
        "entrypoint": None,
        "reads": sorted(set(reads)),
        "writes": sorted(set(writes)),
        "timeout_seconds": None,
        "scripts": scripts,
    }


def build_prismadv_inputs(
    project_manager: MultiTableProjectManager,
    script_id: str,
    *,
    variant: str = "clean",
    corruption_label: str | None = None,
    sample_value_limit: int = DEFAULT_PROFILE_SAMPLE_VALUES,
) -> dict[str, Any]:
    utils.variant_component(variant, corruption_label)
    script_context = build_script_context(project_manager, script_id)
    table_names = list(script_context["reads"])

    loader = MultiTableLoader(project_manager)
    bundle = loader.load_pandas(
        variant=variant,
        corruption_label=corruption_label,
        table_names=table_names,
    )

    code_paths = resolve_code_context_paths(
        project_manager,
        script_id,
    )
    code_context = build_code_context(project_manager, code_paths)
    table_profiles = {
        table_name: profile_table(
            table_name,
            bundle.tables[table_name],
            bundle.paths[table_name],
            project_manager.get_table_spec(table_name),
            sample_value_limit=sample_value_limit,
        )
        for table_name in table_names
    }
    context = build_context_artifact(
        project_manager,
        script_id,
        variant,
        corruption_label,
        code_context,
    )
    return {
        "context": context,
        "code_context": code_context,
        "table_profiles": table_profiles,
    }


def write_yaml(path: Path, data: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"artifact already exists: {path}")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def write_prismadv_input_artifacts(
    output_dir: Path,
    prismadv_inputs: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "context": output_dir / "eid_bench_real_context.yaml",
        "table_profiles": output_dir / "table_profiles.yaml",
        "prismadv_inputs": output_dir / "prismadv_inputs.yaml",
    }
    write_yaml(paths["context"], prismadv_inputs["context"], overwrite=overwrite)
    write_yaml(paths["table_profiles"], prismadv_inputs["table_profiles"], overwrite=overwrite)
    write_yaml(paths["prismadv_inputs"], prismadv_inputs, overwrite=overwrite)
    return paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root).resolve()
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=args.example_id)
    prismadv_inputs = build_prismadv_inputs(
        project_manager,
        args.script_id,
        variant=args.variant,
        corruption_label=args.corruption_label,
        sample_value_limit=args.sample_values,
    )
    output_dir = utils.constraints_dir(
        args.example_id,
        args.script_id,
        project_root,
        create=True,
    )
    paths = write_prismadv_input_artifacts(output_dir, prismadv_inputs, overwrite=args.overwrite)
    print(
        yaml.safe_dump(
            {
                "example_id": args.example_id,
                "script_id": args.script_id,
                "variant": args.variant,
                "corruption_label": args.corruption_label,
                "output_dir": str(output_dir),
                "artifacts": {name: str(path) for name, path in paths.items()},
                "tables": list(prismadv_inputs["table_profiles"].keys()),
                "code_context_files": [
                    file_info["path"]
                    for file_info in prismadv_inputs["context"]["code_context_files"]
                ],
            },
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
