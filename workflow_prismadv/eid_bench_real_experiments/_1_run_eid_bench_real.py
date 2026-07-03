#!/usr/bin/env python3
"""Run EIDBench-real adapter scripts and record execution outcomes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils


ALL_SCRIPT_ID = "all"
VALID_VARIANTS = {"observed", "clean", "corrupted"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--script-id", default=ALL_SCRIPT_ID)
    parser.add_argument("--variant", choices=sorted(VALID_VARIANTS), default="clean")
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Corruption label to run. May be repeated. If omitted for corrupted runs, all labels are used.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def script_ids(project_manager: MultiTableProjectManager, script_id: str) -> list[str]:
    if script_id == ALL_SCRIPT_ID:
        return project_manager.get_script_ids()
    project_manager.get_script_spec(script_id)
    return [script_id]


def variant_inputs(
    project_manager: MultiTableProjectManager,
    variant: str,
    labels: list[str] | None,
) -> list[tuple[str, str | None, Path]]:
    if variant == "clean":
        return [("clean", None, project_manager.get_clean_input_dir())]
    if variant == "observed":
        return [("observed", None, project_manager.resolve_path(project_manager.input_layout["observed_tables_dir"]))]
    if variant == "corrupted":
        selected_labels = labels if labels is not None else utils.corruption_labels(project_manager)
        if not selected_labels:
            raise ValueError("no corruption labels available")
        return [
            ("corrupted", label, project_manager.get_corrupted_input_dir(label))
            for label in selected_labels
        ]
    raise ValueError(f"variant must be one of {sorted(VALID_VARIANTS)}")


def adapter_command(
    project_manager: MultiTableProjectManager,
    script_id: str,
    input_dir: Path,
    output_dir: Path,
) -> list[str]:
    script_spec = project_manager.get_script_spec(script_id)
    entrypoint = project_manager.resolve_path(script_spec["entrypoint"])
    command_prefix = project_manager.manifest.get("environment", {}).get("command_prefix")
    if command_prefix:
        command = list(command_prefix) + ["python", str(entrypoint)]
    else:
        command = [sys.executable, str(entrypoint)]
    command.extend(
        [
            "--script-id",
            script_id,
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
        ]
    )
    return command


def output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_adapter_script(
    project_manager: MultiTableProjectManager,
    script_id: str,
    variant: str,
    corruption_label: str | None,
    input_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = utils.execution_dir(
        project_manager.example_id,
        script_id,
        variant,
        project_manager.project_root,
        corruption_label=corruption_label,
        create=True,
    )
    result_path = output_dir / "run_result.json"
    if result_path.exists() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["skipped_existing_output"] = True
        return result

    command = adapter_command(project_manager, script_id, input_dir, output_dir)
    timeout_seconds = project_manager.get_script_spec(script_id).get("timeout_seconds")
    started = time.perf_counter()
    status = "failed"
    returncode = None
    stdout = ""
    stderr = ""
    error_type = None
    error = None

    try:
        completed = subprocess.run(
            command,
            cwd=project_manager.project_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        status = "passed" if completed.returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout = output_text(exc.stdout)
        stderr = output_text(exc.stderr)
        error_type = exc.__class__.__name__
        error = str(exc)

    elapsed = time.perf_counter() - started
    summary_path = output_dir / "summary.json"
    result = {
        "example_id": project_manager.example_id,
        "script_id": script_id,
        "variant": variant,
        "corruption_label": corruption_label,
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 3),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        "error_type": error_type,
        "error": error,
        "skipped_existing_output": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_eid_bench_real(
    *,
    project_root: Path,
    example_id: str,
    script_id: str = ALL_SCRIPT_ID,
    variant: str = "clean",
    labels: list[str] | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    selected_script_ids = script_ids(project_manager, script_id)
    selected_inputs = variant_inputs(project_manager, variant, labels)

    results = []
    for current_variant, corruption_label, input_dir in selected_inputs:
        for current_script_id in selected_script_ids:
            result = run_adapter_script(
                project_manager,
                current_script_id,
                current_variant,
                corruption_label,
                input_dir,
                overwrite=overwrite,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "script_id": current_script_id,
                        "variant": current_variant,
                        "corruption_label": corruption_label,
                        "status": result["status"],
                        "output_dir": result["output_dir"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root).resolve()
    results = run_eid_bench_real(
        project_root=project_root,
        example_id=args.example_id,
        script_id=args.script_id,
        variant=args.variant,
        labels=args.label,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "example_id": args.example_id,
                "result_count": len(results),
                "passed": sum(result["status"] == "passed" for result in results),
                "failed": sum(result["status"] == "failed" for result in results),
                "timeout": sum(result["status"] == "timeout" for result in results),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
