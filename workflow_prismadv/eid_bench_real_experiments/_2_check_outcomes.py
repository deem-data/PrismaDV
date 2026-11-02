from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import oyaml as yaml

from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils

VOLATILE_KEY_SUFFIXES = ("_tail", "_bytes", "_seconds", "_path", "_paths", "_files", "_dir")
VOLATILE_KEYS = {"figure_files", "model_files", "plots", "elapsed", "timestamp", "command"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example-id", action="append", default=None,
                        help="Example to check (repeatable). Default: every example with error configs.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 when a label is contradicted by the execution outcome.")
    return parser.parse_args(argv)


def strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_volatile(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS and not str(key).endswith(VOLATILE_KEY_SUFFIXES)
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numbers_differ(expected: float, actual: float, tolerance: float) -> bool:
    """Relative difference beyond `tolerance` (exact when tolerance is 0)."""
    if tolerance <= 0:
        return expected != actual
    scale = max(abs(expected), abs(actual))
    return scale > 0 and abs(expected - actual) / scale > tolerance


def changed_paths(expected: Any, actual: Any, prefix: str = "", tolerance: float = 0.0) -> list[str]:
    """Dotted paths at which two JSON documents differ."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        paths = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                paths.append(child)
            else:
                paths.extend(changed_paths(expected[key], actual[key], child, tolerance))
        return paths
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [prefix or "<root>"]
        paths = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            paths.extend(changed_paths(left, right, f"{prefix}[{index}]", tolerance))
        return paths
    if is_number(expected) and is_number(actual):
        return [prefix or "<root>"] if numbers_differ(expected, actual, tolerance) else []
    return [] if expected == actual else [prefix or "<root>"]


def lookup(document: Any, dotted: str) -> Any:
    """`a.b[2].c` -> value, or None when the path does not exist."""
    current = document
    for token in re.findall(r"[^.\[\]]+|\[\d+\]", dotted):
        if token.startswith("["):
            index = int(token[1:-1])
            if not isinstance(current, list) or index >= len(current):
                return None
            current = current[index]
        else:
            if not isinstance(current, dict) or token not in current:
                return None
            current = current[token]
    return current


def outcome_policy(project_manager: MultiTableProjectManager, script_id: str) -> dict[str, Any]:
    """The application's `outcome_check` block, with per-script overrides.

    Keys: `numeric_tolerance` (relative, default 0 = exact),
    `compare_output_files` (default true), `metric_paths` (when given, only
    these summary fields decide drift). An ML pipeline whose every column is
    a model feature declares a tolerance; a pipeline whose output *is* the
    cleaned input keeps the exact file comparison.
    """
    block = dict(project_manager.manifest.get("outcome_check") or {})
    per_script = dict((block.pop("scripts", None) or {}).get(script_id) or {})
    policy = {"numeric_tolerance": 0.0, "compare_output_files": True, "metric_paths": None}
    policy.update({k: v for k, v in block.items() if k in policy})
    policy.update({k: v for k, v in per_script.items() if k in policy})
    return policy


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def script_observation(
    project_manager: MultiTableProjectManager,
    script_id: str,
    label: str,
) -> dict[str, Any]:
    execution_dir = utils.execution_dir(
        project_manager.example_id, script_id, "corrupted", project_manager.project_root, corruption_label=label
    )
    result_path = execution_dir / "run_result.json"
    if not result_path.exists():
        return {"script_id": script_id, "status": "missing"}
    result = load_json(result_path)
    observation: dict[str, Any] = {"script_id": script_id, "status": result.get("status")}
    if result.get("status") != "passed":
        stderr_lines = [line for line in (result.get("stderr") or "").splitlines() if line.strip()]
        observation["error"] = stderr_lines[-1][:300] if stderr_lines else result.get("error")
        return observation

    summary_path = execution_dir / "summary.json"
    expected_path = project_manager.get_expected_clean_summary_path(script_id)
    if not summary_path.exists() or not expected_path.exists():
        observation["drift"] = None
        return observation
    policy = outcome_policy(project_manager, script_id)
    expected_summary = strip_volatile(load_json(expected_path))
    actual_summary = strip_volatile(load_json(summary_path))
    tolerance = float(policy["numeric_tolerance"])
    if policy["metric_paths"]:
        diff = []
        for dotted in policy["metric_paths"]:
            expected_value, actual_value = lookup(expected_summary, dotted), lookup(actual_summary, dotted)
            if changed_paths(expected_value, actual_value, dotted, tolerance):
                diff.append(dotted)
    else:
        diff = changed_paths(expected_summary, actual_summary, tolerance=tolerance)
    changed_files: list[str] = []
    if policy["compare_output_files"]:
        clean_dir = utils.execution_dir(project_manager.example_id, script_id, "clean", project_manager.project_root)
        changed_files = changed_output_files(clean_dir, execution_dir)
    observation["drift"] = bool(diff or changed_files)
    observation["changed"] = diff[:25]
    observation["changed_files"] = changed_files[:25]
    return observation


DATA_SUFFIXES = {".csv", ".json", ".txt", ".parquet", ".tsv"}
RUNNER_FILES = {"run_result.json", "summary.json"}


def output_files(run_dir: Path) -> dict[str, Path]:
    """Data files a run wrote, keyed by relative path.

    Directories starting with `_` (`_warehouse`, `_derby`: Spark and Derby
    internals) are skipped: their part files carry a fresh UUID in the name
    on every run, so comparing them reports drift for any two runs.
    """
    if not run_dir.exists():
        return {}
    files = {}
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in DATA_SUFFIXES or path.name in RUNNER_FILES:
            continue
        relative = path.relative_to(run_dir)
        if any(part.startswith("_") for part in relative.parts):
            continue
        files[str(relative)] = path
    return files


def same_data_file(left: Path, right: Path) -> bool:
    if left.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            return pd.read_parquet(left).equals(pd.read_parquet(right))
        except Exception:
            pass
    return left.read_bytes() == right.read_bytes()


def changed_output_files(clean_dir: Path, corrupted_dir: Path) -> list[str]:
    clean_files = output_files(clean_dir)
    corrupted_files = output_files(corrupted_dir)
    changed = []
    for name in sorted(set(clean_files) | set(corrupted_files)):
        if name not in clean_files or name not in corrupted_files:
            changed.append(name)
        elif not same_data_file(clean_files[name], corrupted_files[name]):
            changed.append(name)
    return changed


def observe_label(project_manager: MultiTableProjectManager, label: str, expected: str | None) -> dict[str, Any]:
    scripts = [script_observation(project_manager, script_id, label) for script_id in project_manager.get_script_ids()]
    statuses = {script["status"] for script in scripts}
    if "missing" in statuses:
        execution = "missing"
    elif statuses == {"passed"}:
        execution = "passed"
    else:
        execution = "failed"
    drift_values = [script.get("drift") for script in scripts if script["status"] == "passed"]
    drift = any(bool(value) for value in drift_values) if execution == "passed" else None

    if execution == "missing":
        observed = None
    elif execution == "failed" or drift:
        observed = "unsafe"
    else:
        observed = "safe"

    if observed is None or expected is None:
        verdict = "not_run"
    elif expected == observed:
        verdict = "consistent"
    elif expected == "safe" and execution == "failed":
        verdict = "label_contradicted_by_execution"
    elif expected == "safe" and drift:
        verdict = "safe_label_with_output_drift"
    elif expected == "unsafe":
        verdict = "unsafe_label_without_observable_effect"
    else:
        verdict = "inconsistent"

    return {
        "label": label,
        "expected_current_outcome": expected,
        "execution": execution,
        "output_drift": drift,
        "observed_outcome": observed,
        "verdict": verdict,
        "scripts": scripts,
    }


def check_example(project_root: Path, example_id: str) -> dict[str, Any]:
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    configs = utils.load_error_configs(project_manager)
    rows = [
        observe_label(project_manager, label, config.get("expected_current_outcome"))
        for label, config in configs.items()
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    report = {"example_id": example_id, "labels": rows, "verdict_counts": counts}
    out_path = utils.outcomes_dir(example_id, project_root, create=True) / "observed_outcomes.yaml"
    out_path.write_text(yaml.dump(report, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return report


def examples_with_errors(project_root: Path) -> list[str]:
    root = project_root / "benchmarks" / "EIDBench-real"
    return sorted(path.name for path in root.iterdir() if (path / "errors").is_dir())


def print_report(report: dict[str, Any]) -> None:
    print(f"\n== {report['example_id']} ==")
    print(f"{'label':<12} {'expected':<8} {'execution':<9} {'drift':<6} {'observed':<8} verdict")
    for row in report["labels"]:
        drift = "-" if row["output_drift"] is None else ("yes" if row["output_drift"] else "no")
        print(f"{row['label']:<12} {str(row['expected_current_outcome']):<8} {row['execution']:<9} "
              f"{drift:<6} {str(row['observed_outcome']):<8} {row['verdict']}")
        if row["verdict"] not in ("consistent", "not_run"):
            for script in row["scripts"]:
                if script.get("error"):
                    print(f"    {script['script_id']}: {script['error']}")
                elif script.get("changed") or script.get("changed_files"):
                    print(f"    {script['script_id']}: summary {script['changed'][:6]} files {script.get('changed_files', [])[:4]}")
    print("counts:", report["verdict_counts"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root).resolve()
    example_ids = args.example_id or examples_with_errors(project_root)
    contradicted = 0
    for example_id in example_ids:
        report = check_example(project_root, example_id)
        print_report(report)
        contradicted += report["verdict_counts"].get("label_contradicted_by_execution", 0)
        contradicted += report["verdict_counts"].get("unsafe_label_without_observable_effect", 0)
    return 1 if (args.strict and contradicted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
