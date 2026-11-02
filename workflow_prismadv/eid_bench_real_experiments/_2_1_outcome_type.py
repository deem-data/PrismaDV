#!/usr/bin/env python3
"""Classify every EIDBench-real label as safe, silent failure or crash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import oyaml as yaml

from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils
from workflow_prismadv.eid_bench_real_experiments._2_check_outcomes import examples_with_errors

OUTCOMES = ("safe", "silent", "crash", "not_run")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example-id", action="append", default=None,
                        help="Example to classify (repeatable). Default: every example with error configs.")
    parser.add_argument("--project-root", type=Path, default=None)
    return parser.parse_args(argv)


def script_results(project_manager: MultiTableProjectManager, label: str) -> list[dict]:
    results = []
    for script_id in project_manager.get_script_ids():
        path = utils.execution_dir(project_manager.example_id, script_id, "corrupted", project_manager.project_root,
                                   corruption_label=label) / "run_result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        results.append({"script_id": script_id, "status": result.get("status"),
                        "error_type": result.get("error_type")})
    return results


def classify_label(project_manager: MultiTableProjectManager, label: str, expected: str | None) -> dict:
    scripts = script_results(project_manager, label)
    failed = [s for s in scripts if s["status"] != "passed"]
    if expected == "safe":
        outcome = "safe"
    elif not scripts:
        outcome = "not_run"
    else:
        outcome = "crash" if failed else "silent"
    row = {"label": label, "expected_current_outcome": expected, "outcome": outcome,
           "scripts_run": len(scripts), "scripts_failed": [s["script_id"] for s in failed]}
    if failed:
        row["error_types"] = sorted({str(s["error_type"]) for s in failed})
    return row


def classify_example(project_root: Path, example_id: str) -> dict:
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    configs = utils.load_error_configs(project_manager)
    rows = [classify_label(project_manager, label, config.get("expected_current_outcome"))
            for label, config in configs.items()]
    counts = {outcome: sum(row["outcome"] == outcome for row in rows) for outcome in OUTCOMES}
    report = {"example_id": example_id, "outcome_counts": counts, "labels": rows}
    out_path = utils.outcomes_dir(example_id, project_root, create=True) / "outcome_types.yaml"
    out_path.write_text(yaml.dump(report, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root).resolve()
    for example_id in args.example_id or examples_with_errors(project_root):
        report = classify_example(project_root, example_id)
        print(f"{example_id}: {report['outcome_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
