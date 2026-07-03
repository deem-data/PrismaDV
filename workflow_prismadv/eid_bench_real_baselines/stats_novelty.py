#!/usr/bin/env python3
"""Statistical-novelty (MMD) task-agnostic baseline for EIDBench-real."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from prismadv.data_models import ValidationResults
from prismadv.loader import MultiTableLoader
from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils
from workflow_prismadv.eid_bench_real_experiments._4_constraints_validation import (
    read_table_names_for_script,
    validation_summary,
)
from workflow_prismadv.eid_bench_real_baselines.single_shot import write_yaml
from workflow_prismadv.task_agnostic_baselines.novelty_models import (
    should_be_rejected_stats_novelty,
)

DEFAULT_EXAMPLE_ID = "omop_cdm_synthea"
PROJECT_SCRIPT_ID = "project"
BASELINE_METHOD = "stats_novelty"
MODEL_TAG = "stats_novelty"
NOVELTY_COLUMN = "__novelty__"
SUGGESTION = "stats_novelty.distribution_unchanged"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", default=DEFAULT_EXAMPLE_ID)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_TAG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def make_output_filename() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{BASELINE_METHOD}--{MODEL_TAG}--{timestamp}.yaml"


def existing_output(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob(f"{BASELINE_METHOD}--{MODEL_TAG}--*.yaml"))
    return candidates[-1] if candidates else None


def _cleanup_previous_runs(project_root: Path, example_id: str, output_dir: Path) -> None:
    """Remove prior artifacts so a deterministic re-run leaves a single stem."""
    for artifact in output_dir.glob(f"{BASELINE_METHOD}--{MODEL_TAG}--*.yaml"):
        artifact.unlink()
    for variant in ("clean", "observed", "corrupted"):
        base = utils.example_processed_root(example_id, project_root) / "constraints_validation" / PROJECT_SCRIPT_ID / variant
        if base.exists():
            for stale in base.rglob(f"validation_results__{BASELINE_METHOD}--{MODEL_TAG}--*.yaml"):
                stale.unlink()


def _table_validation_result(rejected: bool) -> ValidationResults:
    """One synthetic error-level entry per table: passes unless its test rejects."""
    reason = (
        "stats-novelty MMD test rejected (column-stats distribution shifted)"
        if rejected
        else ""
    )
    return ValidationResults.from_dict({
        NOVELTY_COLUMN: {
            "code": [{
                "suggestion": SUGGESTION,
                "status": not rejected,
                "reason_if_failed": reason,
                "level": "error",
            }]
        }
    })


def _validate_bundle(
    project_manager: MultiTableProjectManager,
    references: dict[str, pd.DataFrame],
    table_names: list[str],
    *,
    variant: str,
    corruption_label: str | None,
) -> dict[str, Any]:
    bundle = MultiTableLoader(project_manager).load_pandas(
        variant=variant,
        corruption_label=corruption_label,
        table_names=table_names,
    )
    table_results: dict[str, dict] = {}
    table_summaries: dict[str, dict[str, int]] = {}
    for table_name in table_names:
        rejected = should_be_rejected_stats_novelty(
            references[table_name], bundle.tables[table_name]
        )
        result = _table_validation_result(rejected)
        table_results[table_name] = result.to_dict()["results"]
        table_summaries[table_name] = validation_summary(result)

    totals: dict[str, int] = defaultdict(int)
    for summary in table_summaries.values():
        for key, value in summary.items():
            totals[key] += value

    return {
        "example_id": project_manager.example_id,
        "script_id": PROJECT_SCRIPT_ID,
        "variant": variant,
        "corruption_label": corruption_label,
        "baseline_method": BASELINE_METHOD,
        "model_name": MODEL_TAG,
        "tables": {name: table_results[name] for name in sorted(table_results)},
        "summary_by_table": table_summaries,
        "summary": dict(totals),
    }


def run_stats_novelty(
    *,
    project_root: Path,
    example_id: str,
    overwrite: bool = False,
) -> Path:
    project_root = project_root.resolve()
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    output_dir = utils.constraints_dir(example_id, PROJECT_SCRIPT_ID, project_root, create=True)
    existing = existing_output(output_dir)
    if existing is not None and not overwrite:
        print(f"Stats-novelty EIDBench-real baseline already exists. Skipping: {existing}")
        return existing
    if overwrite:
        _cleanup_previous_runs(project_root, example_id, output_dir)

    table_names = read_table_names_for_script(project_manager, PROJECT_SCRIPT_ID)

    clean_bundle = MultiTableLoader(project_manager).load_pandas(
        variant="clean", table_names=table_names
    )
    references = {table_name: clean_bundle.tables[table_name] for table_name in table_names}

    output_path = output_dir / make_output_filename()
    stub = {"baseline_method": BASELINE_METHOD, "model_name": MODEL_TAG, "constraints": {}}
    write_yaml(output_path, stub)
    stem = output_path.stem

    targets: list[tuple[str, str | None]] = [("clean", None)]
    targets += [("corrupted", label) for label in utils.corruption_labels(project_manager)]
    for variant, corruption_label in targets:
        result = _validate_bundle(
            project_manager,
            references,
            table_names,
            variant=variant,
            corruption_label=corruption_label,
        )
        result["constraint_artifact"] = str(output_path)
        validation_dir = utils.constraints_validation_dir(
            example_id,
            PROJECT_SCRIPT_ID,
            variant,
            project_root,
            corruption_label=corruption_label,
            create=True,
        )
        validation_path = validation_dir / f"validation_results__{stem}.yaml"
        write_yaml(validation_path, result)
        target_desc = corruption_label or variant
        print(f"  [{target_desc}] failed_error={result['summary'].get('failed_error', 0)} -> {validation_path}")

    print(f"example_id: {example_id}")
    print(f"output_path: {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    run_stats_novelty(
        project_root=project_root,
        example_id=args.example_id,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
