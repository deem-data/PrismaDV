#!/usr/bin/env python3
"""Drive repeated EIDBench-real generation + validation runs for stability analysis.

Iterates the (dataset, method, model) grid that backs `results_analysis/
result_full_approach.ipynb` and, for each cell, makes sure at least
`--target-runs` constraint artifacts exist. Idempotent: cells that already have
enough artifacts are skipped. Validation is run for every artifact (clean +
corruptions); the per-target validator skips outputs that already exist.

Usage (from repo root, inside the poetry env):

    poetry run python -m workflow_prismadv.eid_bench_real_experiments._5_repeat_runs \
        --target-runs 2 \
        --stage all \
        --dataset bank_marketing_analysis  # optional filter

The grid mirrors the notebook so re-aggregation Just Works once new artifacts
land in `data_processed/eid_bench_real/<dataset>/constraints/project/`.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from workflow_prismadv.eid_bench_real_experiments import utils


DATASETS = (
    "bank_marketing_analysis",
    "healthy_diet_dashboard",
    "omop_cdm_databricks",
)

# (method_label, module, model_name, artifact_prefix)
# artifact_prefix matches what the generator's make_output_filename emits.
GRID: tuple[tuple[str, str, str, str], ...] = (
    # Model order within prismadv block matches eid_bench's model_order
    # (gemini-2.5-flash, gpt-4.1, gpt-4o, gpt-5-mini, gemini-2.5-pro, gpt-5).
    ("prismadv",  "workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation", "gemini-2.5-flash", "prismadv_real_etl"),
    ("prismadv",  "workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation", "gpt-4.1",          "prismadv_real_etl"),
    ("prismadv",  "workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation", "gpt-5-mini",       "prismadv_real_etl"),
    ("prismadv",  "workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation", "gemini-2.5-pro",   "prismadv_real_etl"),
    ("prismadv",  "workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation", "gpt-5",            "prismadv_real_etl"),
    # Baselines run on the same model set as prismadv (eid_bench model_order).
    ("zero_shot", "workflow_prismadv.eid_bench_real_baselines.single_shot",                   "gemini-2.5-flash", "single_shot_real_etl"),
    ("zero_shot", "workflow_prismadv.eid_bench_real_baselines.single_shot",                   "gpt-4.1",          "single_shot_real_etl"),
    ("zero_shot", "workflow_prismadv.eid_bench_real_baselines.single_shot",                   "gpt-5-mini",       "single_shot_real_etl"),
    ("zero_shot", "workflow_prismadv.eid_bench_real_baselines.single_shot",                   "gemini-2.5-pro",   "single_shot_real_etl"),
    ("zero_shot", "workflow_prismadv.eid_bench_real_baselines.single_shot",                   "gpt-5",            "single_shot_real_etl"),
    ("few_shot",  "workflow_prismadv.eid_bench_real_baselines.few_shot",                      "gemini-2.5-flash", "few_shot_real_etl"),
    ("few_shot",  "workflow_prismadv.eid_bench_real_baselines.few_shot",                      "gpt-4.1",          "few_shot_real_etl"),
    ("few_shot",  "workflow_prismadv.eid_bench_real_baselines.few_shot",                      "gpt-5-mini",       "few_shot_real_etl"),
    ("few_shot",  "workflow_prismadv.eid_bench_real_baselines.few_shot",                      "gemini-2.5-pro",   "few_shot_real_etl"),
    ("few_shot",  "workflow_prismadv.eid_bench_real_baselines.few_shot",                      "gpt-5",            "few_shot_real_etl"),
    # swe-agent is reported only for gpt-5 (and the existing gemini-2.5-flash),
    # not the full model set like zero/few-shot.
    ("swe_agent", "workflow_prismadv.eid_bench_real_baselines.swe_agent",                     "gemini-2.5-flash", "swe_agent_real_etl"),
    ("swe_agent", "workflow_prismadv.eid_bench_real_baselines.swe_agent",                     "gpt-5",            "swe_agent_real_etl"),
    # deequ is deterministic (no LLM); run with --target-runs 1.
    ("deequ",     "workflow_prismadv.eid_bench_real_baselines.deequ",                         "deequ",            "deequ"),
    # pocketflow agent (gpt-5, like the other agents); reported as a single row.
    ("pocketflow", "workflow_prismadv.eid_bench_real_baselines.pocketflow_agent",            "gpt-5",            "pocketflow"),
)

VALIDATION_MODULE = "workflow_prismadv.eid_bench_real_experiments._4_constraints_validation"
SCRIPT_ID = "project"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-runs", type=int, default=2,
                        help="Desired number of constraint artifacts per (dataset, method, model).")
    parser.add_argument("--stage", choices=("generate", "validate", "all"), default="all")
    parser.add_argument("--dataset", action="append", default=None,
                        help="Restrict to one or more dataset ids (default: all).")
    parser.add_argument("--method", action="append", default=None,
                        help="Restrict to one or more methods (prismadv | zero_shot | few_shot | swe_agent).")
    parser.add_argument("--model", action="append", default=None,
                        help="Restrict to one or more model names.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    return parser.parse_args(argv)


def existing_artifacts(constraints_dir: Path, prefix: str, model_name: str) -> list[Path]:
    return sorted(constraints_dir.glob(f"{prefix}--{model_name}--*.yaml"))


def filter_grid(grid: tuple, methods: list[str] | None, models: list[str] | None) -> tuple:
    out = []
    for entry in grid:
        method, _, model, _ = entry
        if methods and method not in methods:
            continue
        if models and model not in models:
            continue
        out.append(entry)
    return tuple(out)


def run(command: list[str], dry_run: bool) -> int:
    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"\n$ {rendered}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def generate_for_cell(
    *,
    dataset: str,
    module: str,
    model_name: str,
    prefix: str,
    target_runs: int,
    constraints_dir: Path,
    dry_run: bool,
) -> int:
    have = existing_artifacts(constraints_dir, prefix, model_name)
    print(f"\n[{dataset} / {prefix} / {model_name}] {len(have)} artifact(s) already present.")
    for path in have:
        print(f"    - {path.name}")
    if len(have) >= target_runs:
        print(f"  -> target_runs={target_runs} already met; skipping generation.")
        return 0
    missing = target_runs - len(have)
    final_rc = 0
    for i in range(missing):
        print(f"\n  ==> launching new run ({i + 1}/{missing}) for {prefix} / {model_name} on {dataset}")
        command = [
            "poetry", "run", "python", "-m", module,
            "--example-id", dataset,
            "--model-name", model_name,
            "--overwrite",
        ]
        rc = run(command, dry_run)
        if rc != 0:
            print(f"  !! generation returned non-zero exit code {rc}; will not abort the sweep.")
            final_rc = rc
    return final_rc


def validate_artifacts(*, dataset: str, constraints_dir: Path, dry_run: bool) -> int:
    artifacts: list[Path] = []
    for prefix in {entry[3] for entry in GRID}:
        artifacts.extend(constraints_dir.glob(f"{prefix}--*.yaml"))
    artifacts = sorted(set(artifacts))
    if not artifacts:
        print(f"  (no constraint artifacts in {constraints_dir})")
        return 0
    final_rc = 0
    for artifact in artifacts:
        command = [
            "poetry", "run", "python", "-m", VALIDATION_MODULE,
            "--example-id", dataset,
            "--script-id", SCRIPT_ID,
            "--target", "all",
            "--constraint-path", str(artifact),
        ]
        rc = run(command, dry_run)
        if rc != 0:
            print(f"  !! validation returned non-zero exit code {rc} for {artifact.name}")
            final_rc = rc
    return final_rc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    datasets = tuple(args.dataset) if args.dataset else DATASETS
    grid = filter_grid(GRID, args.method, args.model)
    print(
        f"sweep:\n"
        f"  project_root = {project_root}\n"
        f"  datasets     = {list(datasets)}\n"
        f"  grid cells   = {len(grid)} (method/model rows)\n"
        f"  target_runs  = {args.target_runs}\n"
        f"  stage        = {args.stage}\n"
        f"  dry_run      = {args.dry_run}"
    )

    sweep_rc = 0
    if args.stage in ("generate", "all"):
        for dataset in datasets:
            constraints_dir = utils.constraints_dir(dataset, SCRIPT_ID, project_root, create=True)
            for _, module, model_name, prefix in grid:
                rc = generate_for_cell(
                    dataset=dataset,
                    module=module,
                    model_name=model_name,
                    prefix=prefix,
                    target_runs=args.target_runs,
                    constraints_dir=constraints_dir,
                    dry_run=args.dry_run,
                )
                sweep_rc = sweep_rc or rc

    if args.stage in ("validate", "all"):
        for dataset in datasets:
            constraints_dir = utils.constraints_dir(dataset, SCRIPT_ID, project_root, create=True)
            print(f"\n=== validating artifacts in {constraints_dir} ===")
            rc = validate_artifacts(
                dataset=dataset,
                constraints_dir=constraints_dir,
                dry_run=args.dry_run,
            )
            sweep_rc = sweep_rc or rc

    return sweep_rc


if __name__ == "__main__":
    sys.exit(main())
