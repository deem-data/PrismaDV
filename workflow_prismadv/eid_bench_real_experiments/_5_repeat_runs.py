#!/usr/bin/env python3
"""Drive repeated EIDBench-real generation + validation runs for stability analysis.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from workflow_prismadv.eid_bench_real_experiments import utils
from workflow_prismadv.utils.replicates import parse_replicate


DATASETS = (
    "bank_marketing_analysis",
    "healthy_diet_dashboard",
    "omop_cdm_databricks",
)

GRID: tuple[tuple[str, str, str, str], ...] = (
    ("prismadv",  "workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation", "gemini-2.5-flash", "prismadv"),
    ("prismadv",  "workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation", "gpt-4.1",          "prismadv"),
    ("prismadv",  "workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation", "gpt-5-mini",       "prismadv"),
    ("prismadv",  "workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation", "gemini-2.5-pro",   "prismadv"),
    ("prismadv",  "workflow_prismadv.eid_bench_real_experiments._3_1_constraints_generation", "gpt-5",            "prismadv"),
    ("zero_shot", "workflow_prismadv.eid_bench_real_baselines.single_shot",                   "gemini-2.5-flash", "single_shot"),
    ("zero_shot", "workflow_prismadv.eid_bench_real_baselines.single_shot",                   "gpt-4.1",          "single_shot"),
    ("zero_shot", "workflow_prismadv.eid_bench_real_baselines.single_shot",                   "gpt-5-mini",       "single_shot"),
    ("zero_shot", "workflow_prismadv.eid_bench_real_baselines.single_shot",                   "gemini-2.5-pro",   "single_shot"),
    ("zero_shot", "workflow_prismadv.eid_bench_real_baselines.single_shot",                   "gpt-5",            "single_shot"),
    ("few_shot",  "workflow_prismadv.eid_bench_real_baselines.few_shot",                      "gemini-2.5-flash", "few_shot"),
    ("few_shot",  "workflow_prismadv.eid_bench_real_baselines.few_shot",                      "gpt-4.1",          "few_shot"),
    ("few_shot",  "workflow_prismadv.eid_bench_real_baselines.few_shot",                      "gpt-5-mini",       "few_shot"),
    ("few_shot",  "workflow_prismadv.eid_bench_real_baselines.few_shot",                      "gemini-2.5-pro",   "few_shot"),
    ("few_shot",  "workflow_prismadv.eid_bench_real_baselines.few_shot",                      "gpt-5",            "few_shot"),
    ("swe_agent", "workflow_prismadv.eid_bench_real_baselines.swe_agent",                     "gemini-2.5-flash", "swe_agent"),
    ("swe_agent", "workflow_prismadv.eid_bench_real_baselines.swe_agent",                     "gpt-5",            "swe_agent"),
    ("deequ",     "workflow_prismadv.eid_bench_real_baselines.deequ",                         "deequ",            "deequ"),
    ("pocketflow", "workflow_prismadv.eid_bench_real_baselines.pocketflow_agent",            "gpt-5",            "pocketflow"),
    ("codex",     "workflow_prismadv.eid_bench_real_baselines.codex_agent",                  "gpt-5",            "codex"),
    ("codex",     "workflow_prismadv.eid_bench_real_baselines.codex_agent",                  "gpt-6-sol",        "codex"),
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
    present = {parse_replicate(path.stem) for path in have}
    print(f"\n[{dataset} / {prefix} / {model_name}] {len(have)} artifact(s) already present, "
          f"replicates {sorted(present) or '-'}.")
    for path in have:
        print(f"    - {path.name}")
    missing = [k for k in range(1, target_runs + 1) if k not in present]
    if not missing:
        print(f"  -> target_runs={target_runs} already met; skipping generation.")
        return 0
    final_rc = 0
    for i, replicate in enumerate(missing):
        print(f"\n  ==> launching replicate {replicate} ({i + 1}/{len(missing)}) "
              f"for {prefix} / {model_name} on {dataset}")
        command = [
            "uv", "run", "python", "-m", module,
            "--example-id", dataset,
            "--model-name", model_name,
        ]
        if replicate > 1:
            command += ["--replicate", str(replicate)]
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
            "uv", "run", "python", "-m", VALIDATION_MODULE,
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
