#!/usr/bin/env python3
"""CLI driver for the mini-swe-agent EIDBench-real baseline.

Mirrors `single_shot.py` / `few_shot.py` CLI shape so the orchestration sweeps
can re-use the same conventions (--example-id, --model-name, --overwrite).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import oyaml as yaml

from workflow_prismadv.eid_bench_real_baselines.pipelines.run_mini_swe_agent import (
    DEFAULT_COST_LIMIT,
    DEFAULT_SAMPLE_ROWS,
    run_mini_swe_agent_real_etl,
)
from workflow_prismadv.eid_bench_real_experiments import utils


DEFAULT_EXAMPLE_ID = "omop_cdm_synthea"
DEFAULT_MODEL_NAME = "gemini-2.5-flash"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", default=DEFAULT_EXAMPLE_ID)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Prepared EIDBench-real input artifact. Defaults to constraints/project/prismadv_inputs.yaml.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=DEFAULT_COST_LIMIT,
        help="USD budget cap for the agent run (passed to `mini --cost-limit`).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="Sample rows per table to include in the prompt.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    output_path = run_mini_swe_agent_real_etl(
        project_root=project_root,
        example_id=args.example_id,
        input_path=args.input_path,
        model_name=args.model_name,
        cost_limit=args.cost_limit,
        sample_rows=args.sample_rows,
        overwrite=args.overwrite,
    )
    print(
        yaml.safe_dump(
            {
                "example_id": args.example_id,
                "script_id": "project",
                "output_path": str(output_path),
            },
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
