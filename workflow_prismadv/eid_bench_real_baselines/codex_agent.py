#!/usr/bin/env python3
"""CLI driver for the Codex CLI EIDBench-real baseline.

The prompt is a document, `prompts/codex.md`, read fresh on every run; changing
the wording does not mean changing any code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import oyaml as yaml

from workflow_prismadv.eid_bench_real_baselines.pipelines.run_codex import run_codex
from workflow_prismadv.eid_bench_real_experiments import utils
from workflow_prismadv.utils.codex_runner import DEFAULT_CODEX_HOME, DEFAULT_MODEL
from workflow_prismadv.utils.replicates import add_replicate_argument

DEFAULT_EXAMPLE_ID = "omop_cdm_synthea"
DEFAULT_SAMPLE_ROWS = 3


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
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="Sample rows per table to include in the prompt.",
    )
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=DEFAULT_CODEX_HOME,
        help="Isolated CODEX_HOME holding the API-key credential.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=(
            "Prompt document to run instead of the checked-in prompts/codex.md. "
            "Use it to try a wording without editing the version other runs share."
        ),
    )
    parser.add_argument(
        "--dump-prompt",
        type=Path,
        default=None,
        help=(
            "Render the prompt for the selected example, write it there, and exit "
            "without calling codex. This is the exact text that would be staged as "
            "task.txt. Writes no artifact, so an existing one is no reason to skip."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Codex model_reasoning_effort override, e.g. low | medium | high. Omitted by "
            "default, which leaves the CLI default in place."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    add_replicate_argument(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    output_path = run_codex(
        project_root=project_root,
        example_id=args.example_id,
        input_path=args.input_path,
        model_name=args.model_name,
        sample_rows=args.sample_rows,
        codex_home=args.codex_home,
        prompt_path=args.prompt_file,
        dump_prompt=args.dump_prompt,
        reasoning_effort=args.reasoning_effort,
        overwrite=args.overwrite,
        replicate=args.replicate,
    )
    if args.dump_prompt is not None:
        # Not an artifact: the sweep summary names `output_path`, and callers
        # parse it as the constraints file that was written.
        print(yaml.safe_dump({"prompt_path": str(output_path)}, sort_keys=False))
        return 0
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
