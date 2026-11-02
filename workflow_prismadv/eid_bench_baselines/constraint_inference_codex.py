#!/usr/bin/env python3
"""CLI driver for the Codex EIDBench-synth baseline."""
import argparse
import sys
from pathlib import Path

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_baselines.pipelines.run_codex import run_codex
from workflow_prismadv.eid_bench_baselines.prompts import PROMPT_PATH
from workflow_prismadv.utils.codex_runner import (
    DEFAULT_CODEX_HOME,
    DEFAULT_ISOLATION,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    ISOLATION_AUDIT,
    ISOLATION_BWRAP,
)

DATASET_NAME_OPTIONS = ["imdb", "hr_analytics", "sleep_health", "students", "IPL_win_prediction"]
SUBTASK_OPTIONS = ["general_task"]
PROCESSED_DATA_LABEL = "0"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", default=None,
                        choices=DATASET_NAME_OPTIONS,
                        help="Restrict to one or more datasets (default: all).")
    parser.add_argument("--script", action="append", default=None,
                        help="Restrict to one or more script stems, e.g. general_task_11.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help=f"Prompt document to run instead of the checked-in one "
                             f"(default: {PROMPT_PATH}). Use it to try a wording "
                             f"without editing the version other runs share.")
    parser.add_argument("--dump-prompt", type=Path, default=None,
                        help="Render the prompt for each selected script, write it there, "
                             "and exit without calling codex. A directory (or more than one "
                             "script) gets one file per script. This is the exact text that "
                             "would be staged as task.txt.")
    parser.add_argument("--codex-home", type=Path, default=DEFAULT_CODEX_HOME,
                        help="CODEX_HOME holding the credential to run with, which keeps the "
                             "API key out of the agent's own shell.")
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        help=f"Codex model_reasoning_effort, e.g. low | medium | high "
                             f"(default: {DEFAULT_REASONING_EFFORT}). It is part of the artifact "
                             f"filename, so runs at different efforts sit side by side.")
    parser.add_argument("--isolation", choices=[ISOLATION_BWRAP, ISOLATION_AUDIT],
                        default=DEFAULT_ISOLATION,
                        help="audit (default) gives the agent a folder, instructs it to stay "
                             "in there, and reads back off the trajectory whether it did -- "
                             "detection, not containment. bwrap instead runs it in a mount "
                             "namespace where the benchmark tree does not exist.")
    parser.add_argument("--replicate", type=int, default=None,
                        help="Replicate index for a variance study. Part of the artifact name, "
                             "so repeats of one configuration coexist instead of skipping over "
                             "each other; replicate 1 keeps the bare name.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_names = args.dataset or DATASET_NAME_OPTIONS
    for dataset_name in dataset_names:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        for subtask_name in SUBTASK_OPTIONS:
            try:
                project_manager.get_subtask_description(subtask_name=subtask_name)
            except ValueError as e:
                print(f"Skipping invalid subtask '{subtask_name}' for dataset '{dataset_name}': {e}")
                continue
            run_codex(
                dataset_name=dataset_name,
                subtask_name=subtask_name,
                processed_data_label=PROCESSED_DATA_LABEL,
                model_name=args.model_name,
                codex_home=args.codex_home,
                prompt_path=args.prompt_file,
                dump_prompt=args.dump_prompt,
                script_stems=args.script,
                reasoning_effort=args.reasoning_effort,
                replicate=args.replicate,
                isolation=args.isolation,
                overwrite=args.overwrite,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
