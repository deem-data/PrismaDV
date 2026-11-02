"""Stage 3 of the SIFTA pipeline: validates programs' constraint artifacts on the corrupted batch of every
label of a dataset."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from prismadv.utils import get_project_root
from workflow_prismadv.results.benchmarks import synth
from workflow_sifta._2_generation import ABLATION_PROGRAMS, PROGRAMS

VALIDATOR = "workflow_prismadv/eid_bench_experiments/_5_constraints_validation.py"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=synth.DATASETS)
    ap.add_argument("--program", action="append", required=True, choices=PROGRAMS + ABLATION_PROGRAMS,
                    help="repeatable")
    ap.add_argument("--llm", default="gemini-3.8-flash")
    ap.add_argument("--label", action="append", help="only these labels (repeatable); default: every label")
    ap.add_argument("--task", action="append", help="only these tasks (repeatable); default: every task")
    a = ap.parse_args(argv)
    cmd = [sys.executable, "-u", VALIDATOR, "--dataset_name", a.dataset]
    for p in a.program:
        cmd += ["--constraint_glob", f"{p}--{a.llm}--*.yaml"]
    for n in a.label or []:
        cmd += ["--label", n]
    for t in a.task or []:
        cmd += ["--script_name", t]
    env = {**os.environ, "SPARK_VERSION": os.environ.get("SPARK_VERSION", "3.5")}
    return subprocess.run(cmd, cwd=get_project_root(), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
