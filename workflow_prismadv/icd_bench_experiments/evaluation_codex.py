"""Evaluate the codex row on ICDBench.

Scored as the other ICDBench rows (see `scoring.py`). Every constraint in the
artifact is used, unfiltered, as for the mini-swe-agent and PocketFlow rows.

Also reports the F1 with the reject batch as the positive class, the macro F1 of
the two, and the cost, LLM calls and wall-clock time of the generation, each
over the accepted attempt plus every failed one.
"""
import argparse
import os
import signal
import sys
import warnings

import oyaml as yaml

from prismadv.dq_manager import DeequDataQualityManager
from workflow_prismadv.icd_bench_experiments import ALL_EVALUATION_CASES
from workflow_prismadv.icd_bench_experiments.codex import case_id, output_path
from workflow_prismadv.icd_bench_experiments.scoring import report, score_case
from workflow_prismadv.utils.codex_runner import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT

warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-name", default=DEFAULT_MODEL)
parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
parser.add_argument("--replicate", type=int, default=1)
parser.add_argument("--out", default=None, help="Write the per-case results to this TSV.")
args = parser.parse_args()

dq_manager = DeequDataQualityManager()
rows, missing = [], []
spark = None
for evaluation_case in ALL_EVALUATION_CASES:
    path = output_path(evaluation_case, args.model_name, args.reasoning_effort, args.replicate)
    if not path.exists():
        missing.append(case_id(evaluation_case))
        continue
    with open(path, "r") as f:
        artifact = yaml.safe_load(f)
    constraints = artifact.get("constraints") or []
    scores, spark = score_case(dq_manager, evaluation_case, constraints)

    # Calls and time of the accepted attempt plus every failed one, the same
    # span as `cost_all_attempts`. The call count is exact when codex left its
    # session rollout, which records one token_count event per model request.
    accounting = artifact.get("token_accounting") or {}
    failed = artifact.get("failed_attempts") or []
    llm_calls = (accounting.get("llm_calls") or accounting.get("llm_calls_estimate") or 0) \
        + sum(f.get("llm_calls") or 0 for f in failed)
    wall_clock = (artifact.get("wall_clock_seconds") or 0) + sum(f.get("wall_clock_seconds") or 0 for f in failed)

    row = {
        "case": case_id(evaluation_case),
        **scores,
        "usd": (artifact.get("cost_all_attempts") or {}).get("usd"),
        "usd_without_cache": (artifact.get("cost_all_attempts") or {}).get("usd_without_cache"),
        "attempts": 1 + len(failed),
        "llm_calls": llm_calls,
        "llm_calls_exact": accounting.get("llm_calls") is not None,
        "wall_clock_seconds": round(wall_clock, 3),
    }
    rows.append(row)
    print(f"{row['case']}: {row['num_constraints']} constraints, exact={row['exact_match']}, "
          f"rejects={row['rejects_reject_batch']}, passes={row['passes_pass_batch']}")
    print(f"\tground truth: {evaluation_case.ground_truth_constraint()}")
    for constraint in constraints:
        print(f"\t\t{constraint}")

report(f"ICDBench codex, model {args.model_name}, effort {args.reasoning_effort}, replicate {args.replicate}",
       rows, missing, args.out)

if spark is not None:
    spark.stop()
# The signal below ends the process without flushing, so output piped to a
# file or a pager would be lost.
sys.stdout.flush()
# As in the other ICDBench evaluators, py4j keeps the process alive otherwise.
os.kill(os.getpid(), signal.SIGTERM)
