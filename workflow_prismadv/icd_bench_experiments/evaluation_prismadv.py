"""Evaluate PrismaDV on ICDBench.
"""
import argparse
import json
import os
import signal
import sys
import warnings

import oyaml as yaml
import pandas as pd

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.post_processing.individual_constraint import determine_validity
from workflow_prismadv.icd_bench_experiments import ALL_EVALUATION_CASES, find_constraints_output_path
from workflow_prismadv.icd_bench_experiments.codex import case_id
from workflow_prismadv.icd_bench_experiments.scoring import report, score_case

warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_name", required=True)
parser.add_argument("--replicate", type=int, default=1)
parser.add_argument("--out", default=None, help="Write the per-case results to this TSV.")
parser.add_argument("--error-report", default=None,
                    help="Write the missed and false-alarm cases, with their constraints, to this JSON file.")
args = parser.parse_args()

dq_manager = DeequDataQualityManager()
rows, missing, error_report = [], [], []
spark = None
for evaluation_case in ALL_EVALUATION_CASES:
    constraints_output_path = find_constraints_output_path(evaluation_case, args.model_name, args.replicate)
    if not constraints_output_path.exists():
        missing.append(case_id(evaluation_case))
        continue
    with open(constraints_output_path, "r") as f:
        res_dict = yaml.safe_load(f)

    generated_constraints = []
    target_column = evaluation_case.target_column()
    if target_column in res_dict['constraints']:
        for code in res_dict['constraints'][target_column]['code']:
            generated_constraints.append(code['suggestion'])

    data_sample_df, spark = dq_manager.spark_df_from_pandas_df(pd.DataFrame(evaluation_case.sample_data()))
    generated_constraints = [constraint for constraint in generated_constraints
                             if determine_validity(constraint, spark, data_sample_df)[0]]

    scores, spark = score_case(dq_manager, evaluation_case, generated_constraints)
    # Artifacts written before cost tracking carry no cost summary.
    stages = (res_dict.get("cost_summary") or {}).values()
    usd = sum(stage.get("cost") or 0 for stage in stages)
    row = {
        "case": case_id(evaluation_case),
        **scores,
        "usd": usd,
        "usd_without_cache": usd,
        "attempts": 1,
        "llm_calls": sum(stage.get("requests") or 0 for stage in stages),
        "llm_calls_exact": bool(stages),
        "wall_clock_seconds": res_dict.get("wall_clock_seconds") or 0,
    }
    rows.append(row)

    print(f"{row['case']}: {row['num_constraints']} constraints, exact={row['exact_match']}, "
          f"rejects={row['rejects_reject_batch']}, passes={row['passes_pass_batch']}")
    print(f"\tground truth: {evaluation_case.ground_truth_constraint()}")
    for constraint in generated_constraints:
        print(f"\t\t{constraint}")

    case_name = f"{evaluation_case.__class__.__module__}.{evaluation_case.__class__.__name__}"
    if not row["rejects_reject_batch"]:
        error_report.append({'case': case_name, 'error_type': 'missed',
                             'assumption': evaluation_case.assumption_in_natural_language(),
                             'generated_constraints': generated_constraints,
                             'passing_constraints': generated_constraints})
    if not row["passes_pass_batch"]:
        error_report.append({'case': case_name, 'error_type': 'false_alarm',
                             'assumption': evaluation_case.assumption_in_natural_language(),
                             'generated_constraints': generated_constraints,
                             'failing_constraints': row["failing_on_pass_batch"]})

report(f"ICDBench PrismaDV, model {args.model_name}, replicate {args.replicate}", rows, missing, args.out)

if args.error_report:
    with open(args.error_report, "w") as f:
        json.dump(error_report, f, indent=2, sort_keys=True)
    print(f"Wrote {args.error_report}")

if spark is not None:
    spark.stop()
sys.stdout.flush()
os.kill(os.getpid(), signal.SIGTERM)
