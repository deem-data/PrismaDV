"""Evaluate AutoTest (Chen et al., SIGMOD 2025) on ICDBench.

AutoTest rejects a batch when it flags an outlier in any column. Every pass and reject batch of the 63 cases goes
to one AutoTest call (an external conda env on a GPU, see eid_bench_baselines/pipelines/run_autotest.py; staging
under AUTOTEST_STAGING/<--run-name>, a resubmit resumes). A case whose batch AutoTest failed on gets no row rather
than a silent pass. Wall clock per case = detection time of its two batches (the first table also loads models).
"""
import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

from workflow_prismadv.icd_bench_experiments import instantiate_evaluation_case
from workflow_prismadv.icd_bench_experiments.scoring import OUTCOMES_DIR, case_dirs, write_outcomes
from workflow_prismadv.eid_bench_baselines.pipelines.run_autotest import autotest_detect_tables, autotest_table_timings

SDC_NAME = "rt_train"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default=str(OUTCOMES_DIR / "autotest--autotest--r1.tsv"))
parser.add_argument("--run-name", default="icd_eval")
args = parser.parse_args()

id_to_df = {}
for case_dir in case_dirs():
    case = instantiate_evaluation_case(case_dir)
    id_to_df[f"{case_dir}__reject"] = pd.DataFrame(case.data_to_reject())
    id_to_df[f"{case_dir}__pass"] = pd.DataFrame(case.data_to_pass())

print(f"Running AutoTest ({SDC_NAME} SDCs) on {len(id_to_df)} batches of {len(id_to_df) // 2} ICDBench cases...")
detections = autotest_detect_tables(id_to_df, sdc_name=SDC_NAME, run_name=args.run_name, skip_missing=True)
timings = autotest_table_timings(id_to_df, args.run_name)

rows = []
for case_dir in case_dirs():
    ids = (f"{case_dir}__reject", f"{case_dir}__pass")
    if not all(i in detections for i in ids):
        print(f"{case_dir}: AutoTest failed on a batch; no row written")
        continue
    seconds = [timings.get(i, {}).get("elapsed_s") for i in ids]
    rows.append({"case": case_dir, "passes_pass_batch": len(detections[ids[1]]) == 0,
                 "rejects_reject_batch": len(detections[ids[0]]) > 0,
                 "wall_clock_seconds": sum(seconds) if None not in seconds else None})
write_outcomes(rows, args.out)
