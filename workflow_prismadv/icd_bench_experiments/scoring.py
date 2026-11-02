"""Scoring shared by the ICDBench evaluators.

Per case, the constraints must all pass on `data_to_pass` and at least one must
fail on `data_to_reject`. A passed pass batch is a true positive (safe is the
positive class); a missed reject batch is a false positive.
"""
import argparse
import csv
import os
import time

import pandas as pd

from prismadv.utils import get_project_root

# Per-case outcome tables that workflow_prismadv/results reads: <method>--<model>--r<N>.tsv. Not inside
# data_processed/icd_bench, whose sub-directories the evaluators read as cases.
OUTCOMES_DIR = get_project_root() / "data_processed" / "icd_bench_outcomes"
OUTCOME_FIELDS = ["case", "passes_pass_batch", "rejects_reject_batch", "wall_clock_seconds"]


def f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def spark_df(dq_manager, evaluation_case, data):
    frame = pd.DataFrame(data)
    # Spark cannot infer a type for this case's all-null column.
    if evaluation_case.__class__.__name__ == 'GithubFashionTrendsCategory':
        from pyspark.sql.types import StructType, StructField, StringType, FloatType
        schema = StructType([
            StructField("comments", StringType(), True),
            StructField("category", FloatType(), True),
        ])
        return dq_manager.spark_df_from_pandas_df(frame, schema=schema)
    return dq_manager.spark_df_from_pandas_df(frame)


def score_case(dq_manager, evaluation_case, constraints):
    """Whether the constraints reject the reject batch and pass the pass batch; returns the Spark session too."""
    reject_df, spark = spark_df(dq_manager, evaluation_case, evaluation_case.data_to_reject())
    reject_results = dq_manager.validate_on_spark_df(spark, reject_df, constraints) if constraints else []
    pass_df, spark = spark_df(dq_manager, evaluation_case, evaluation_case.data_to_pass())
    pass_results = dq_manager.validate_on_spark_df(spark, pass_df, constraints) if constraints else []
    return {
        "num_constraints": len(constraints),
        "exact_match": evaluation_case.ground_truth_constraint() in constraints,
        "rejects_reject_batch": 'Failure' in reject_results,
        "passes_pass_batch": 'Failure' not in pass_results,
        "failing_on_pass_batch": [c for c, r in zip(constraints, pass_results) if r == 'Failure'],
    }, spark


def case_dirs():
    return sorted(os.listdir(get_project_root() / "data_processed" / "icd_bench"))


def out_argument(description, default_name):
    """--out for a non-LLM evaluator; defaults to OUTCOMES_DIR/<default_name> (one replicate: deterministic)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", default=str(OUTCOMES_DIR / default_name),
                        help="Per-case outcome TSV (default: %(default)s).")
    return parser.parse_args()


def write_outcomes(rows, out):
    """Write the per-case outcomes and print the counts (safe = pass batch is the positive class)."""
    n = len(rows)
    tp = sum(r["passes_pass_batch"] for r in rows)
    tn = sum(r["rejects_reject_batch"] for r in rows)
    print(f"\n{n} cases: {tp} pass batches passed, {n - tp} false alarms, {tn} reject batches rejected, "
          f"{n - tn} missed; F1 (safe positive) {f1(tp, n - tn, n - tp):.4f}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTCOME_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


def evaluate_task_agnostic(fit, out):
    """Score a detector that is fitted on each case's sample: fit(sample_df) -> rejects(batch_df) -> bool."""
    from workflow_prismadv.icd_bench_experiments import instantiate_evaluation_case
    rows = []
    for case_dir in case_dirs():
        case = instantiate_evaluation_case(case_dir)
        started = time.perf_counter()
        rejects = fit(pd.DataFrame(case.sample_data()))
        rejected = bool(rejects(pd.DataFrame(case.data_to_reject())))
        passed = not rejects(pd.DataFrame(case.data_to_pass()))
        rows.append({"case": case_dir, "passes_pass_batch": passed, "rejects_reject_batch": rejected,
                     "wall_clock_seconds": round(time.perf_counter() - started, 3)})
        print(f"{case_dir}: rejects reject batch {rejected}, passes pass batch {passed}")
    write_outcomes(rows, out)


def report(title, rows, missing, out=None):
    """Print the scores, cost, LLM calls and wall-clock time; optionally write the per-case TSV.

    Each row carries `usd`, `usd_without_cache`, `attempts`, `llm_calls`,
    `llm_calls_exact` and `wall_clock_seconds` besides the fields of `score_case`.
    """
    n = len(rows)
    tp = sum(r["passes_pass_batch"] for r in rows)
    fn = n - tp
    tn = sum(r["rejects_reject_batch"] for r in rows)
    fp = n - tn
    f1_safe = f1(tp, fp, fn)
    f1_unsafe = f1(tn, fn, fp)

    print(f"\n{title}: {n} cases scored, {len(missing)} missing")
    if missing:
        print(f"missing: {missing}")
    print(f"{sum(r['num_constraints'] for r in rows)} constraints generated.")
    print(f"{sum(r['exact_match'] for r in rows)}/{n} exact matches of the ground truth constraint.")
    print(f"{tp}/{n} correct data batches passed.")
    print(f"{fn}/{n} false alarms for correct data batches.")
    print(f"{tn}/{n} problematic data batches identified.")
    print(f"{fp}/{n} problematic data batches missed.")
    print(f"F1 (safe positive):   {f1_safe:.4f}")
    print(f"F1 (unsafe positive): {f1_unsafe:.4f}")
    print(f"Macro F1:             {(f1_safe + f1_unsafe) / 2:.4f}")
    print(f"Cost: ${sum(r['usd'] or 0 for r in rows):.2f} "
          f"(without cache ${sum(r['usd_without_cache'] or 0 for r in rows):.2f})")
    if rows:
        calls = [r["llm_calls"] for r in rows]
        seconds = [r["wall_clock_seconds"] for r in rows]
        estimated = sum(not r["llm_calls_exact"] for r in rows)
        print(f"LLM calls: {sum(calls)} in total, {sum(calls) / n:.1f} per case (min {min(calls)}, max {max(calls)})"
              + (f"; {estimated} case(s) estimated" if estimated else ""))
        print(f"Wall clock: {sum(seconds):.0f} s in total, {sum(seconds) / n:.1f} s per case "
              f"(min {min(seconds):.0f}, max {max(seconds):.0f})")
        print(f"Attempts: {sum(r['attempts'] for r in rows)} for {n} cases")

    if out:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["case"], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out}")
