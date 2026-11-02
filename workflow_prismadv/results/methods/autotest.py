"""AutoTest (Chen et al., SIGMOD 2025): semantic-domain constraints mined offline from web tables, applied to every
column; a batch is rejected when any value is flagged. Runs in its own conda env on a GPU
(eid_bench_baselines/pipelines/run_autotest.py).

  synth  {pass, anomalies} per label (eid_bench_baselines/constraint_validation_autotest.py)
  real   summary.failed_error per label (eid_bench_real_baselines/autotest.py; 1000-row sample per table)
  icd    icd_bench_experiments/evaluation_autotest.py --out

  uv run python -m workflow_prismadv.results.methods.autotest
"""
from workflow_prismadv.results.methods._common import Row, icd_arms, real_arms, synth_per_label

FAMILY = "autotest"
ROWS = [Row("autotest", "auto-test", False, False)]


def synth():
    return synth_per_label({"autotest": "autotest_constraints"})


def real():
    return real_arms({"autotest": ("autotest", "autotest")}, reps=(1,))


def icd():
    return icd_arms({"autotest": "autotest--autotest"}, reps=(1,))


if __name__ == "__main__":
    from workflow_prismadv.results.combined_table import main
    raise SystemExit(main(["--family", FAMILY]))
