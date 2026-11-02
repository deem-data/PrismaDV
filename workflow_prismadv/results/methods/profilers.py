"""Profilers that infer checks from the observed data: Deequ constraint suggestion and TensorFlow Data Validation
schema inference (tfdv runs in its own env, ~/tfdv, with the repo on PYTHONPATH).

  synth  one artifact per dataset in general/constraints/, one result per label
         (eid_bench_baselines/constraint_{inference,validation}_{deequ,tfdv}.py)
  real   eid_bench_real_baselines/deequ.py + _4_constraints_validation.py; eid_bench_real_baselines/tfdv.py
  icd    icd_bench_experiments/evaluation_{deequ,tfdv}.py --out

  uv run python -m workflow_prismadv.results.methods.profilers
"""
from workflow_prismadv.results.methods._common import Row, icd_arms, real_arms, synth_per_label

FAMILY = "profilers"
ROWS = [Row("deequ", "deequ", False, True),
        Row("tfdv", "tensorflow-dv", False, True)]


def synth():
    return synth_per_label({"deequ": "deequ_constraints", "tfdv": "schema"})


def real():
    return real_arms({"deequ": ("deequ", "deequ"), "tfdv": ("tfdv", "tfdv")}, reps=(1,))


def icd():
    return icd_arms({"deequ": "deequ--deequ", "tfdv": "tfdv--tfdv"}, reps=(1,))


if __name__ == "__main__":
    from workflow_prismadv.results.combined_table import main
    raise SystemExit(main(["--family", FAMILY]))
