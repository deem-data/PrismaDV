"""Stage 4 of the SIFTA pipeline: macro F1 of each program on the three test types of EIDBench-synth.
"""
from __future__ import annotations

import argparse

import oyaml as yaml
import pandas as pd

from workflow_prismadv.results.alarm import alarm, load
from workflow_prismadv.results.benchmarks import COLUMNS, synth
from workflow_prismadv.results.metrics import dataset_summary, summary
from workflow_prismadv.utils.replicates import parse_replicate
from workflow_sifta._0_splits import DATASETS, SPLIT_SEEDS, load_split, test_units
from workflow_sifta._2_generation import ABLATION_PROGRAMS, BASE_PROGRAMS, FAIL_PRECISION_PROGRAMS, PROGRAMS
from workflow_sifta.baselines.redyuk.run import result_stem as stats_novelty_stem
from workflow_sifta.metrics.optimization_metrics import METRIC_REV

LLM = "gemini-3.8-flash"
BASE_REPLICATE = 1
LABEL_PROGRAMS = {"stats_novelty": stats_novelty_stem}            # program -> stem of its verdicts per replicate
NA_TESTS = {"stats_novelty": {"test_2_new_script_train_data"}}     # its fit batches are the test 2 labels
TABLE_PROGRAMS = PROGRAMS + list(LABEL_PROGRAMS)
REFERENCE_PROGRAMS = ["prismadv"]                                   # Manual: PrismaDV's hand-tuned prompts


def newest_artifact(ds: str, task: str, program: str, model: str, replicate: int):
    found = [p for p in sorted((synth.general(ds) / "constraints" / task).glob(f"{program}--{model}--*.yaml"))
             if parse_replicate(p) == replicate]
    return found[-1] if found else None


def metric_rev(artifact) -> str | None:
    """The fail precision version of the run that made the artifact (stage 2 provenance)."""
    return (yaml.safe_load(artifact.read_text()).get("sifta_generation") or {}).get("metric_rev")


def outcomes(datasets, splits, programs=TABLE_PROGRAMS, model=LLM, models=None) -> pd.DataFrame:
    """Outcome rows of the programs; `models` maps a program to the model of its artifacts (default `model`)."""
    rows, missing = [], []
    models = models or {}
    for ds in datasets:
        truth = {(n, t): safe for n, t, safe in synth.units(ds)}
        for s in splits:
            units = test_units(load_split(ds, s))
            for program in programs:
                replicate = BASE_REPLICATE if program in BASE_PROGRAMS else s
                artifacts = {}
                for test_type, (scripts, labels) in units.items():
                    if test_type in NA_TESTS.get(program, ()):
                        continue
                    if program in LABEL_PROGRAMS:
                        for n in labels:
                            f = synth.label_result(ds, n, LABEL_PROGRAMS[program](replicate))
                            if not f.exists():
                                missing.append(f"{ds} split{s} {program} {n}: no verdict")
                                continue
                            verdict = alarm(load(f))
                            rows += [(test_type, ds, f"{n}/{t}", program, s, truth[(n, t)], verdict)
                                     for t in scripts if (n, t) in truth]
                        continue
                    for t in scripts:
                        if t not in artifacts:
                            artifacts[t] = newest_artifact(ds, t, program, models.get(program, model), replicate)
                            if artifacts[t] is None:
                                missing.append(f"{ds} split{s} {program} {t}: no artifact")
                            elif program in FAIL_PRECISION_PROGRAMS and metric_rev(artifacts[t]) != METRIC_REV:
                                missing.append(f"{ds} split{s} {program} {t}: {artifacts[t].name} is from a run "
                                               f"before fail precision {METRIC_REV!r}")
                                artifacts[t] = None
                        if artifacts[t] is None:
                            continue
                        for n in labels:
                            if (n, t) not in truth:
                                continue
                            f = synth.task_result(ds, n, t, artifacts[t].stem)
                            if not f.exists():
                                missing.append(f"{ds} split{s} {program} {n}/{t}: no validation result")
                                continue
                            rows.append((test_type, ds, f"{n}/{t}", program, s, truth[(n, t)], alarm(load(f))))
    if missing:
        shown = sorted(set(missing))
        raise ValueError(f"incomplete outcomes ({len(shown)}):\n" + "\n".join(shown[:40]))
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--splits", nargs="+", type=int, default=list(SPLIT_SEEDS))
    ap.add_argument("--programs", nargs="+", default=TABLE_PROGRAMS,
                    choices=TABLE_PROGRAMS + ABLATION_PROGRAMS + REFERENCE_PROGRAMS,
                    help="default: the main rows; ablations and the reference rows (prismadv = Manual) on request")
    ap.add_argument("--model", action="append", default=[], metavar="PROGRAM=LLM",
                    help=f"score a program with this model's artifacts (repeatable; default {LLM})")
    a = ap.parse_args(argv)
    models = {}
    for item in a.model:
        program, sep, llm = item.partition("=")
        if not sep or program not in a.programs or not llm:
            ap.error(f"--model {item!r}: give PROGRAM=LLM for one of --programs")
        models[program] = llm
    out = outcomes(a.datasets, a.splits, a.programs, models=models)
    print(f"datasets {a.datasets}, splits {a.splits}, artifacts of {LLM}"
          f"{''.join(f', {p}: {m}' for p, m in models.items())}: {len(out)} outcomes\n")
    s = summary(out)[["f1_mean", "f1_sd", "reps"]].rename(columns={"f1_mean": "macro_f1", "f1_sd": "sd"})
    print(s.unstack("benchmark").reindex(a.programs).round(3).to_string())
    print("\nper dataset, mean over splits:")
    d = dataset_summary(out)["f1_mean"].unstack("benchmark")
    print(d.reindex(a.programs, level="arm").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
