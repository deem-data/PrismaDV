from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from prismadv.data_models import ValidationResults
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.results.benchmarks import synth
from workflow_prismadv.utils.replicates import replicate_part
from workflow_sifta._0_splits import DATASETS, SPLIT_SEEDS, SUBTASK, load_split
from workflow_sifta.baselines.redyuk.data_profiler import KNNNoveltyDetector, compute_profile, infer_column_kinds

PROGRAM = "stats_novelty"
RESULT_STEM = "sifta_stats_novelty"
CLEAN_LABEL = "0"
CONTAMINATION = 0.01
N_NEIGHBORS = 5
NOVELTY_COLUMN = "__novelty__"
SUGGESTION = "stats_novelty.batch_profile_not_novel"


def result_stem(split: int) -> str:
    """Stem of the verdict files of split N (= replicate N, no suffix for 1); stage 4 reads them by this stem."""
    return "--".join(filter(None, [RESULT_STEM, replicate_part(split)]))


def _validation_result(rejected: bool) -> ValidationResults:
    return ValidationResults.from_dict({
        NOVELTY_COLUMN: {
            "code": [{
                "suggestion": SUGGESTION,
                "status": not rejected,
                "reason_if_failed": "batch profile is novel w.r.t. the observed batches (one-class kNN)" if rejected else "",
                "level": "error",
            }]
        }
    })


def run_split(ds: str, split: int, overwrite: bool = False) -> dict | None:
    pm = ProjectManager(project_root=get_project_root(), dataset_name=ds, downstream_task_type="general")
    sp = load_split(ds, split)
    stem = result_stem(split)
    existing = [n for n in sp["test_labels"] if synth.label_result(ds, n, stem).exists()]
    if existing and not overwrite:
        print(f"{ds} split{split}: {len(existing)}/{len(sp['test_labels'])} verdicts exist, skipping "
              f"(--overwrite replaces them)")
        return None

    def batch(label: str, clean: bool) -> pd.DataFrame:
        return pd.read_csv(pm.get_new_test_data_path(SUBTASK, label, clean=clean))

    started = time.perf_counter()
    kinds = infer_column_kinds(batch(CLEAN_LABEL, clean=True))
    fit_labels = sp["train_eval_labels"]
    profiles = [compute_profile(batch(n, clean=False), kinds) for n in fit_labels]
    k = max(1, min(N_NEIGHBORS, len(profiles) - 1))
    model = KNNNoveltyDetector(contamination=CONTAMINATION, n_neighbors=k).fit(profiles)
    fit_s = time.perf_counter() - started

    verdicts = {}
    for n in sp["test_labels"]:
        t0 = time.perf_counter()
        rejected = bool(model.predict([compute_profile(batch(n, clean=False), kinds)])[0])
        result = _validation_result(rejected)
        result.validation_summary = {"elapsed_s": round(time.perf_counter() - t0, 3), "fit_s": round(fit_s, 3)}
        path = synth.label_result(ds, n, stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.save_to_yaml(path)
        verdicts[n] = rejected

    summary = {"dataset": ds, "split": split, "program": PROGRAM, "result_stem": stem,
               "detector": "one-class kNN (Redyuk et al., EDBT 2021)", "contamination": CONTAMINATION,
               "n_neighbors": N_NEIGHBORS, "n_neighbors_used": k, "fit_labels": fit_labels,
               "fit_includes_clean_batch": False, "fit_s": round(fit_s, 3),
               "rejected": sum(verdicts.values()), "test_labels": len(verdicts), "verdicts": verdicts}
    out = get_project_root() / "optimization_runs" / "stats_novelty" / f"{ds}_split{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"{ds} split{split}: fit {len(profiles)} batches, k={k}, rejected {summary['rejected']}/{len(verdicts)} "
          f"test labels -> {out}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--splits", nargs="+", type=int, default=list(SPLIT_SEEDS))
    ap.add_argument("--overwrite", action="store_true", help="replace the verdicts of a split that has some already")
    a = ap.parse_args(argv)
    for ds in a.datasets:
        for s in a.splits:
            run_split(ds, s, a.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
