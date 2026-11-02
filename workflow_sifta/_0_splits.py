"""Stage 0 of the SIFTA pipeline: train/eval/test splits of EIDBench-synth for SIFTA and its GEPA baselines.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import oyaml as yaml

from prismadv.project_manager import is_clean_label
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

SUBTASK = "general_task"
DATASETS = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
SPLIT_SEEDS = (1, 2, 3)
LABEL_SPLIT_RATIO = 0.5
SCRIPT_SPLIT_RATIO = 0.6
TRAIN_EVAL_SCRIPT_SPLIT_RATIO = 0.5
TEST_SCRIPT_FRACTION = 0.5


def split_path(dataset: str, seed: int, root: Path | None = None) -> Path:
    return (root or get_project_root()) / "benchmarks/EIDBench-synth" / dataset / "splits" / f"split{seed}.yaml"


def make_split(dataset: str, seed: int) -> dict:
    from workflow_sifta.optimization_fns import OptimizationWorkflow

    labels = [label for label in ProjectManager(project_root=get_project_root(), dataset_name=dataset)
              .get_available_processed_data_labels_for_subtask(SUBTASK) if not is_clean_label(label)]
    random.seed(seed)
    random.shuffle(labels)
    n_labels = int(len(labels) * LABEL_SPLIT_RATIO)
    train_eval_labels, test_labels = labels[:n_labels], labels[n_labels:]

    workflow = OptimizationWorkflow(train_dataset_subtasks={dataset: [SUBTASK]},
                                    train_processed_data_label_list=train_eval_labels,
                                    project_root=get_project_root())
    scripts = list(workflow.get_available_script_names_for_subtask(dataset, SUBTASK))
    random.shuffle(scripts)
    n_train_eval = int(len(scripts) * SCRIPT_SPLIT_RATIO)
    train_eval_scripts, test_scripts = scripts[:n_train_eval], scripts[n_train_eval:]
    n_train = int(len(train_eval_scripts) * TRAIN_EVAL_SCRIPT_SPLIT_RATIO)
    train_scripts, eval_scripts = train_eval_scripts[:n_train], train_eval_scripts[n_train:]

    return {
        "dataset": dataset,
        "subtask": SUBTASK,
        "split_seed": seed,
        "ratios": {"labels_train_eval": LABEL_SPLIT_RATIO, "scripts_train_eval": SCRIPT_SPLIT_RATIO,
                   "train_of_train_eval_scripts": TRAIN_EVAL_SCRIPT_SPLIT_RATIO},
        "train_scripts": train_scripts,
        "eval_scripts": eval_scripts,
        "test_scripts": test_scripts,
        "train_eval_labels": train_eval_labels,
        "test_labels": test_labels,
    }


def load_split(dataset: str, seed: int, root: Path | None = None) -> dict:
    path = split_path(dataset, seed, root)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; write it with python -m workflow_sifta._0_splits")
    return yaml.safe_load(path.read_text())


def scored_half(scripts: list[str], seed: int) -> list[str]:
    """The fixed TEST_SCRIPT_FRACTION of the scripts that a test type scores."""
    return sorted(random.Random(seed).sample(sorted(scripts), math.ceil(len(scripts) * TEST_SCRIPT_FRACTION)))


def test_units(split: dict) -> dict[str, tuple[list[str], list[str]]]:
    """(scripts, labels) of each test type."""
    seed = split["split_seed"]
    seen_scripts = scored_half(split["train_scripts"] + split["eval_scripts"], seed)
    new_scripts = scored_half(split["test_scripts"], seed)
    return {
        "test_1_cross_new_data": (seen_scripts, split["test_labels"]),
        "test_2_new_script_train_data": (new_scripts, split["train_eval_labels"]),
        "test_3_new_script_new_data": (new_scripts, split["test_labels"]),
    }


def scored_scripts(split: dict) -> list[str]:
    """Every script that a test type of the split scores: the ones generation must cover."""
    return sorted({t for scripts, _ in test_units(split).values() for t in scripts})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SPLIT_SEEDS))
    a = ap.parse_args(argv)
    for dataset in a.datasets:
        for seed in a.seeds:
            path = split_path(dataset, seed)
            if path.exists():
                print(f"{path} exists; left unchanged")
                continue
            split = make_split(dataset, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(split, sort_keys=False))
            print(f"{path}: {len(split['train_scripts'])}/{len(split['eval_scripts'])}/{len(split['test_scripts'])} "
                  f"scripts, {len(split['train_eval_labels'])}/{len(split['test_labels'])} labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
