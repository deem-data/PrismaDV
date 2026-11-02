"""The label an optimization run sees for a unit (dataset, label, script): "undiscovered silent failures"."""
from __future__ import annotations

import json
import random

from prismadv.project_manager.manager.base import ProjectManager

SILENT_AS_SAFE_SEED = 42
NUM_BATCHES_SEED = 42

Unit = tuple[str, str, str]  # (dataset, label, script)


def outcome_type(pm: ProjectManager, subtask: str, label: str, script: str) -> str:
    """"safe", "silent" or "crash": how the task behaves on the label's corrupted batch."""
    path = pm.get_execution_output_validation_path(subtask, label, script) / "outcome_type.json"
    return json.loads(path.read_text())["outcome"]


def silent_units(pm: ProjectManager, dataset: str, subtask: str, scripts: list[str], labels: list[str]) -> list[Unit]:
    return sorted((dataset, label, script) for script in scripts for label in labels
                  if outcome_type(pm, subtask, label, script) == "silent")


def choose_silent_as_safe(pm: ProjectManager, dataset: str, subtask: str, scripts: list[str], labels: list[str],
                          ratio: float, seed: int = SILENT_AS_SAFE_SEED) -> list[Unit]:
    """round(ratio x n_silent) silent units of scripts x labels, drawn with its own RNG (the global `random` state,
    which draws the eval sample, stays unchanged)."""
    if not 0 <= ratio <= 1:
        raise ValueError(f"silent_as_safe ratio must be in [0, 1], got {ratio}")
    units = silent_units(pm, dataset, subtask, scripts, labels)
    return sorted(random.Random(seed).sample(units, round(ratio * len(units))))


def choose_batches(labels: list[str], n: int, seed: int = NUM_BATCHES_SEED) -> list[str]:
    """n batches drawn from `labels` (the split's train+eval labels) with its own RNG, for train and eval alike.

    The first n of one shuffle of the sorted labels, so the batches of a smaller n are a subset of those of a larger
    n, and n = len(labels) observes every label, as the main arm does."""
    if not 1 <= n <= len(labels):
        raise ValueError(f"num_batches must be between 1 and {len(labels)}, got {n}")
    order = sorted(labels)
    random.Random(seed).shuffle(order)
    return order[:n]


def optimization_is_safe(dataset: str, label: str, script: str, is_safe: bool,
                         relabelled: frozenset[Unit] = frozenset()) -> bool:
    """The label the optimization sees: safe for a relabelled unit, else the true corrupted_data_is_safe."""
    return True if (dataset, label, script) in relabelled else is_safe
