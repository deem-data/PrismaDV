from dataclasses import dataclass
from typing import Dict, List, Tuple, FrozenSet, Union

from prismadv.data_models.trajectory import DVTrajectoryColumnGroupSuite

# Type aliases for clarity
TrajectoryKey = Tuple[str, str, str, str]  # (dataset_name, subtask_name, script_name, llm_name)
ColumnGroup = Union[FrozenSet[str], str]
ConstraintUID = str
FailPrecisionScore = float


@dataclass
class SampledTrajectoryResult:
    """
    Result container for sampled trajectories with associated scores.

    Attributes:
        suites: List of sampled DVTrajectoryColumnGroupSuite objects
        overall_fail_precision: Overall fail precision score aggregated across all column groups
                               for this trajectory key. This represents the overall quality metric
                               for the (dataset, task, script, model) combination.
        constraint_scores: Dict mapping constraint UIDs to their fail precision scores.
                          Each unique constraint has one score, regardless of how many
                          trajectories share that constraint.
    """
    suites: List[DVTrajectoryColumnGroupSuite]
    overall_fail_precision: float
    constraint_scores: Dict[ConstraintUID, FailPrecisionScore]
