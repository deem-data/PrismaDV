from dataclasses import dataclass
from pathlib import PosixPath
from typing import Union, List

from prismadv.data_models import AssumptionEntry, ValidationCodeEntry
from prismadv.data_models.code_container import CodeContainer
from prismadv.data_models.constraints_v2 import CodeEntry


@dataclass
class DVTrajectory:
    dataset_name: str
    llm_name: str
    script_path: PosixPath
    script: CodeContainer
    data_path: PosixPath
    processed_data_label: str
    column_group: Union[str, frozenset]
    assumptions: List[AssumptionEntry]
    constraint: CodeEntry
    validation_results: ValidationCodeEntry
    is_safe: bool

    # TODO: add label: helpful or not helpful by comparing constraints validate with

    def __post_init__(self):
        if not isinstance(self.column_group, (str, frozenset)):
            raise TypeError(
                f"column_group must be str or frozenset, "
                f"got {type(self.column_group).__name__}"
            )

    def __repr__(self):
        return (
            f"DVTrajectory(dataset_name={self.dataset_name}, "
            f"llm_name={self.llm_name}, "
            f"data_path={self.data_path}, "
            f"processed_data_label={self.processed_data_label}, "
            f"column_group={self.column_group}, "
            f"assumptions={self.assumptions}, "
            f"constraint={self.constraint}, "
            f"validation_results={self.validation_results}, "
            f"is_safe={self.is_safe})"
        )


@dataclass
class DVTrajectoryColumnGroupSuite:
    column_group: Union[str, frozenset]
    trajectories: List[DVTrajectory]

    def to_dspy_example(self):
        pass
        # code_wo_assertion, assertions_sorted = self.trajectories[0].script.extract_assertions()
        # # return dspy.Example(
        # #     # add data profile
        # #     script=code_wo_assertion,
        # #     column_group=self.column_group,
        # #     # assumptions=self.assumptions,
        # #     constraints=self.constraints,
        # #     # validation_results=self.validation_results, # should be in evaluation stage
        # #     # is_safe=self.is_safe, # should be in evaluation stage
        # # ).with_inputs(script,)

    def metric_with_feedback(self):
        pass
