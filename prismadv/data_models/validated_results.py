from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union

import oyaml as yaml

from prismadv.data_models.constraints_v2 import ColumnGroup, ser_column_group_key, de_column_group_key


@dataclass
class ValidationCodeEntry:
    suggestion: str
    status: bool = None  # True for Passed, False for Failed, None for Not Run
    reason_if_failed: str = ""  # Optional reason if failed
    level: str = "undefined"  # Default level can be "error", "warning"


@dataclass
class ColumnValidationResults:
    code: List[ValidationCodeEntry] = field(default_factory=list)


@dataclass
class ValidationResults:
    results: Dict[ColumnGroup, ColumnValidationResults] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        instance = cls()
        for column, result in data.items():
            column_group = de_column_group_key(column)
            instance.results[column_group] = ColumnValidationResults(
                code=[
                    ValidationCodeEntry(
                        suggestion=entry['suggestion'],
                        status=entry['status'],
                        reason_if_failed=entry['reason_if_failed'],
                        level=entry.get('level', 'undefined')
                    )
                    for entry in result['code']
                ],
            )
        return instance

    def to_dict(self):
        return {
            "results": {
                ser_column_group_key(column): {
                    "code": [
                        {"suggestion": entry.suggestion, "status": entry.status,
                         "reason_if_failed": entry.reason_if_failed, "level": entry.level}
                        for entry in validation_result.code
                    ]
                }
                for column, validation_result in self.results.items()
            }
        }

    def save_to_yaml(self, output_path: str):
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def from_yaml(cls, input_path: Union[str, Path]):
        with open(input_path, "r") as f:
            data = yaml.safe_load(f)
        results = cls.from_dict(data["results"])
        return results

    def check_result(self, column_skipped=None, constraints_on_single_column_only=False) -> tuple[
        int, int, int, int, int]:
        num_passed_warning = 0
        num_failed_warning = 0
        num_passed_error = 0
        num_failed_error = 0
        num_non_compilable = 0

        column_skipped = [] if column_skipped is None else column_skipped

        if constraints_on_single_column_only:
            results_to_check = {
                column_name: column_result
                for column_name, column_result in self.results.items()
                if type(column_name) == str or (type(column_name) == frozenset and len(column_name) == 1)
            }
        else:
            results_to_check = self.results

        for column_name, column_result in results_to_check.items():
            if column_name in column_skipped:
                continue
            for entry in column_result.code:
                if "java.lang.OutOfMemoryError" in entry.reason_if_failed:
                    raise MemoryError("Validation process ran out of memory. Need to rerun with more memory allocated.")
                if "Unable to instantiate constraint" in entry.reason_if_failed:
                    num_non_compilable += 1
                    continue
                if entry.level == "warning" or entry.level == "undefined":
                    if entry.status:
                        num_passed_warning += 1
                    elif entry.status == False:
                        num_failed_warning += 1
                elif entry.level == "error":
                    if entry.status:
                        num_passed_error += 1
                    elif entry.status == False:
                        num_failed_error += 1
        return num_passed_warning, num_failed_warning, num_failed_error, num_passed_error, num_non_compilable

    def filter_by_status(self, status: bool = True):
        """
        Return a dict of {column: [ValidationCodeEntry, ...]} where each entry has the given status.
        Args:
            status (bool): True to return passed entries, False to return failed ones.
        """
        filtered = {}
        for column_name, column_result in self.results.items():
            matched = [entry for entry in column_result.code if entry.status is status]
            if matched:
                filtered[column_name] = matched
        return filtered

    def to_string(self, status: bool | None = None) -> str:
        """
        Return a readable YAML string of all results or only those with a given status.
        Args:
            status: True or False to filter by status, None for all.
        """
        if status is None:
            data = self.to_dict()
        else:
            filtered = self.filter_by_status(status)
            data = {
                "results": {
                    ser_column_group_key(column): {
                        "code": [
                            {
                                "suggestion": entry.suggestion,
                                "status": entry.status,
                                "reason_if_failed": entry.reason_if_failed,
                                "level": entry.level,
                            }
                            for entry in entries
                        ]
                    }
                    for column, entries in filtered.items()
                }
            }
        return yaml.dump(data, sort_keys=False, allow_unicode=True)

    def retrieve_by_column_and_suggestion(self, column: ColumnGroup,
                                          suggestion: str) -> Union[ValidationCodeEntry, None]:
        """
        Retrieve a specific ValidationCodeEntry by column and suggestion.
        Args:
            column (ColumnGroup): The column group to look into.
            suggestion (str): The suggestion text to match.
        Returns:
            ValidationCodeEntry if found, else None.
        """
        if column in self.results:
            for entry in self.results[column].code:
                if entry.suggestion == suggestion:
                    return entry
