from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union, Optional

import oyaml as yaml


@dataclass
class CodeEntry:
    suggestion: str
    validity: str  # "Valid" or "Invalid"


@dataclass
class ColumnConstraints:
    code: List[CodeEntry] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    table_name: Optional[str] = None
    column_name: Optional[str] = None


@dataclass
class Constraints:
    constraints: Dict[str, ColumnConstraints] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, input_path: Union[str, Path]):
        constraints = cls()
        constraints._load_from_yaml(input_path)
        return constraints

    def to_dict(self):
        # Convert the dataclass structure to a dictionary that yaml.dump can use
        return {
            "constraints": {
                column: self._column_constraints_to_dict(column, constraint)
                for column, constraint in sorted(self.constraints.items())
            }
        }

    @staticmethod
    def _qualified_key(column_name: str, table_name: Optional[str] = None) -> str:
        if table_name:
            return f"{table_name}.{column_name}"
        return column_name

    @staticmethod
    def _effective_column_name(column_key: str, constraint: ColumnConstraints) -> str:
        return constraint.column_name or column_key

    @staticmethod
    def _column_constraints_to_dict(column: str, constraint: ColumnConstraints):
        result = {
            "code": sorted([[entry.suggestion, entry.validity] for entry in constraint.code],
                           key=lambda x: x[0]),
            "assumptions": constraint.assumptions
        }
        if constraint.table_name is not None:
            result["table_name"] = constraint.table_name
        if constraint.column_name is not None:
            result["column_name"] = constraint.column_name
        return result

    @classmethod
    def from_deequ_output(cls, suggestion, code_list_for_constraints_valid, table_name: Optional[str] = None):
        accessed_columns = {
            cls._qualified_key(item["column_name"], item.get("table_name", table_name)): {
                "column_name": item["column_name"],
                "table_name": item.get("table_name", table_name),
            }
            for item in suggestion
        }
        yaml_dict = {
            "constraints": {
                key: {
                    "code": [],
                    "assumptions": [],
                    **{k: v for k, v in metadata.items() if v is not None}
                }
                for key, metadata in accessed_columns.items()
            }
        }
        for item in suggestion:
            code = item["code_for_constraint"]
            column_name = item["column_name"]
            key = cls._qualified_key(column_name, item.get("table_name", table_name))
            if code in code_list_for_constraints_valid:
                yaml_dict["constraints"][key]["code"].append([code, "Valid"])
            else:
                yaml_dict["constraints"][key]["code"].append([code, "Invalid"])
        return cls.from_dict(yaml_dict)

    @classmethod
    def from_dict(cls, data):
        constraints = cls()
        for column, constraint in data["constraints"].items():
            constraints.constraints[column] = ColumnConstraints(
                code=[CodeEntry(suggestion=suggestion, validity=validity) for suggestion, validity in
                      constraint["code"]],
                assumptions=constraint["assumptions"],
                table_name=constraint.get("table_name"),
                column_name=constraint.get("column_name"),
            )
        return constraints

    def save_to_yaml(self, output_path: str):
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def to_string(self):
        return yaml.dump(self.to_dict())

    def _load_from_yaml(self, input_path: str):
        with open(input_path, "r") as f:
            data = yaml.safe_load(f)
            for column, constraint in data["constraints"].items():
                self.constraints[column] = ColumnConstraints(
                    code=[CodeEntry(suggestion=suggestion, validity=validity) for suggestion, validity in
                          constraint["code"]],
                    assumptions=constraint["assumptions"],
                    table_name=constraint.get("table_name"),
                    column_name=constraint.get("column_name"),
                )

    def get_suggestions_code_column_map(self, valid_only=False):
        return {
            code.suggestion: {
                "column": self._effective_column_name(column, constraint),
                "table": constraint.table_name,
                "level": 'error'
            }
            for column, constraint in self.constraints.items()
            for code in constraint.code
            if not valid_only or code.validity == "Valid"
        }

    def group_by_table(self, default_table_name: Optional[str] = None) -> Dict[Optional[str], "Constraints"]:
        grouped: Dict[Optional[str], Constraints] = {}
        for column, constraint in self.constraints.items():
            table_name = constraint.table_name or default_table_name
            if table_name not in grouped:
                grouped[table_name] = Constraints()
            effective_column = self._effective_column_name(column, constraint)
            grouped[table_name].constraints[effective_column] = ColumnConstraints(
                code=list(constraint.code),
                assumptions=list(constraint.assumptions),
                table_name=table_name,
                column_name=effective_column,
            )
        return grouped

    def get_invalid_suggestions_column_code_map(self):
        return {
            self._qualified_key(self._effective_column_name(column, constraint), constraint.table_name): [
                code.suggestion for code in constraint.code if code.validity == "Invalid"
            ]
            for column, constraint in self.constraints.items()
        }
