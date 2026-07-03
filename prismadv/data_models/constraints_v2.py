import copy
import json
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from typing import Union, FrozenSet

import oyaml as yaml

ColumnGroup = Union[FrozenSet[str], str]


def ser_column_group_key(key: Union[str, FrozenSet[str], set, list, tuple]) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, (set, frozenset, list, tuple)):
        items = tuple(sorted(str(x) for x in key))
        if len(items) == 1:
            return items[0]
        return json.dumps(items, ensure_ascii=False, separators=(',', ':'))
    raise TypeError(f"Unsupported key type: {type(key).__name__}")


def de_column_group_key(s: Union[str, FrozenSet[str]]) -> Union[str, FrozenSet[str]]:
    if isinstance(s, (set, frozenset)):
        return frozenset(s)
    elif isinstance(s, str):
        if s.startswith('["') and s.endswith('"]'):
            items = json.loads(s)
            return frozenset(items)
        else:
            return s
    raise TypeError(f"Unsupported key type: {type(s).__name__}")


@dataclass
class CodeEntry:
    suggestion: str
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    validity: bool = None  # True for valid, False for invalid
    reason_if_invalid: str = ""  # Optional reason if invalid
    level: str = "undefined"  # Default level can be "error", "warning", or "info"
    source_assumptions: List[str] = field(default_factory=list)  # List of linked assumption UIDs

    @classmethod
    def from_dict(cls, data: Dict):
        # Validate that data is a dict and has required 'suggestion' field
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for CodeEntry, got {type(data)}: {data}")
        if "suggestion" not in data:
            raise ValueError(f"Missing required 'suggestion' field in CodeEntry: {data}")

        code_entry = cls(
            uid=data.get("uid", str(uuid.uuid4())),
            suggestion=data["suggestion"],
            level=data.get("level", "undefined")  # Default to "undefined" if not provided
        )
        if "validity" in data:
            code_entry.validity = data["validity"]
        if "reason_if_invalid" in data:
            code_entry.reason_if_invalid = data["reason_if_invalid"]
        if "source_assumptions" in data:
            code_entry.source_assumptions = data["source_assumptions"]
        return code_entry


@dataclass
class SourceLocation:
    start_line: int
    end_line: int
    file: str = ""  # Optional field for file name


@dataclass
class SourceLocations:
    sources: List[SourceLocation] = field(default_factory=list)

    def to_dict(self):
        return [source.__dict__ for source in self.sources]

    @classmethod
    def from_dict(cls, data: List[Dict]):
        sources = []
        for source in data:
            # Skip empty dictionaries or non-dict entries
            if not isinstance(source, dict) or not source:
                continue
            # Ensure both start_line and end_line are present
            if "start_line" not in source or "end_line" not in source:
                continue
            try:
                sources.append(SourceLocation(**source))
            except Exception:
                # Skip sources that fail to construct
                continue
        return cls(sources=sources)

    @classmethod
    def merge_sources(cls, *instances: "SourceLocations") -> "SourceLocations":
        # flatten all sources
        all_sources = []
        for inst in instances:
            all_sources.extend(inst.sources)
        # sort by (file, start_line)
        all_sources.sort(key=lambda s: (s.file, s.start_line, s.end_line))
        merged = []
        for s in all_sources:
            if not merged:
                merged.append(SourceLocation(s.start_line, s.end_line, s.file))
            else:
                last = merged[-1]
                if last.file == s.file and s.start_line <= last.end_line + 1:
                    # merge overlapping or adjacent
                    last.end_line = max(last.end_line, s.end_line)
                else:
                    merged.append(SourceLocation(s.start_line, s.end_line, s.file))
        return cls(sources=merged)

    def line_count(self):
        without_duplicates = set()
        for source in self.sources:
            for line in range(source.start_line, source.end_line + 1):
                without_duplicates.add((source.file, line))
        return len(without_duplicates)


@dataclass
class AssumptionEntry:
    text: str
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    sources: List[SourceLocation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict):
        # Validate that data is a dict and has required 'text' field
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for AssumptionEntry, got {type(data)}: {data}")
        if "text" not in data:
            raise ValueError(f"Missing required 'text' field in AssumptionEntry: {data}")

        # Parse sources with validation
        sources = []
        for source in data.get("sources", []):
            if not isinstance(source, dict):
                # Skip invalid source entries
                continue
            # Ensure both start_line and end_line are present
            if "start_line" not in source or "end_line" not in source:
                # Skip incomplete source entries
                continue
            try:
                sources.append(SourceLocation(**source))
            except Exception:
                # Skip sources that fail to construct
                continue

        uid = data.get("uid", str(uuid.uuid4()))
        return cls(uid=uid, text=data["text"], sources=sources)

    def to_string(self):
        return f"{self.text} (Sources: {', '.join([f'{source.file}:{source.start_line}-{source.end_line}' for source in self.sources])})"


@dataclass
class ColumnConstraintsWithSources:
    assumptions: List[AssumptionEntry] = field(default_factory=list)
    code: List[CodeEntry] = field(default_factory=list)
    table_name: Optional[str] = None
    column_group: Optional[ColumnGroup] = None


@dataclass
class ConstraintsWithSources:
    data_map: Dict[ColumnGroup, ColumnConstraintsWithSources] = field(default_factory=dict)

    def copy(self) -> "ConstraintsWithSources":
        return copy.deepcopy(self)

    def to_dict(self):
        # Convert the dataclass structure to a dictionary that yaml.dump can use
        return {
            "constraints": {
                ser_column_group_key(column_group): self._column_constraints_to_dict(column_group, constraint)
                for column_group, constraint in sorted(
                    self.data_map.items(),
                    key=lambda kv: ser_column_group_key(kv[0])
                )
            }
        }

    @staticmethod
    def qualified_key(column_group: ColumnGroup, table_name: Optional[str] = None) -> str:
        column_key = ser_column_group_key(column_group)
        if table_name:
            return f"{table_name}.{column_key}"
        return column_key

    @staticmethod
    def _effective_column_group(
            column_group: ColumnGroup,
            constraint: ColumnConstraintsWithSources
    ) -> ColumnGroup:
        return constraint.column_group or column_group

    @staticmethod
    def _column_constraints_to_dict(
            column_group: ColumnGroup,
            constraint: ColumnConstraintsWithSources
    ) -> Dict:
        result = {
            "code": [
                {
                    "suggestion": entry.suggestion,
                    "validity": entry.validity,
                    "reason_if_invalid": entry.reason_if_invalid,
                    "level": entry.level,
                    "uid": entry.uid,
                    "source_assumptions": entry.source_assumptions
                }
                for entry in constraint.code
            ],
            "assumptions": [
                {
                    "text": assumption.text,
                    "sources": [source.__dict__ for source in assumption.sources],
                    "uid": assumption.uid,
                }
                for assumption in constraint.assumptions
            ]
        }
        if constraint.table_name is not None:
            result["table_name"] = constraint.table_name
        if constraint.column_group is not None:
            result["column_group"] = ser_column_group_key(constraint.column_group)
        return result

    @classmethod
    def from_dict(cls, data: dict):
        constraints = cls()
        for column_group, constraint_data in data["constraints"].items():
            column_group = de_column_group_key(column_group)
            effective_column_group = constraint_data.get("column_group")
            if effective_column_group is not None:
                effective_column_group = de_column_group_key(effective_column_group)
            assumptions = [
                AssumptionEntry(
                    text=assumption["text"],
                    sources=[SourceLocation(**source) for source in assumption["sources"]],
                    uid=assumption.get("uid", str(uuid.uuid4()))
                )
                for assumption in constraint_data["assumptions"]
            ]
            code_entries = [
                CodeEntry(
                    suggestion=code["suggestion"],
                    validity=code["validity"],
                    reason_if_invalid=code.get("reason_if_invalid", ""),
                    level=code["level"],
                    uid=code.get("uid", str(uuid.uuid4())),
                    source_assumptions=code.get("source_assumptions", [])
                ) for code in constraint_data["code"]]
            constraints.data_map[column_group] = ColumnConstraintsWithSources(
                assumptions=assumptions,
                code=code_entries,
                table_name=constraint_data.get("table_name"),
                column_group=effective_column_group,
            )
        return constraints

    @classmethod
    def from_assumptions_and_code_dict(cls, assumptions_dict: Dict,
                                       code_dict: Dict):
        constraints = cls()
        assumptions_dict = {de_column_group_key(k): v for k, v in assumptions_dict.items()}
        code_dict = {de_column_group_key(k): v for k, v in code_dict.items()}
        for column_group, assumptions_list in assumptions_dict.items():
            if column_group not in code_dict:
                code_entries = [CodeEntry(suggestion="not provided", validity=False, level="undefined")]
            else:
                code_entries = code_dict[column_group]
            constraints.data_map[column_group] = ColumnConstraintsWithSources(
                assumptions=assumptions_list,
                code=code_entries,
            )
        return constraints

    def save_to_yaml(self, output_path: str):
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def to_yaml(self):
        return yaml.dump(self.to_dict())

    def to_string(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

    def get_suggestions_code_column_map(self, valid_only=False):
        return {
            code.suggestion: {
                "column": self._effective_column_group(column, constraint),
                "table": constraint.table_name,
                "level": code.level
            }
            for column, constraint in self.data_map.items()
            for code in constraint.code
            if not valid_only or code.validity == True
        }

    def group_by_table(
            self,
            default_table_name: Optional[str] = None
    ) -> Dict[Optional[str], "ConstraintsWithSources"]:
        grouped: Dict[Optional[str], ConstraintsWithSources] = {}
        for column_group, constraint in self.data_map.items():
            table_name = constraint.table_name or default_table_name
            if table_name not in grouped:
                grouped[table_name] = ConstraintsWithSources()
            effective_column_group = self._effective_column_group(column_group, constraint)
            grouped[table_name].data_map[effective_column_group] = ColumnConstraintsWithSources(
                assumptions=list(constraint.assumptions),
                code=list(constraint.code),
                table_name=table_name,
                column_group=effective_column_group,
            )
        return grouped

    def get_constraint_validity_info(self) -> dict:
        """
        Calculate validity statistics for constraints.
        
        Returns:
            Dict with keys:
                - valid_ratio: float, ratio of valid constraints (0.0 to 1.0)
                - total_count: int, total number of constraints
                - valid_count: int, number of valid constraints
                - invalid_count: int, number of invalid constraints
                - invalid_constraints: list of dicts with 'suggestion' and 'reason_if_invalid'
        """
        all_code_entries = []
        for column_group in self.data_map.keys():
            for code_entry in self.data_map[column_group].code:
                all_code_entries.append(code_entry)
        
        total_count = len(all_code_entries)
        if total_count == 0:
            return {
                "valid_ratio": 0.0,
                "total_count": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "invalid_constraints": []
            }
        
        valid_count = sum(1 for entry in all_code_entries if entry.validity is True)
        invalid_count = sum(1 for entry in all_code_entries if entry.validity is False)
        # Note: entries with validity=None are not counted as valid or invalid
        
        invalid_constraints = [
            {
                "suggestion": entry.suggestion,
                "reason_if_invalid": entry.reason_if_invalid or "No reason provided"
            }
            for entry in all_code_entries
            if entry.validity is False
        ]
        
        valid_ratio = valid_count / total_count if total_count > 0 else 0.0
        
        return {
            "valid_ratio": valid_ratio,
            "total_count": total_count,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "invalid_constraints": invalid_constraints
        }


if __name__ == "__main__":
    s1 = SourceLocations([SourceLocation(1, 3, "a.py")])
    s2 = SourceLocations([SourceLocation(2, 4, "a.py"), SourceLocation(10, 12, "a.py"), SourceLocation(2, 4, "a.py"),
                          SourceLocation(10, 13, "a.py")])
    s3 = SourceLocations([SourceLocation(20, 22, "a.py")])
    print(s1.line_count())  # 3
    print(s2.line_count())  # 6
    print(s3.line_count())  # 3

    merged = SourceLocations.merge_sources(s1, s2, s3)
    print(merged.to_dict())
    print(merged.line_count())
    # [{'start_line': 1, 'end_line': 4, 'file': 'a.py'},
    #  {'start_line': 10, 'end_line': 12, 'file': 'a.py'}]
