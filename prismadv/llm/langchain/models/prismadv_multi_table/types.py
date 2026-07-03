from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet

from prismadv.data_models.constraints_v2 import SourceLocation


@dataclass(frozen=True, order=True)
class ColumnRef:
    table: str
    column: str

    @property
    def key(self) -> str:
        return f"{self.table}.{self.column}"

    def to_dict(self) -> dict[str, str]:
        return {"table": self.table, "column": self.column}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnRef":
        return cls(table=str(data["table"]), column=str(data["column"]))

    @classmethod
    def from_key(cls, key: str) -> "ColumnRef":
        table, separator, column = key.partition(".")
        if not separator or not table or not column:
            raise ValueError(f"invalid table-qualified column key: {key}")
        return cls(table=table, column=column)


@dataclass(frozen=True)
class ColumnGroupRef:
    """Same-table group of correlated columns.

    Cross-table groups are out of scope for the current EIDBench-real pipeline; the
    discovery filter rejects any group whose columns don't all live in `table`.
    """
    table: str
    columns: FrozenSet[str]
    correlation_type: str = ""

    @property
    def key(self) -> FrozenSet[str]:
        return frozenset(f"{self.table}.{column}" for column in self.columns)

    @property
    def column_group(self) -> FrozenSet[str]:
        return self.columns

    def to_column_refs(self) -> list[ColumnRef]:
        return [ColumnRef(self.table, column) for column in sorted(self.columns)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "columns": sorted(self.columns),
            "correlation_type": self.correlation_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnGroupRef":
        columns = data.get("correlated_columns") or data.get("columns") or []
        return cls(
            table=str(data["table"]),
            columns=frozenset(str(column) for column in columns),
            correlation_type=str(data.get("correlation_type", "")),
        )


@dataclass(frozen=True)
class CodeFile:
    path: str
    content: str


class MultiFileCodeContext:
    def __init__(self, files: list[dict[str, Any]]):
        self.files = [
            CodeFile(path=str(file_info["path"]), content=str(file_info["content"]))
            for file_info in files
        ]

    @classmethod
    def from_prismadv_inputs(cls, prismadv_inputs: dict[str, Any]) -> "MultiFileCodeContext":
        return cls(prismadv_inputs["code_context"]["files"])

    def with_line_numbers(self) -> str:
        return "\n\n".join(self._render_file(file.path, file.content) for file in self.files)

    def add_highlighted_line_numbers(self, source_locations: list[SourceLocation]) -> str:
        highlights = {}
        for source in source_locations:
            highlights.setdefault(source.file, set()).update(range(source.start_line, source.end_line + 1))
        return "\n\n".join(
            self._render_file(file.path, file.content, highlights.get(file.path, set()))
            for file in self.files
        )

    @staticmethod
    def _render_file(path: str, content: str, highlighted_lines: set[int] | None = None) -> str:
        highlighted_lines = highlighted_lines or set()
        rendered = [f"# File: {path}"]
        for line_number, line in enumerate(content.rstrip().splitlines(), start=1):
            prefix = "-**-> " if line_number in highlighted_lines else "      "
            rendered.append(f"{prefix}{line_number:04}: {line}")
        return "\n".join(rendered)
