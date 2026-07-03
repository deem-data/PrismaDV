from attr import dataclass

from prismadv.data_models import SourceLocation


@dataclass
class CodeInspection:
    """Data flow inspection for code snippets."""
    file_name: str
    target_column: str
    source_locations: list[SourceLocation]
    sink_variable: str = ""
