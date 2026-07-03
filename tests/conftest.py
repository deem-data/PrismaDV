from prismadv.data_models import CodeEntry, CodeEntryV2, ColumnConstraints, \
    Constraints, ColumnConstraintsWithSources, ConstraintsWithSources, SourceLocation, AssumptionEntry
from prismadv.utils import get_current_folder, get_project_root
from prismadv.utils import load_dotenv

load_dotenv()

import pytest
from prismadv.dq_manager import DeequDataQualityManager


@pytest.fixture
def dq_manager():
    return DeequDataQualityManager()


@pytest.fixture
def resources_path():
    return get_current_folder() / "resources"


@pytest.fixture
def constraints_instance():
    code_entries = [
        CodeEntry(suggestion="Code 1", validity="Valid"),
        CodeEntry(suggestion="Code 2", validity="Invalid"),
    ]
    column_constraints = ColumnConstraints(code=code_entries, assumptions=["Assumption 1", "Assumption 2"])
    constraints = Constraints(constraints={"column1": column_constraints})
    return constraints


@pytest.fixture
def gx_expectation_path():
    return get_project_root() / "prismadv" / "llm" / "langchain" / "prompts" / "gx" / "expectations"


@pytest.fixture
def constraints_with_sources_instance():
    code_entries = [
        CodeEntryV2(suggestion="Code 1", validity=True, level="error"),
        CodeEntryV2(suggestion="Code 2", validity=False, level="warning"),
    ]
    source_location = [
        SourceLocation(file="file1.py", start_line=1, end_line=2),
        SourceLocation(file="file1.py", start_line=5, end_line=8),
    ]
    assumption_entry = AssumptionEntry(text="Assumption 1", sources=source_location)
    column_constraints = ColumnConstraintsWithSources(code=code_entries, assumptions=[assumption_entry])
    constraints = ConstraintsWithSources(data_map={"column1": column_constraints})
    return constraints
