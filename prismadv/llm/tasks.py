from enum import Enum, auto


class SequentialTADVTasks(Enum):
    """
    Enumeration of tasks for the Sequential TADV strategy.

    This strategy executes tasks in a fixed, linear order:
    1. COLUMN_ACCESS_DETECTION: Identify columns accessed by the downstream task.
    2. ASSUMPTION_EXTRACTION: Extract assumptions or expectations based on accessed columns.
    3. CODE_GENERATION: Generate constraint code from the extracted assumptions.
    """
    COLUMN_ACCESS_DETECTION = auto()
    ASSUMPTION_EXTRACTION = auto()
    CODE_GENERATION = auto()


class PrismaDVTasks(Enum):
    COLUMN_ACCESS_DETECTION = auto()
    COLUMN_CORRELATION_DISCOVERY = auto()

    SINGLE_COLUMN_ASSUMPTION_GENERATION = auto()
    MULTI_COLUMN_ASSUMPTION_GENERATION = auto()

    CODE_GENERATION = auto()
    MULTI_COLUMN_CODE_GENERATION = auto()

    CODE_FIXING = auto()
    CODE_CONSOLIDATION = auto()
