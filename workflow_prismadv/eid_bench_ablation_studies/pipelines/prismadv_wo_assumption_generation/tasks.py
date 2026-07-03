from enum import Enum, auto


class PrismaDVTasks(Enum):
    COLUMN_ACCESS_DETECTION = auto()
    COLUMN_CORRELATION_DISCOVERY = auto()

    SINGLE_DIRECT_CODE_GENERATION = auto()
    MULTI_DIRECT_CODE_GENERATION = auto()
