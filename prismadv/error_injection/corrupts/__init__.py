# Existing corruptions
from prismadv.error_injection.corrupts.categorical_value_missing import MissingCategoricalValueCorruption
from prismadv.error_injection.corrupts.column_dropping import ColumnDropping
from prismadv.error_injection.corrupts.column_inserting import ColumnInserting
from prismadv.error_injection.corrupts.column_rename import ColumnRename
from prismadv.error_injection.corrupts.gaussian_noise import GaussianNoise
from prismadv.error_injection.corrupts.masking_values import MaskValues
from prismadv.error_injection.corrupts.row_inserting import DuplicatedRows
from prismadv.error_injection.corrupts.row_dropping import RowDropping
from prismadv.error_injection.corrupts.scaling_values import Scaling
from prismadv.error_injection.corrupts.string_noise import StringNoise
from prismadv.error_injection.corrupts.string_truncation import StringTruncation

# Data Quality Issues
from prismadv.error_injection.corrupts.outlier_injection import OutlierInjection
from prismadv.error_injection.corrupts.data_type_violation import DataTypeViolation
from prismadv.error_injection.corrupts.range_violation import RangeViolation
from prismadv.error_injection.corrupts.unique_constraint_violation import UniqueConstraintViolation
from prismadv.error_injection.corrupts.cross_column_inconsistency import CrossColumnInconsistency
from prismadv.error_injection.corrupts.value_replacement import ValueReplacement

# Format & Pattern Violations
from prismadv.error_injection.corrupts.date_format_corruption import DateFormatCorruption
from prismadv.error_injection.corrupts.format_change import FormatChange
from prismadv.error_injection.corrupts.regex_pattern_violation import RegexPatternViolation

# Statistical & Distribution Issues
from prismadv.error_injection.corrupts.distribution_shift import DistributionShift
from prismadv.error_injection.corrupts.imbalanced_categories import ImbalancedCategories

# Referential & Relational Issues
from prismadv.error_injection.corrupts.foreign_key_violation import ForeignKeyViolation
from prismadv.error_injection.corrupts.functional_dependency_violation import FunctionalDependencyViolation
from prismadv.error_injection.corrupts.field_swap import FieldSwap

__all__ = [
    # Existing
    "MissingCategoricalValueCorruption",
    "Scaling",
    "GaussianNoise",
    "ColumnInserting",
    "MaskValues",
    "ColumnDropping",
    "ColumnRename",
    "DuplicatedRows",
    "RowDropping",
    "StringNoise",
    "StringTruncation",
    # Data Quality
    "OutlierInjection",
    "DataTypeViolation",
    "RangeViolation",
    "UniqueConstraintViolation",
    "CrossColumnInconsistency",
    "ValueReplacement",
    # Format & Pattern
    "DateFormatCorruption",
    "FormatChange",
    "RegexPatternViolation",
    # Statistical & Distribution
    "DistributionShift",
    "ImbalancedCategories",
    # Referential & Relational
    "ForeignKeyViolation",
    "FunctionalDependencyViolation",
    "FieldSwap",
]
