"""Project manager module for handling project structure and operations.

This module provides both the main ProjectManager class and specialized
manager classes for fine-grained control over specific aspects of project management.
"""

from .abstract import AbstractProjectManager
from .labels import (CLEAN_LABEL, config_labels, config_path, corruption_variants, is_clean_label, is_label,
                     label_sort_key)
from .manager.base import ProjectManager
from .manager.multi_table import MultiTableProjectManager

# Export specialized managers for users who want fine-grained access
from .managers import (
    PathManager,
    MetadataManager,
    ScriptManager,
    DataManager,
    ConstraintManager,
    ExecutionManager,
    AnnotationManager,
    TrajectoryManager,
)

__all__ = [
    # The clean-baseline label convention
    "CLEAN_LABEL",
    "is_clean_label",
    "is_label",
    "label_sort_key",
    "config_path",
    "config_labels",
    "corruption_variants",
    # Main classes (backward compatibility)
    "AbstractProjectManager",
    "ProjectManager",
    "MultiTableProjectManager",
    # Specialized managers
    "PathManager",
    "MetadataManager",
    "ScriptManager",
    "DataManager",
    "ConstraintManager",
    "ExecutionManager",
    "AnnotationManager",
    "TrajectoryManager",
]
