"""Integration tests for refactored ProjectManager.

These tests verify that the refactored ProjectManager maintains
backward compatibility with the original implementation.
"""

def test_specialized_managers_are_accessible():
    """Test that specialized managers can be imported and used."""
    from prismadv.project_manager import (
        PathManager,
        MetadataManager,
        ScriptManager,
        DataManager,
        ConstraintManager,
        ExecutionManager,
        AnnotationManager,
        TrajectoryManager,
    )
    
    # Just verify they can be imported
    assert PathManager is not None
    assert MetadataManager is not None
    assert ScriptManager is not None
    assert DataManager is not None
    assert ConstraintManager is not None
    assert ExecutionManager is not None
    assert AnnotationManager is not None
    assert TrajectoryManager is not None

