"""Column discovery functions (fixed, not optimized)."""

from typing import Dict, List

from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.llm.dspy.models.column_wise_module import ColumnDiscoveryModule
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager


def discover_columns_and_groups(
    project_manager: ProjectManager,
    subtask_name: str,
    script_name: str,
    processed_data_label: str = "0",
) -> Dict[str, List]:
    """
    Discover accessed columns and correlated groups using ColumnDiscoveryModule.
    
    This function is fixed and not optimized. It uses ColumnDiscoveryModule to find
    which columns are accessed in the script and which groups of columns are correlated.
    
    Args:
        project_manager: ProjectManager instance
        subtask_name: Name of the subtask
        script_name: Name of the script (without .py extension)
        processed_data_label: Label for processed data (default: "0" for training data)
        
    Returns:
        Dict with keys:
            - "columns_to_consider": List of column names
            - "correlated_groups": List of correlated group dicts
            - "column_desc_dict": Dict mapping column names to descriptions
            - "source_code": Source code string
            - "downstream_task_description": Task description string
    """
    from prismadv.dq_manager import DeequDataQualityManager
    
    # Load training data
    train_data = FileLoader.load_csv(
        project_manager.get_observed_data_path(subtask_name, processed_data_label)
    )
    dq_manager = DeequDataQualityManager()
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    
    try:
        # Get column descriptions
        column_desc_dict = DeequInspectorManager().spark_df_to_column_desc_dict(
            spark_train, spark_train_data
        )
        
        # Load script
        task_name = project_manager._metadata_manager.get_task_name_from_subtask(subtask_name)
        script_path = project_manager.get_script_path(task_name, script_name)
        source_code, assertions = FileLoader.load_py_file(script_path).extract_assertions()
        
        # Get downstream task description
        downstream_task_description = project_manager.get_subtask_description(
            subtask_name=subtask_name
        )
        
        # Use ColumnDiscoveryModule (fixed, not optimized)
        discovery_module = ColumnDiscoveryModule()
        discovery_result = discovery_module(
            code_script=source_code,
            column_desc_dict=column_desc_dict,
            downstream_task_description=downstream_task_description,
            use_column_correlation_detection=True
        )
        
        return {
            "columns_to_consider": discovery_result["columns_to_consider"],
            "correlated_groups": discovery_result["correlated_groups"],
            "column_desc_dict": column_desc_dict,
            "source_code": source_code,
            "downstream_task_description": downstream_task_description,
            "script_path": script_path,
        }
    finally:
        spark_train.sparkContext._gateway.close()
        spark_train.stop()
