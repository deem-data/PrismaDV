from typing import List, Dict, Any

import dspy
import oyaml as yaml

from prismadv.llm.dspy.models.prismadv.steps.column_access_detection import ColumnAccessDetectionSig
from prismadv.llm.dspy.models.prismadv.steps.column_correlation import ColumnCorrelationDiscoverySig


class ColumnDiscoveryModule(dspy.Module):
    """
    Module for detecting accessed columns and discovering correlated column groups.
    
    This module performs two main tasks:
    1. Column Access Detection: Identifies which columns from the dataset are accessed in the code
    2. Column Correlation Discovery: Identifies groups of columns that are correlated and require joint constraints
    """

    def __init__(self):
        super().__init__()
        self.column_access_detector = dspy.Predict(ColumnAccessDetectionSig)
        self.column_correlation_discoverer = dspy.Predict(ColumnCorrelationDiscoverySig)

    def forward(
            self,
            code_script,
            column_desc_dict: Dict[str, Any],
            downstream_task_description: str,
            use_column_correlation_detection: bool = True
    ):
        """
        Detect accessed columns and discover correlated column groups.
        
        Args:
            code_script: CodeContainer or str - the code snippet to analyze
            column_desc_dict: Dict[str, Any] - dictionary mapping column names to their descriptions
            downstream_task_description: str - description of the downstream task
            use_column_correlation_detection: bool - whether to perform correlation discovery (default: True)
        
        Returns:
            dict with:
                - "columns_to_consider": List[str] - list of accessed column names
                - "correlated_groups": List[Dict[str, Any]] - list of correlated column groups (if enabled)
                    Each group has:
                    - "correlated_columns": List[str] - list of column names in the group
                    - "correlation_type": str - type of correlation
        """
        # Ensure code_script is a string
        if hasattr(code_script, '__str__'):
            code_script_str = str(code_script)
        else:
            code_script_str = code_script

        # Convert column_desc_dict to YAML format for column access detection
        columns_desc = yaml.dump(column_desc_dict, default_flow_style=False, sort_keys=False)

        # 1) Column Access Detection
        column_access_result = self.column_access_detector(
            columns_desc=columns_desc,
            code_script=code_script_str,
            downstream_task_description=downstream_task_description,
        )
        columns_to_consider = column_access_result.columns

        result = {
            "columns_to_consider": columns_to_consider,
        }

        # 2) Column Correlation Discovery (if enabled)
        if use_column_correlation_detection and len(columns_to_consider) > 1:
            # Filter column descriptions to only include accessed columns
            considered_columns_desc_dict = {
                col: column_desc_dict[col] 
                for col in column_desc_dict 
                if col in columns_to_consider
            }
            considered_columns_desc = yaml.dump(
                considered_columns_desc_dict, 
                default_flow_style=False, 
                sort_keys=False
            )

            correlation_result = self.column_correlation_discoverer(
                columns_to_consider=columns_to_consider,
                considered_columns_desc=considered_columns_desc,
                code_script=code_script_str,
                downstream_task_description=downstream_task_description,
            )

            # Filter out groups with only one column (they're not really correlated groups)
            correlated_groups = [
                group for group in correlation_result.correlated_groups
                if len(group.get("correlated_columns", [])) > 1
            ]

            result["correlated_groups"] = correlated_groups
        else:
            result["correlated_groups"] = []

        return result
