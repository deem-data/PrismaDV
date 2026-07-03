"""Constraint generation functions (optimizable)."""

from typing import Dict, Optional

import oyaml as yaml

from prismadv.data_models.constraints_v2 import ConstraintsWithSources
from prismadv.llm.dspy.models.column_wise_module import (
    ConstraintGenerationModule,
)


def generate_constraints_for_column(
        constraint_module: ConstraintGenerationModule,
        column_name: str,
        column_desc_dict: Dict[str, Dict],
        source_code: str,
        downstream_task_description: str,
        sink_variable: str = "",
) -> Dict:
    """
    Generate constraints for a single column using ConstraintGenerationModule.
    
    This function uses the provided ConstraintGenerationModule (which can be optimized).
    
    Args:
        constraint_module: ConstraintGenerationModule instance (optimizable)
        column_name: Name of the column to generate constraints for
        column_desc_dict: Dict mapping column names to descriptions
        source_code: Source code string
        downstream_task_description: Task description string
        sink_variable: Sink variable name (default: "")
        
    Returns:
        Dict with keys:
            - "assumptions": List of AssumptionEntry objects
            - "code": List of CodeEntry objects
    """
    target_column_desc = yaml.dump(
        {column_name: column_desc_dict[column_name]},
        default_flow_style=False,
        sort_keys=False
    )

    result = constraint_module(
        code_script=source_code,
        target_column=column_name,
        target_column_desc=target_column_desc,
        downstream_task_description=downstream_task_description,
        sink_variable=sink_variable
    )

    return {
        "assumptions": result["assumptions"],
        "code": result["code"],
    }


def generate_constraints_for_column_from_example(
        constraint_module: ConstraintGenerationModule,
        training_example: Dict,
) -> Dict:
    """
    Generate constraints for a single column using a training example dict.
    
    This is a convenience wrapper around generate_constraints_for_column() that
    accepts a training example dict from prepare_single_column_training_dataset().
    
    Args:
        constraint_module: ConstraintGenerationModule instance (optimizable)
        training_example: Dict from prepare_single_column_training_dataset() containing:
            - "column_name": str
            - "column_desc_dict": Dict[str, Dict]
            - "source_code": str
            - "downstream_task_description": str
            
    Returns:
        Dict with keys:
            - "assumptions": List of AssumptionEntry objects
            - "code": List of CodeEntry objects
    """
    return generate_constraints_for_column(
        constraint_module=constraint_module,
        column_name=training_example["column_name"],
        column_desc_dict=training_example["column_desc_dict"],
        source_code=training_example["source_code"],
        downstream_task_description=training_example["downstream_task_description"],
        sink_variable="",  # Default empty string
    )


def combine_constraints(
        single_column_results: Optional[Dict[str, Dict]] = None,
        multi_column_results: Optional[Dict[frozenset, Dict]] = None,
) -> ConstraintsWithSources:
    """
    Combine single-column and multi-column constraint results into ConstraintsWithSources.
    
    Args:
        single_column_results: Optional dict mapping column names to result dicts.
            If None, single-column constraints are not included.
        multi_column_results: Optional dict mapping column groups (frozenset) to result dicts.
            If None, multi-column constraints are not included.
        
    Returns:
        ConstraintsWithSources object containing all constraints
        
    Raises:
        ValueError: If both single_column_results and multi_column_results are None
    """
    if single_column_results is None and multi_column_results is None:
        raise ValueError("At least one of single_column_results or multi_column_results must be provided")

    # Build assumptions and code dicts
    assumptions_dict = {}
    code_dict = {}

    # Add single-column constraints if provided
    if single_column_results is not None:
        for column_name, result in single_column_results.items():
            assumptions_dict[column_name] = result["assumptions"]
            code_dict[column_name] = result["code"]

    # Add multi-column constraints if provided
    if multi_column_results is not None:
        for column_group, result in multi_column_results.items():
            assumptions_dict[column_group] = result["assumptions"]
            code_dict[column_group] = result["code"]

    # Create ConstraintsWithSources
    constraints_with_sources = ConstraintsWithSources.from_assumptions_and_code_dict(
        assumptions_dict, code_dict
    )

    return constraints_with_sources
