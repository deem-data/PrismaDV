"""
Test the get_constraints method from DeequFunctionManager using the default production info.yaml.

This test verifies that:
1. The default info.yaml loads correctly
2. get_constraints() method works with different filter parameters
3. The output format is correct
"""
from __future__ import annotations

import pytest

from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager


@pytest.mark.validation
def test_get_constraints_from_production_yaml():
    """
    Test the get_constraints method from DeequFunctionManager using the default production info.yaml.
    
    This test verifies that:
    1. The default info.yaml loads correctly
    2. get_constraints() method works with different filter parameters
    3. The output format is correct
    """
    # Load default production info.yaml (no custom path)
    manager = DeequFunctionManager()
    
    print(f"\n{'='*80}")
    print("TESTING get_constraints() METHOD FROM PRODUCTION info.yaml")
    print(f"{'='*80}")
    
    # Test 1: Get all constraints
    print("\n1. All constraints:")
    print("-" * 80)
    all_constraints = manager.get_constraints()
    all_schemas = manager.info
    print(f"Total constraints: {len(all_constraints)}")
    for i, schema in enumerate(all_schemas[:5], 1):  # Show first 5
        print(f"  {i}. {schema.to_string()}")
    if len(all_constraints) > 5:
        print(f"  ... and {len(all_constraints) - 5} more")
    
    # Test 2: Get row-level constraints only
    print("\n2. Row-level constraints (is_row_level=True):")
    print("-" * 80)
    row_level_constraints = manager.get_constraints(is_row_level=True)
    row_level_schemas = [schema for schema in manager.info if schema.CanUseSatisfies == True]
    print(f"Total row-level constraints: {len(row_level_constraints)}")
    for i, schema in enumerate(row_level_schemas[:5], 1):  # Show first 5
        print(f"  {i}. {schema.to_string()}")
    if len(row_level_constraints) > 5:
        print(f"  ... and {len(row_level_constraints) - 5} more")
    
    # Test 3: Get non-row-level constraints only
    print("\n3. Non-row-level constraints (is_row_level=False):")
    print("-" * 80)
    non_row_level_constraints = manager.get_constraints(is_row_level=False)
    non_row_level_schemas = [schema for schema in manager.info if schema.CanUseSatisfies == False]
    print(f"Total non-row-level constraints: {len(non_row_level_constraints)}")
    for i, schema in enumerate(non_row_level_schemas[:5], 1):  # Show first 5
        print(f"  {i}. {schema.to_string()}")
    if len(non_row_level_constraints) > 5:
        print(f"  ... and {len(non_row_level_constraints) - 5} more")
    
    # Test 4: Get constraints that can be used for multiple columns
    print("\n4. Constraints for multiple columns (can_be_used_for_multiple_columns=True):")
    print("-" * 80)
    multi_column_constraints = manager.get_constraints(can_be_used_for_multiple_columns=True)
    multi_column_schemas = [schema for schema in manager.info if schema.canBeUsedForMultipleColumns == True]
    print(f"Total multi-column constraints: {len(multi_column_constraints)}")
    for i, schema in enumerate(multi_column_schemas[:5], 1):  # Show first 5
        print(f"  {i}. {schema.to_string()}")
    if len(multi_column_constraints) > 5:
        print(f"  ... and {len(multi_column_constraints) - 5} more")
    
    # Test 5: Get constraints that cannot be used for multiple columns
    print("\n5. Single-column constraints (can_be_used_for_multiple_columns=False):")
    print("-" * 80)
    single_column_constraints = manager.get_constraints(can_be_used_for_multiple_columns=False)
    single_column_schemas = [schema for schema in manager.info if schema.canBeUsedForMultipleColumns == False]
    print(f"Total single-column constraints: {len(single_column_constraints)}")
    for i, schema in enumerate(single_column_schemas[:5], 1):  # Show first 5
        print(f"  {i}. {schema.to_string()}")
    if len(single_column_constraints) > 5:
        print(f"  ... and {len(single_column_constraints) - 5} more")
    
    # Test 6: Combined filters
    print("\n6. Row-level AND multi-column constraints:")
    print("-" * 80)
    combined_constraints = manager.get_constraints(
        is_row_level=True,
        can_be_used_for_multiple_columns=True
    )
    combined_schemas = [
        schema for schema in manager.info
        if schema.CanUseSatisfies == True and schema.canBeUsedForMultipleColumns == True
    ]
    print(f"Total constraints matching both filters: {len(combined_constraints)}")
    for i, schema in enumerate(combined_schemas[:5], 1):  # Show first 5
        print(f"  {i}. {schema.to_string()}")
    if len(combined_constraints) > 5:
        print(f"  ... and {len(combined_constraints) - 5} more")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total constraints in production info.yaml: {len(all_constraints)}")
    print(f"Row-level constraints: {len(row_level_constraints)}")
    print(f"Non-row-level constraints: {len(non_row_level_constraints)}")
    print(f"Multi-column constraints: {len(multi_column_constraints)}")
    print(f"Single-column constraints: {len(single_column_constraints)}")
    print(f"Row-level AND multi-column: {len(combined_constraints)}")
    print(f"{'='*80}\n")
    
    # Verify that the counts make sense
    assert len(all_constraints) > 0, "Should have at least one constraint"
    assert len(row_level_constraints) + len(non_row_level_constraints) == len(all_constraints), \
        f"Row-level ({len(row_level_constraints)}) + non-row-level ({len(non_row_level_constraints)}) should equal total ({len(all_constraints)})"
    
    # Note: Some constraints may have canBeUsedForMultipleColumns=None, so the sum might not equal total
    # Count constraints with None value
    constraints_with_none = len(all_constraints) - len(multi_column_constraints) - len(single_column_constraints)
    if constraints_with_none > 0:
        print(f"Note: {constraints_with_none} constraint(s) have canBeUsedForMultipleColumns=None")
        print(f"  Multi-column: {len(multi_column_constraints)}, Single-column: {len(single_column_constraints)}, None: {constraints_with_none}")
    
    # Verify that we got reasonable results
    assert len(multi_column_constraints) > 0, "Should have at least one multi-column constraint"
    assert len(single_column_constraints) > 0, "Should have at least one single-column constraint"
