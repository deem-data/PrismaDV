"""
Test to verify that isolated_check=True and isolated_check=False produce equivalent results.
"""
import pandas as pd


def test_isolated_check_produces_same_results_as_bundled(dq_manager):
    """Verify that isolated and bundled modes produce identical results when all checks are valid."""
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    
    check_strings = [
        ".hasSize(lambda x: x >= 3)",
        ".isNonNegative('b')",
        ".hasMin('b', lambda x: x == 0)",
        ".isComplete('c')",
        ".isUnique('a')",
        ".isContainedIn('a', ['foo', 'bar', 'baz'])",
    ]
    
    # Run with isolated_check=False
    spark_df1, spark1 = dq_manager.spark_df_from_pandas_df(df)
    bundled_results = dq_manager.apply_checks_from_strings_on_spark_df(
        spark1, spark_df1, check_strings, isolated_check=False
    )
    spark1.sparkContext._gateway.shutdown_callback_server()
    spark1.stop()
    
    # Run with isolated_check=True
    spark_df2, spark2 = dq_manager.spark_df_from_pandas_df(df)
    isolated_results = dq_manager.apply_checks_from_strings_on_spark_df(
        spark2, spark_df2, check_strings, isolated_check=True
    )
    spark2.sparkContext._gateway.shutdown_callback_server()
    spark2.stop()
    
    # Verify same number of results
    assert len(bundled_results) == len(isolated_results), \
        f"Result count mismatch: bundled={len(bundled_results)}, isolated={len(isolated_results)}"
    
    # Verify each constraint status matches
    for i, (bundled, isolated) in enumerate(zip(bundled_results, isolated_results)):
        assert bundled["constraint_status"] == isolated["constraint_status"], \
            f"Constraint {i} status mismatch: bundled={bundled['constraint_status']}, isolated={isolated['constraint_status']}"
        assert bundled["constraint"] == isolated["constraint"], \
            f"Constraint {i} name mismatch"


def test_hasDataType_constraint(dq_manager):
    """Test that hasDataType constraint works correctly in both isolated and bundled modes."""
    # Create a DataFrame with string and integer columns
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [85.5, 90.0, 88.5]
    })
    
    check_strings = [
        ".hasDataType('name', ConstrainableDataTypes.String)",
        ".hasDataType('age', ConstrainableDataTypes.Integral)",
        ".hasDataType('score', ConstrainableDataTypes.Fractional)",
    ]
    
    # Run with isolated_check=False
    spark_df1, spark1 = dq_manager.spark_df_from_pandas_df(df)
    bundled_results = dq_manager.apply_checks_from_strings_on_spark_df(
        spark1, spark_df1, check_strings, isolated_check=False
    )
    spark1.sparkContext._gateway.shutdown_callback_server()
    spark1.stop()
    
    # Run with isolated_check=True
    spark_df2, spark2 = dq_manager.spark_df_from_pandas_df(df)
    isolated_results = dq_manager.apply_checks_from_strings_on_spark_df(
        spark2, spark_df2, check_strings, isolated_check=True
    )
    spark2.sparkContext._gateway.shutdown_callback_server()
    spark2.stop()
    
    # Verify same number of results
    assert len(bundled_results) == len(isolated_results), \
        f"Result count mismatch: bundled={len(bundled_results)}, isolated={len(isolated_results)}"
    
    # Verify each constraint status matches
    for i, (bundled, isolated) in enumerate(zip(bundled_results, isolated_results)):
        assert bundled["constraint_status"] == isolated["constraint_status"], \
            f"Constraint {i} status mismatch: bundled={bundled['constraint_status']}, isolated={isolated['constraint_status']}"
        assert bundled["constraint"] == isolated["constraint"], \
            f"Constraint {i} name mismatch"

