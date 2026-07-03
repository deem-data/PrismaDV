import sys

from prismadv.utils import FilteredStream

sys.stdout = FilteredStream(sys.stdout)
sys.stderr = FilteredStream(sys.stderr)

from prismadv.utils import load_dotenv

load_dotenv()


def test_hasdatatype_not_skipped_in_validation_functions():
    """
    Direct unit test to verify that hasDataType constraints are not skipped in validation functions.

    This test directly calls the validation functions to ensure hasDataType constraints
    are processed through the normal validation path, not skipped.
    """
    from prismadv.dq_manager.deequ.wrapper import DeequDataQualityManager
    import pandas as pd

    # Create a simple test dataframe
    test_data = pd.DataFrame({"test_column": ["value1", "value2", "value3"]})

    dq_manager = DeequDataQualityManager()
    spark_df, spark = dq_manager.spark_df_from_pandas_df(test_data)

    try:
        str1 = ".hasDataType('test_column', ConstrainableDataTypes.String)"
        str2 = "hasDataType('test_column', ConstrainableDataTypes.String)"
        str3 = ".hasDataType('test_column', 'String')"
        str4 = ".hasDataType('test_column', ConstrainableDataTypes.Integral)"
        validation_results = dq_manager.validate_constraints_with_reasons(
            spark, spark_df, [str1, str2, str3, str4], isolated_check=False
        )

        assert len(validation_results) == 4
        validity, reason_if_invalid = validation_results[0]
        assert validity is True
        validity, reason_if_invalid = validation_results[1]
        assert validity is True
        validity, reason_if_invalid = validation_results[2]
        assert "Unable to instantiate constraint" in reason_if_invalid
        validity, reason_if_invalid = validation_results[3]
        assert "failed on data sample" in reason_if_invalid

    finally:
        spark.sparkContext._gateway.close()
        spark.stop()
