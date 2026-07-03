"""
Test constraint grammatical correctness - verify that constraints are syntactically valid.

This test focuses on whether constraints are grammatically correct (can be instantiated),
not whether they can be executed or pass/fail on the data.
"""

from __future__ import annotations

from __future__ import annotations

from typing import List

import pandas as pd
import pytest
from pydeequ.checks import *


def _build_student_dataframe(num_rows: int = 20000) -> pd.DataFrame:
    """Create a test dataframe with various column types to support different constraints."""
    targets = ["Graduate", "Enrolled", "Dropout", "Unknown"]
    rows = []
    for i in range(num_rows):
        credits_attempted = 24 + (i % 4) * 6
        credits_completed = max(0, credits_attempted - (i % 5))
        gpa = round(min(4.0, 2.6 + (i % 5) * 0.35 - (0.2 if i % 17 == 0 else 0)), 2)
        target = targets[i % len(targets)]

        # Add some test data for pattern/format constraints
        email = f"student{i}@university.edu" if i % 3 == 0 else f"user{i}@test.com"
        url = (
            f"https://example.com/student/{i}"
            if i % 5 == 0
            else f"http://test.org/page{i}"
        )
        # Credit card format (not real, just format)
        credit_card = (
            f"4532-{i%10000:04d}-{i%10000:04d}-{i%10000:04d}" if i % 7 == 0 else None
        )
        # SSN format (not real, just format)
        ssn = f"{i%1000:03d}-{i%100:02d}-{i%10000:04d}" if i % 11 == 0 else None

        # Add boolean column for testing Boolean data type
        is_active = i % 2 == 0

        rows.append(
            {
                "student_id": f"id_{i}",
                "credits_attempted": credits_attempted,
                "credits_completed": credits_completed,
                "gpa": None if i % 37 == 0 else gpa,
                "target": target,
                "email": email,
                "url": url,
                "credit_card": credit_card,
                "ssn": ssn,
                "is_active": is_active,  # Boolean column for testing
            }
        )
    return pd.DataFrame(rows)


def _get_all_constraints(sample_size: int) -> List[tuple[str, str]]:
    """
    Return hardcoded list of all constraint strings from test_info.yaml.

    Returns:
        List of tuples: (constraint_name, constraint_string)
    """
    return [
        # Size constraints
        ("hasSize", f".hasSize(lambda x: x >= {sample_size})"),
        # Completeness constraints
        ("isComplete", ".isComplete('student_id')"),
        (
            "hasCompleteness",
            ".hasCompleteness('credits_attempted', lambda x: x >= 0.95)",
        ),
        ("areComplete", ".areComplete(['student_id', 'target'])"),
        (
            "haveCompleteness",
            ".haveCompleteness(['credits_attempted', 'credits_completed'], lambda x: x >= 0.95)",
        ),
        ("areAnyComplete", ".areAnyComplete(['student_id', 'target'])"),
        (
            "haveAnyCompleteness",
            ".haveAnyCompleteness(['credits_attempted', 'credits_completed'], lambda x: x >= 0.95)",
        ),
        # Uniqueness constraints
        ("isUnique", ".isUnique('student_id')"),
        ("hasUniqueness", ".hasUniqueness(['student_id'], lambda x: x >= 0.99)"),
        ("hasDistinctness", ".hasDistinctness(['student_id'], lambda x: x >= 0.99)"),
        (
            "hasUniqueValueRatio",
            ".hasUniqueValueRatio(['student_id'], lambda x: x >= 0.99)",
        ),
        (
            "hasNumberOfDistinctValues",
            ".hasNumberOfDistinctValues('credits_attempted', lambda x: x >= 0, binningUdf=None, maxBins=1000)",
        ),
        # Statistical constraints
        (
            "hasMutualInformation",
            ".hasMutualInformation('credits_attempted', 'credits_completed', lambda x: x >= 0)",
        ),
        ("hasApproxQuantile", ".hasApproxQuantile('gpa', 0.5, lambda x: x >= 0)"),
        # Length constraints
        ("hasMinLength", ".hasMinLength('student_id', lambda x: x >= 3)"),
        ("hasMaxLength", ".hasMaxLength('student_id', lambda x: x <= 20)"),
        # Min/Max constraints
        ("hasMin", ".hasMin('credits_attempted', lambda x: x >= 18)"),
        ("hasMax", ".hasMax('credits_attempted', lambda x: x <= 48)"),
        ("hasMean", ".hasMean('gpa', lambda x: x >= 2.0 and x <= 3.5)"),
        (
            "hasStandardDeviation",
            ".hasStandardDeviation('gpa', lambda x: x >= 0.5 and x <= 1.5)",
        ),
        (
            "hasCorrelation",
            ".hasCorrelation('credits_attempted', 'credits_completed', lambda x: x >= 0)",
        ),
        # Custom satisfies constraints
        (
            "satisfies",
            ".satisfies('`credits_completed` <= `credits_attempted`', 'completed_le_attempted')",
        ),
        (
            "satisfies_with_assertion",
            ".satisfies('`credits_completed` <= `credits_attempted`', 'completed_le_attempted', lambda x: x >= 0.95)",
        ),
        # Pattern constraints
        ("hasPattern", ".hasPattern('student_id', '^id_\\\\d+$')"),
        (
            "hasPattern_with_assertion",
            ".hasPattern('student_id', '^id_\\\\d+$', lambda x: x >= 0.95)",
        ),
        # Format validation constraints
        ("containsCreditCardNumber", ".containsCreditCardNumber('credit_card')"),
        (
            "containsCreditCardNumber_with_assertion",
            ".containsCreditCardNumber('credit_card', lambda x: x >= 0)",
        ),
        ("containsEmail", ".containsEmail('email')"),
        ("containsEmail_with_assertion", ".containsEmail('email', lambda x: x >= 0)"),
        ("containsURL", ".containsURL('url')"),
        ("containsURL_with_assertion", ".containsURL('url', lambda x: x >= 0)"),
        ("containsSocialSecurityNumber", ".containsSocialSecurityNumber('ssn')"),
        (
            "containsSocialSecurityNumber_with_assertion",
            ".containsSocialSecurityNumber('ssn', lambda x: x >= 0)",
        ),
        # Data type constraints - test all ConstrainableDataTypes enum values
        ("hasDataType_String", ".hasDataType('student_id', ConstrainableDataTypes.String)"),
        (
            "hasDataType_String_with_assertion",
            ".hasDataType('student_id', ConstrainableDataTypes.String, lambda x: x >= 0.95)",
        ),
        ("hasDataType_Integral", ".hasDataType('credits_attempted', ConstrainableDataTypes.Integral)"),
        (
            "hasDataType_Integral_with_assertion",
            ".hasDataType('credits_attempted', ConstrainableDataTypes.Integral, lambda x: x >= 0.95)",
        ),
        ("hasDataType_Fractional", ".hasDataType('gpa', ConstrainableDataTypes.Fractional)"),
        (
            "hasDataType_Fractional_with_assertion",
            ".hasDataType('gpa', ConstrainableDataTypes.Fractional, lambda x: x >= 0.95)",
        ),
        ("hasDataType_Numeric", ".hasDataType('credits_attempted', ConstrainableDataTypes.Numeric)"),
        (
            "hasDataType_Numeric_with_assertion",
            ".hasDataType('credits_attempted', ConstrainableDataTypes.Numeric, lambda x: x >= 0.95)",
        ),
        ("hasDataType_Boolean", ".hasDataType('is_active', ConstrainableDataTypes.Boolean)"),
        (
            "hasDataType_Boolean_with_assertion",
            ".hasDataType('is_active', ConstrainableDataTypes.Boolean, lambda x: x >= 0.95)",
        ),
        ("hasDataType_Null", ".hasDataType('gpa', ConstrainableDataTypes.Null)"),
        (
            "hasDataType_Null_with_assertion",
            ".hasDataType('gpa', ConstrainableDataTypes.Null, lambda x: x >= 0)",
        ),
        # Non-negative/positive constraints
        ("isNonNegative", ".isNonNegative('credits_attempted')"),
        (
            "isNonNegative_with_assertion",
            ".isNonNegative('credits_attempted', lambda x: x >= 0.95)",
        ),
        ("isPositive", ".isPositive('credits_attempted')"),
        (
            "isPositive_with_assertion",
            ".isPositive('credits_attempted', lambda x: x >= 0.95)",
        ),
        # Comparison constraints
        ("isLessThan", ".isLessThan('credits_completed', 'credits_attempted')"),
        (
            "isLessThan_with_assertion",
            ".isLessThan('credits_completed', 'credits_attempted', lambda x: x >= 0.95)",
        ),
        (
            "isLessThanOrEqualTo",
            ".isLessThanOrEqualTo('credits_completed', 'credits_attempted')",
        ),
        (
            "isLessThanOrEqualTo_with_assertion",
            ".isLessThanOrEqualTo('credits_completed', 'credits_attempted', lambda x: x >= 0.95)",
        ),
        ("isGreaterThan", ".isGreaterThan('credits_attempted', 'credits_completed')"),
        (
            "isGreaterThan_with_assertion",
            ".isGreaterThan('credits_attempted', 'credits_completed', lambda x: x >= 0.95)",
        ),
        (
            "isGreaterThanOrEqualTo",
            ".isGreaterThanOrEqualTo('credits_attempted', 'credits_completed')",
        ),
        (
            "isGreaterThanOrEqualTo_with_assertion",
            ".isGreaterThanOrEqualTo('credits_attempted', 'credits_completed', lambda x: x >= 0.95)",
        ),
        # Containment constraints
        (
            "isContainedIn",
            ".isContainedIn('target', ['Graduate', 'Dropout', 'Enrolled'])",
        ),
        (
            "isContainedIn_with_assertion",
            ".isContainedIn('target', ['Graduate', 'Dropout', 'Enrolled'], lambda x: x >= 0.95)",
        ),
    ]


@pytest.mark.validation
def test_constraints_are_grammatically_correct(dq_manager):
    """
    Test that all hardcoded constraints are grammatically correct (syntactically valid).

    This test verifies:
    1. Constraints can be instantiated (syntax is valid)
    2. Constraints are grammatically correct, not whether they execute successfully

    The test passes if constraints can be instantiated, regardless of execution results.
    """
    df = _build_student_dataframe()
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)

    # Get all hardcoded constraints
    constraint_schemas = _get_all_constraints(sample_size=len(df))
    constraint_list = [constraint_str for _, constraint_str in constraint_schemas]
    schema_names = [schema_name for schema_name, _ in constraint_schemas]

    try:
        print(
            f"\nTesting grammatical correctness of {len(constraint_list)} hardcoded constraints..."
        )

        # Validate all constraints to check grammatical correctness
        results = dq_manager.validate_constraints_with_reasons(
            spark,
            spark_df,
            constraint_list,
            isolated_check=True,
        )

        # Verify results structure
        assert len(results) == len(
            constraint_list
        ), f"Expected {len(constraint_list)} results, got {len(results)}"

        # Check each constraint for grammatical correctness
        grammatically_correct_count = 0
        grammatically_incorrect_count = 0
        grammatically_incorrect_constraints = []

        for i, (
            (schema_name, constraint_str),
            (validity, reason_if_invalid),
        ) in enumerate(zip(constraint_schemas, results)):
            # A constraint is grammatically correct if it can be instantiated
            # We only check for instantiation errors, not execution errors
            
            is_grammatically_correct = True

            # Check for instantiation errors (grammatical correctness)
            if (
                reason_if_invalid
                and "Unable to instantiate constraint" in reason_if_invalid
            ):
                is_grammatically_correct = False
                grammatically_incorrect_constraints.append(
                    {
                        "schema_name": schema_name,
                        "constraint": constraint_str,
                        "reason": reason_if_invalid,
                        "issue": "grammatical_error",
                    }
                )

            if is_grammatically_correct:
                grammatically_correct_count += 1
            else:
                grammatically_incorrect_count += 1

        # Print summary
        print(f"\n{'='*80}")
        print("GRAMMATICAL CORRECTNESS TEST RESULTS")
        print(f"{'='*80}")
        print(f"Total constraints tested: {len(constraint_list)}")
        print(f"Grammatically correct constraints: {grammatically_correct_count}")
        print(f"Grammatically incorrect constraints: {grammatically_incorrect_count}")

        if grammatically_incorrect_constraints:
            print(f"\nGrammatically incorrect constraints:")
            for item in grammatically_incorrect_constraints:
                print(f"  Schema: {item['schema_name']}")
                print(f"  Constraint: {item['constraint']}")
                print(f"  Issue: {item['issue']}")
                print(f"  Reason: {item['reason']}")
                print()

        print(f"{'='*80}\n")

        # Report results - test fails only if constraints are grammatically incorrect
        if grammatically_incorrect_count > 0:
            pytest.fail(
                f"Found {grammatically_incorrect_count} grammatically incorrect constraints out of {len(constraint_list)}. "
                f"See output above for details. "
                f"Grammatically correct: {grammatically_correct_count}/{len(constraint_list)} ({grammatically_correct_count/len(constraint_list)*100:.1f}%)"
            )

        # Additional check: verify result structure is correct
        for i, (validity, reason_if_invalid) in enumerate(results):
            assert isinstance(
                validity, bool
            ), f"Result {i} validity should be bool, got {type(validity)}"
            assert isinstance(
                reason_if_invalid, str
            ), f"Result {i} reason_if_invalid should be str, got {type(reason_if_invalid)}"

    finally:
        spark.sparkContext._gateway.shutdown_callback_server()
        spark.stop()


@pytest.mark.validation
def test_individual_constraint_grammatical_correctness(dq_manager):
    """
    Test individual hardcoded constraints one by one to isolate any grammatical issues.

    This helps identify which specific constraints are grammatically incorrect.
    """
    df = _build_student_dataframe()
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)

    # Get all hardcoded constraints
    constraint_schemas = _get_all_constraints(sample_size=len(df))

    try:
        print(
            f"\nTesting individual grammatical correctness of {len(constraint_schemas)} hardcoded constraints..."
        )

        grammatically_incorrect_constraints = []

        for schema_name, constraint_str in constraint_schemas:
            try:
                # Test each constraint individually
                results = dq_manager.validate_constraints_with_reasons(
                    spark,
                    spark_df,
                    [constraint_str],
                    isolated_check=True,
                )

                # Check if validation succeeded
                assert (
                    len(results) == 1
                ), f"Expected 1 result for constraint '{constraint_str}', got {len(results)}"

                validity, reason_if_invalid = results[0]

                # Check for grammatical errors (instantiation errors only)
                if reason_if_invalid:
                    if "Unable to instantiate constraint" in reason_if_invalid:
                        grammatically_incorrect_constraints.append(
                            {
                                "schema_name": schema_name,
                                "constraint": constraint_str,
                                "reason": reason_if_invalid,
                                "type": "grammatical_error",
                            }
                        )

                # Verify result structure
                assert isinstance(
                    validity, bool
                ), f"Validity should be bool for '{constraint_str}'"
                assert isinstance(
                    reason_if_invalid, str
                ), f"Reason should be str for '{constraint_str}'"

            except Exception as e:
                grammatically_incorrect_constraints.append(
                    {
                        "schema_name": schema_name,
                        "constraint": constraint_str,
                        "reason": f"Exception during validation: {str(e)}",
                        "type": "exception",
                    }
                )

        # Print results
        print(f"\n{'='*80}")
        print("INDIVIDUAL GRAMMATICAL CORRECTNESS TEST RESULTS")
        print(f"{'='*80}")
        print(f"Total constraints tested: {len(constraint_schemas)}")
        print(f"Grammatically incorrect constraints: {len(grammatically_incorrect_constraints)}")

        if grammatically_incorrect_constraints:
            print(f"\nGrammatically incorrect constraints:")
            for item in grammatically_incorrect_constraints:
                print(f"  Schema: {item['schema_name']}")
                print(f"  Constraint: {item['constraint']}")
                print(f"  Type: {item['type']}")
                print(f"  Reason: {item['reason']}")
                print()
        else:
            print("\n✓ All constraints are grammatically correct!")

        print(f"{'='*80}\n")

        # Report results - test fails only if constraints are grammatically incorrect
        if len(grammatically_incorrect_constraints) > 0:
            pytest.fail(
                f"Found {len(grammatically_incorrect_constraints)} grammatically incorrect constraints out of {len(constraint_schemas)}. "
                f"See output above for details. "
                f"Grammatically correct: {len(constraint_schemas) - len(grammatically_incorrect_constraints)}/{len(constraint_schemas)} "
                f"({(len(constraint_schemas) - len(grammatically_incorrect_constraints))/len(constraint_schemas)*100:.1f}%)"
            )

    finally:
        spark.sparkContext._gateway.shutdown_callback_server()
        spark.stop()
