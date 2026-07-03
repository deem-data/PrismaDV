"""Constraint validation functions."""

import logging
from collections import defaultdict
from typing import Dict

from prismadv.data_models import ValidationResults
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.data_models.constraints_v2 import ConstraintsWithSources
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager

# Suppress py4j error logs (they're handled internally by pydeequ)
logging.getLogger("py4j").setLevel(logging.CRITICAL)


def validate_constraints_on_training_data(
    project_manager: ProjectManager,
    subtask_name: str,
    processed_data_label: str,
    constraints_with_sources: ConstraintsWithSources,
    spark_session=None,
    session_reset_callback=None,
) -> ConstraintsWithSources:
    """
    Validate constraints on training data to set validity flags.
    
    Args:
        project_manager: ProjectManager instance
        subtask_name: Name of the subtask
        processed_data_label: Label for processed training data
        constraints_with_sources: ConstraintsWithSources to validate
        spark_session: Optional SparkSession to reuse. If None, creates a new session.
        session_reset_callback: Optional callback to reset the session if it becomes corrupted.
            Called if an exception occurs while using a shared session.
        
    Returns:
        ConstraintsWithSources with validity flags set
    """
    from prismadv.dq_manager import DeequDataQualityManager
    
    # Load training data
    train_data = FileLoader.load_csv(
        project_manager.get_observed_data_path(subtask_name, processed_data_label)
    )
    dq_manager = DeequDataQualityManager()
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data, spark_session=spark_session)
    
    try:
        # Collect all constraints to validate in batch
        all_code_entries = []
        for column_group in constraints_with_sources.data_map.keys():
            for code_entry in constraints_with_sources.data_map[column_group].code:
                all_code_entries.append(code_entry)

        # Validate all constraints together in batch
        constraint_strings = [entry.suggestion for entry in all_code_entries]

        try:
            validation_results = dq_manager.validate_constraints_with_reasons(
                spark_train, spark_train_data, constraint_strings, isolated_check=True
            )

            # Set validity and reason_if_invalid for each code entry
            for code_entry, (validity, reason_if_invalid) in zip(all_code_entries, validation_results):
                code_entry.validity = validity
                code_entry.reason_if_invalid = reason_if_invalid

        except Exception as batch_error:
            # Batch validation failed (likely due to py4j errors from malformed constraints)
            # Fall back to validating constraints one by one
            print(f"Batch validation failed: {batch_error}. Falling back to individual validation.")

            for code_entry in all_code_entries:
                try:
                    individual_result = dq_manager.validate_constraints_with_reasons(
                        spark_train, spark_train_data, [code_entry.suggestion], isolated_check=True
                    )
                    validity, reason_if_invalid = individual_result[0]
                    code_entry.validity = validity
                    code_entry.reason_if_invalid = reason_if_invalid
                except Exception as individual_error:
                    # This constraint has malformed syntax (e.g., missing lambda)
                    # Mark it as invalid with the error message
                    code_entry.validity = False
                    error_msg = str(individual_error)
                    if "TypeError" in error_msg and "not callable" in error_msg:
                        code_entry.reason_if_invalid = "Malformed constraint syntax: expected lambda function but got literal value"
                    else:
                        code_entry.reason_if_invalid = f"Validation error: {error_msg}"
                    print(f"Constraint '{code_entry.suggestion}' marked as invalid: {code_entry.reason_if_invalid}")

        return constraints_with_sources
    except Exception:
        # If using a shared session and an exception occurred, the session might be corrupted
        # Reset it so it gets recreated on next use
        if spark_session is not None and session_reset_callback is not None:
            try:
                session_reset_callback()
            except Exception:
                pass  # Ignore errors in reset callback
        raise  # Re-raise the original exception
    finally:
        # Only stop session if it was created here (not from pool)
        if spark_session is None:
            try:
                spark_train.sparkContext._gateway.close()
                spark_train.stop()
            except Exception:
                pass  # Session might already be stopped


def validate_constraints_on_test_data(
    project_manager: ProjectManager,
    subtask_name: str,
    processed_data_label: str,
    constraints_with_sources: ConstraintsWithSources,
    spark_session=None,
    session_reset_callback=None,
    clean: bool = False,
) -> ValidationResults:
    """
    Validate constraints on test data with the given processed_data_label.
    
    Args:
        project_manager: ProjectManager instance
        subtask_name: Name of the subtask
        processed_data_label: Label for processed test data
        constraints_with_sources: ConstraintsWithSources to validate
        spark_session: Optional SparkSession to reuse. If None, creates a new session.
        session_reset_callback: Optional callback to reset the session if it becomes corrupted.
            Called if an exception occurs while using a shared session.
        clean: Whether to use clean test data (default: False for corrupted data)
        
    Returns:
        ValidationResults object containing validation results
    """
    dq_manager = DeequDataQualityManager()
    
    # Load test data
    test_data_path = project_manager.get_new_test_data_path(
        subtask_name, processed_data_label, clean=clean
    )
    test_data = FileLoader.load_csv(test_data_path)
    spark_test_data, spark_test = dq_manager.spark_df_from_pandas_df(test_data, spark_session=spark_session)
    
    try:
        # Get valid code from constraints_with_sources
        valid_code_column_map = constraints_with_sources.get_suggestions_code_column_map(
            valid_only=True
        )
        code_list_for_constraints = list(valid_code_column_map.keys())
        
        # Validate all constraints together in batch
        validation_results_list = dq_manager.validate_constraints_with_reasons(
            spark_test, spark_test_data, code_list_for_constraints, isolated_check=True
        )
        
        # Create validation results
        result_dict = defaultdict(lambda: {"code": []})
        for code, (status, reason_if_failed) in zip(code_list_for_constraints, validation_results_list):
            code_info = valid_code_column_map[code]
            result_dict[code_info['column']]['code'].append({
                "suggestion": code,
                "status": status,
                "reason_if_failed": reason_if_failed,
                "level": code_info['level']
            })
        
        validation_results = ValidationResults.from_dict(result_dict)
        return validation_results
        
    except Exception:
        # If using a shared session and an exception occurred, the session might be corrupted
        # Reset it so it gets recreated on next use
        if spark_session is not None and session_reset_callback is not None:
            try:
                session_reset_callback()
            except Exception:
                pass  # Ignore errors in reset callback
        raise  # Re-raise the original exception
    finally:
        # Only stop session if it was created here (not from pool)
        if spark_session is None:
            try:
                spark_test.sparkContext._gateway.close()
                spark_test.stop()
            except Exception:
                pass  # Session might already be stopped
