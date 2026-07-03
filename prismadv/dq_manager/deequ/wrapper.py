from threading import Lock
from typing import Optional

from pyspark.sql import SparkSession

from prismadv.data_models import Constraints, ValidationResults
from prismadv.dq_manager.abstract_data_quality_manager import AbstractDataQualityManager
from prismadv.dq_manager.deequ._analyzing import analyze_on_spark_df
from prismadv.dq_manager.deequ._constraint_suggestion import \
    get_suggestion_for_spark_df
from prismadv.dq_manager.deequ._constraint_validation import apply_checks_from_strings_on_spark_df
from prismadv.dq_manager.deequ._profiling import profile_on_spark_df
from prismadv.dq_manager.interfaces.constraint_suggestion import ConstraintSuggesting
from prismadv.dq_manager.deequ._constraint_validation import _normalize_check_str

class DeequDataQualityManager(AbstractDataQualityManager, ConstraintSuggesting):
    def __init__(self):
        super().__init__()
        # Single Spark session reused across all operations for efficiency
        self._spark_session: Optional[SparkSession] = None
        self._spark_lock = Lock()  # Thread-safe access to session

    def __del__(self):
        """Cleanup Spark session when manager is destroyed."""
        try:
            self.cleanup_spark_session()
        except Exception:
            pass  # Ignore errors during cleanup

    def get_or_create_spark_session(self) -> SparkSession:
        """
        Get or create a Spark session. The session is reused across
        all operations for efficiency.

        Returns:
            SparkSession instance
        """
        with self._spark_lock:
            if self._spark_session is not None:
                try:
                    # More robust health check: try to access sparkContext and create a simple operation
                    # This catches cases where the session appears alive but is actually corrupted
                    sc = self._spark_session.sparkContext
                    # Try a simple operation to verify the session is actually functional
                    _ = sc.version
                    return self._spark_session
                except Exception:
                    # Session is dead or corrupted, reset it
                    try:
                        self._spark_session.sparkContext._gateway.close()
                        self._spark_session.stop()
                    except Exception:
                        pass
                    self._spark_session = None

            # Create a new session
            import pandas as pd
            dummy_df = pd.DataFrame({"dummy": [1]})
            _, session = self.spark_df_from_pandas_df(dummy_df)
            self._spark_session = session
            return session

    def reset_spark_session(self):
        """
        Reset the Spark session. Called when a session becomes corrupted.
        """
        with self._spark_lock:
            if self._spark_session is not None:
                try:
                    self._spark_session.sparkContext._gateway.close()
                    self._spark_session.stop()
                except Exception:
                    pass  # Session might already be stopped
                self._spark_session = None

    def cleanup_spark_session(self):
        """
        Stop the Spark session. Call this when done with the manager
        to free up resources.
        """
        self.reset_spark_session()

    @staticmethod
    def analyze_on_spark_df(spark, spark_df, analyzers):
        return analyze_on_spark_df(spark, spark_df, analyzers)

    @staticmethod
    def profile_on_spark_df(spark, spark_df):
        """
        This function is based on the profiling function from Deequ. So it couldn't be implemented by great expectations.
        """
        return profile_on_spark_df(spark, spark_df)

    @staticmethod
    def apply_checks_from_strings_on_spark_df(spark, spark_df, code_list_for_constraints, isolated_check=True):
        return apply_checks_from_strings_on_spark_df(spark, spark_df, code_list_for_constraints, isolated_check)

    def validate_on_spark_df(self, spark, spark_df, code_list_for_constraints, return_raw=False, isolated_check=True):
        check_result = apply_checks_from_strings_on_spark_df(spark, spark_df,
                                                             code_list_for_constraints=code_list_for_constraints,
                                                             isolated_check=isolated_check)
        if return_raw:
            return check_result
        status = [item['constraint_status'] if
                  item is not None else None for item in check_result]
        return status

    def validate_constraints_with_reasons(self, spark, spark_df, code_list_for_constraints, isolated_check=True):
        """
        Validate multiple constraints and return validity and reason_if_invalid for each.
        
        This is a batched version of determine_validity that can validate multiple constraints
        together for better performance. It returns results in the same format as determine_validity:
        a list of (validity: bool, reason_if_invalid: str) tuples.
        
        Args:
            spark: Spark session
            spark_df: Spark DataFrame to validate against
            code_list_for_constraints: List of constraint strings to validate
            isolated_check: If True, each check is run in isolation (default: True)
            
        Returns:
            List of (validity: bool, reason_if_invalid: str) tuples, one per constraint
        """
        from pydeequ.checks import Check, CheckLevel, ConstrainableDataTypes
        
        # Pre-process constraints: filter and validate instantiation
        # Track which constraints need validation vs which are pre-filtered
        results = [None] * len(code_list_for_constraints)
        constraints_to_validate = []
        indices_to_validate = []

        for i, constraint_str in enumerate(code_list_for_constraints):
            # Try to instantiate the check to filter out invalid constraints early
            try:
                constraint_str = _normalize_check_str(constraint_str)
                check = Check(spark, CheckLevel.Warning, "Check for data")
                exec(f"check.addConstraint(check.{constraint_str})")
                constraints_to_validate.append(constraint_str)
                indices_to_validate.append(i)
            except Exception as e:
                reason_if_invalid = f"Unable to instantiate constraint '{constraint_str}': {e}"
                results[i] = (False, reason_if_invalid)
        
        # If no valid constraints remain, return early
        if not constraints_to_validate:
            return results
        
        # Validate the filtered constraints in batch
        check_results = apply_checks_from_strings_on_spark_df(
            spark, spark_df, constraints_to_validate, isolated_check=isolated_check
        )
        
        # Process validation results and fill in the results list
        for idx, check_result in enumerate(check_results):
            original_idx = indices_to_validate[idx]
            constraint_str = constraints_to_validate[idx]
            
            if check_result is None:
                validity = False
                reason_if_invalid = f"Constraint '{constraint_str}' returned None result"
            else:
                # Use dictionary-style access which works for both Spark Row objects and dicts
                # Same pattern as filter_valid_constraints_on_spark (line 150)
                try:
                    constraint_status = check_result["constraint_status"] if check_result is not None else None
                except (KeyError, AttributeError):
                    constraint_status = None
                
                # Get constraint_message, handling both dict and Spark Row
                try:
                    if isinstance(check_result, dict):
                        constraint_message = check_result.get("constraint_message", "")
                    else:
                        # Spark Row - use dictionary-style access
                        constraint_message = check_result["constraint_message"] if check_result is not None else ""
                except (KeyError, AttributeError):
                    constraint_message = ""
                
                if constraint_status == "Failure":
                    validity = False
                    reason_if_invalid = f"Constraint '{constraint_str}' failed on data sample: {constraint_message or 'Unknown error'}"
                else:
                    validity = True
                    reason_if_invalid = ""
            
            results[original_idx] = (validity, reason_if_invalid)
        
        return results

    def inference_constraints_for_spark_df(self, spark, spark_df, spark_validation=None,
                                           spark_validation_df=None) -> Constraints:
        """
        This function is based on the suggestion from Deequ. So it couldn't be implemented by great expectations.
        """
        suggestion = self._get_suggestion_for_spark_df(spark, spark_df)
        code_list_for_constraints = [item["code_for_constraint"] for item in suggestion]
        if spark_validation is None or spark_validation_df is None:
            code_list_for_constraints_valid = self.filter_valid_constraints_on_spark(code_list_for_constraints, spark,
                                                                                     spark_df)
        else:
            code_list_for_constraints_valid = self.filter_valid_constraints_on_spark(code_list_for_constraints,
                                                                                     spark_validation,
                                                                                     spark_validation_df)
        constraints = Constraints.from_deequ_output(suggestion, code_list_for_constraints_valid)
        return constraints

    def filter_valid_constraints_on_spark(self, code_list_for_constraints, spark,
                                          spark_df) -> list:
        check_result_on_original_validation_df = self.apply_checks_from_strings_on_spark_df(spark, spark_df,
                                                                                            code_list_for_constraints)
        status_on_original_validation_df = [item['constraint_status'] if
                                            item is not None else None for item in
                                            check_result_on_original_validation_df]
        # remove the constraints that are not grammarly correct
        code_list_for_constraints = [code_list_for_constraints[i] for i in range(len(code_list_for_constraints)) if
                                     status_on_original_validation_df[i] == "Success"]
        return code_list_for_constraints

    @staticmethod
    def build_validation_results(code_list_for_constraints, status, valid_code_column_map):
        code_status_map = {code_list_for_constraints[i]: status[i] for i in
                           range(len(code_list_for_constraints))}
        validation_results_dict = {"results": {column: {"code": []} for column in valid_code_column_map.values()}}
        for code, column in valid_code_column_map.items():
            validation_results_dict["results"][column]["code"].append(
                [code, "Passed" if code_status_map[code] == "Success" else "Failed"])
        validation_results = ValidationResults.from_dict(validation_results_dict)
        return validation_results

    @staticmethod
    def _get_suggestion_for_spark_df(spark, spark_df):
        return get_suggestion_for_spark_df(spark, spark_df)
