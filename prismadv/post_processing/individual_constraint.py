from pydeequ.checks import *
from pydeequ.checks import ConstrainableDataTypes


from prismadv.dq_manager.deequ._constraint_validation import single_check


def determine_validity(constraint_str, spark, data_sample):
    reason_if_invalid = ""
    # Try to instantiate the check, filter out those that cannot be instantiated
    try:
        check = Check(spark, CheckLevel.Warning, "Check for data")
        exec(f"check.addConstraint(check{constraint_str})")
    except Exception as e:
        reason_if_invalid = f"Unable to instantiate constraint '{constraint_str}': {e}"
        return False, reason_if_invalid

    # Validate the constraint on the sample data
    check_result = single_check(spark, data_sample, constraint_str)
    if check_result["constraint_status"] == "Failure":
        reason_if_invalid = f"Constraint '{constraint_str}' failed on data sample: {check_result['constraint_message']}"
        return False, reason_if_invalid

    return True, reason_if_invalid
