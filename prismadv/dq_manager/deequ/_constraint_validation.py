import ast
import re
import warnings

from pydeequ import CheckLevel
from pydeequ import scala_utils as _scala_utils
from pydeequ.verification import *
from pydeequ.checks import *

warnings.filterwarnings("ignore", message="DataFrame constructor is internal.*")
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

_CHECK_DESCRIPTION = "Check for data"


def _patch_python_function_proxies():
    """Let the JVM hash pydeequ's Python-side scala.Function1/2 proxies.
    """
    for cls in (_scala_utils.ScalaFunction1, _scala_utils.ScalaFunction2):
        if not hasattr(cls, "hashCode"):
            cls.hashCode = lambda self: id(self) & 0x7FFFFFFF
            cls.equals = lambda self, other: self is other
            cls.toString = lambda self: f"<python {type(self).__name__} {id(self):x}>"


_patch_python_function_proxies()


def apply_checks_from_strings_on_spark_df(spark, spark_df, code_list_for_constraints, isolated_check):
    """
    Apply checks from strings on a Spark DataFrame.
    :param spark: Spark session.
    :param spark_df: Spark DataFrame to apply checks on.
    :param code_list_for_constraints: List of strings representing the checks to apply.
    :param isolated_check: If True, each check is run in isolation. It is recommended to set this when you are not sure whether all the checks are grammarly correct to avoid all checks failing without result.
    """
    if isolated_check:
        final_check_result = isolated_checks(spark, spark_df, code_list_for_constraints)
    else:
        check = Check(spark, CheckLevel.Warning, _CHECK_DESCRIPTION)
        try:
            for check_str in code_list_for_constraints:
                check_str = _normalize_check_str(check_str)
                exec(f"check.addConstraint(check.{check_str})")
            check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
            final_check_result = VerificationResult.checkResultsAsDataFrame(
                spark, check_result
            ).collect()
        except Exception as e:
            final_check_result = [
                {
                    "constraint_status": "Failure",
                    "constraint_message": str(e),
                }
            ]
    return final_check_result


def isolated_checks(spark, spark_df, code_list_for_constraints):
    """Run every constraint in its own Check, all Checks in one VerificationSuite.

    Returns the same list as ``[single_check(...) for each constraint]``: the first
    result row of the constraint's Check, or a failure dict when the constraint
    could not be instantiated or its run raised. One suite means one scan of the
    data per call instead of one Spark job per constraint, which is what makes
    the difference on the small tables validated here.

    Each Check gets its own description so result rows can be mapped back to
    their constraint (Deequ returns check results as an unordered map and merges
    equal Checks). If the shared run raises, the batch is halved until the
    offending constraints run alone and report their own error, as single_check
    would.
    """
    results = [None] * len(code_list_for_constraints)
    checks = []
    for i, check_str in enumerate(code_list_for_constraints):
        check = Check(spark, CheckLevel.Warning, f"{_CHECK_DESCRIPTION} [{i}]")
        try:
            check_str = _normalize_check_str(check_str)
            exec(f"check.addConstraint(check.{check_str})")
        except Exception as e:
            results[i] = _failure(check_str, e, stage="instantiate")
            continue
        checks.append((i, check, check_str))
    _run_checks(spark, spark_df, checks, results)
    return results


def _run_checks(spark, spark_df, checks, results):
    if not checks:
        return
    suite = VerificationSuite(spark).onData(spark_df)
    for _, check, _ in checks:
        suite = suite.addCheck(check)
    try:
        rows = VerificationResult.checkResultsAsDataFrame(spark, suite.run()).collect()
    except Exception as e:
        if len(checks) == 1:
            i, _, check_str = checks[0]
            results[i] = _failure(check_str, e, stage="run")
            return
        half = len(checks) // 2
        _run_checks(spark, spark_df, checks[:half], results)
        _run_checks(spark, spark_df, checks[half:], results)
        return
    first_row_by_check = {}
    for row in rows:
        first_row_by_check.setdefault(row["check"], row)
    for i, check, check_str in checks:
        row = first_row_by_check.get(check.description)
        if row is not None and _needs_isolated_rerun(row):
            row = single_check(spark, spark_df, check_str)
        results[i] = row


def _needs_isolated_rerun(row):
    """A failure that did not come from comparing a computed metric.

    Deequ evaluates the analyzers of all Checks in one aggregation query. When
    that query fails (a `satisfies` with unparsable SQL is enough) every analyzer
    in the batch is marked failed with the same exception, so the other
    constraints' verdicts are poison rather than measurements. Assertion
    failures on real metrics always start with "Value:", so anything else is
    re-run alone, where it either fails on its own account or passes.
    """
    if row["constraint_status"] != "Failure":
        return False
    message = row["constraint_message"] or ""
    return not message.startswith("Value:")


def _failure(check_str, error, stage):
    return {
        "constraint_status": "Failure",
        "constraint_message": str(error),
        "constraint_code": check_str,
        "error_stage": stage,
    }


def single_check(spark, spark_df, check_str):
    check = Check(spark, CheckLevel.Warning, _CHECK_DESCRIPTION)
    try:
        check_str = _normalize_check_str(check_str)
        exec(f"check.addConstraint(check.{check_str})")
        check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
        check_result = VerificationResult.checkResultsAsDataFrame(
            spark, check_result
        ).collect()[0]
    except Exception as e:
        check_result = {
            "constraint_status": "Failure",
            "constraint_message": str(e),
            "constraint_code": check_str,
        }
    return check_result


# Deequ pastes the column name of these methods into a SQL expression without quoting it, so a
# name such as 'Daily Steps' raises PARSE_SYNTAX_ERROR on every dataset. The value is the number
# of leading column arguments.
_SQL_COLUMN_METHODS = {"isNonNegative": 1, "isPositive": 1, "isLessThan": 2, "isLessThanOrEqualTo": 2,
                       "isGreaterThan": 2, "isGreaterThanOrEqualTo": 2}
# isContainedIn quotes the column itself, so a name that is already backticked breaks it.
_SELF_QUOTING_METHODS = {"isContainedIn": 1}
_STRING_ARG = re.compile(r"""\s*(?P<q>['"])(?P<name>(?:\\.|(?!(?P=q))[^\\])*)(?P=q)\s*(?:,|\))""")
_PLAIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


_METHOD_CALL = re.compile(r"(?:^|(?<=\.))\s*(?P<method>" + "|".join(_SQL_COLUMN_METHODS | _SELF_QUOTING_METHODS)
                          + r")\s*\(")


def _requote_columns(check_str, pos, n_columns, quote):
    """Backtick (quote=True) or unbacktick (quote=False) the first n string-literal column arguments after pos."""
    for _ in range(n_columns):
        m = _STRING_ARG.match(check_str, pos)
        if not m:
            break
        name = m["name"]
        if quote and not _PLAIN_NAME.fullmatch(name) and not name.startswith("`"):
            name = f"`{name}`"
        elif not quote and len(name) > 1 and name.startswith("`") and name.endswith("`"):
            name = name[1:-1]
        new = f"{m['q']}{name}{m['q']}"
        start, end = m.start("q"), m.end("name") + 1
        check_str = check_str[:start] + new + check_str[end:]
        pos = start + len(new) + (m.end() - end)
    return check_str


_MULTI_COMPLETENESS = {"haveCompleteness": "AND", "haveAnyCompleteness": "OR"}


def _rewrite_multi_completeness(check_str):
    """haveCompleteness / haveAnyCompleteness drop backticks, so a column name with a space can never work.

    Deequ defines them as the compliance of '(c1 IS NOT NULL) AND/OR (c2 IS NOT NULL) ...'; the same predicate
    is written here as a `satisfies` call with quoted names. Only a single, unchained call with a non-plain
    column name is rewritten.
    """
    try:
        call = ast.parse(check_str, mode="eval").body
    except SyntaxError:
        return check_str
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in _MULTI_COMPLETENESS
            and call.args and isinstance(call.args[0], (ast.List, ast.Tuple))
            and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in call.args[0].elts)):
        return check_str
    columns = [e.value.strip("`") for e in call.args[0].elts]
    if all(_PLAIN_NAME.fullmatch(c) for c in columns):
        return check_str
    predicate = f" {_MULTI_COMPLETENESS[call.func.id]} ".join(f"(`{c}` IS NOT NULL)" for c in columns)
    rest = [ast.get_source_segment(check_str, a) for a in call.args[1:]]
    rest += [f"{k.arg}={ast.get_source_segment(check_str, k.value)}" for k in call.keywords]
    return f"satisfies({predicate!r}, {call.func.id!r}" + "".join(f", {r}" for r in rest) + ")"


def _normalize_check_str(check_str):
    normalized = check_str if not check_str.startswith(".") else check_str[1:]
    normalized = _rewrite_multi_completeness(normalized)
    # Models chain calls ('isComplete(...).isContainedIn(...)'); rewrite every call, last first so
    # the earlier match positions stay valid.
    for m in reversed(list(_METHOD_CALL.finditer(normalized))):
        method = m["method"]
        if method in _SQL_COLUMN_METHODS:
            normalized = _requote_columns(normalized, m.end(), _SQL_COLUMN_METHODS[method], quote=True)
        else:
            normalized = _requote_columns(normalized, m.end(), _SELF_QUOTING_METHODS[method], quote=False)
    return normalized
