"""The synth codex runner reads constraints.json in the formats the staged checker accepts."""
from workflow_prismadv.eid_bench_baselines.pipelines.run_codex import normalize_raw_constraints

COLUMNS = {"Age", "Sleep Duration", "BMI Category"}


def test_dict_entries_pass_through():
    kept, stats = normalize_raw_constraints(
        [{"column_name": "Age", "code_for_constraint": ".isComplete('Age')"}], COLUMNS)
    assert kept == [{"column_name": "Age", "code_for_constraint": ".isComplete('Age')"}]
    assert stats["column_inferred"] == 0 and stats["plain_strings"] == 0


def test_missing_column_is_read_from_the_check():
    kept, stats = normalize_raw_constraints(
        [{"code_for_constraint": '.isNonNegative("Sleep Duration")'}], COLUMNS)
    assert kept[0]["column_name"] == "Sleep Duration" and stats["column_inferred"] == 1


def test_plain_strings_are_accepted():
    kept, stats = normalize_raw_constraints([".isComplete('BMI Category')"], COLUMNS)
    assert kept == [{"code_for_constraint": ".isComplete('BMI Category')", "column_name": "BMI Category"}]
    assert stats["plain_strings"] == 1 and stats["column_inferred"] == 1


def test_sql_expressions_name_their_column():
    columns = COLUMNS | {"Blood Pressure", "Sleep Disorder"}
    codes = [
        ".satisfies(\"`Blood Pressure` RLIKE '^[0-9]{2,3}/[0-9]{2,3}$'\", \"parseable\", lambda f: f == 1.0)",
        ".satisfies(\"`Sleep Disorder` IS NULL OR `Sleep Disorder` IN ('Insomnia')\", \"known\")",
        '.satisfies("Age >= 0 AND Age < 120", "plausible age")',
    ]
    kept, stats = normalize_raw_constraints(codes, columns)
    assert [k["column_name"] for k in kept] == ["Blood Pressure", "Sleep Disorder", "Age"]
    assert stats["dropped_no_column"] == 0


def test_checks_naming_no_known_column_are_dropped():
    kept, stats = normalize_raw_constraints([".hasSize(lambda n: n > 0)", {"column_name": "Age"}], COLUMNS)
    assert kept == [] and stats["dropped_no_column"] == 1 and stats["dropped_no_code"] == 1


def test_a_list_of_columns_is_read_from_the_check():
    kept, stats = normalize_raw_constraints(
        [{"column_name": ["Age", "BMI Category"], "code_for_constraint": ".isComplete('Age')"}], COLUMNS)
    assert kept[0]["column_name"] == "Age"
    assert stats["column_not_a_string"] == 1 and stats["column_inferred"] == 1
