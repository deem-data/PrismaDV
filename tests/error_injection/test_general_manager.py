import numpy as np
import pandas as pd
import pytest

from prismadv.error_injection.corrupts import (
    CrossColumnInconsistency,
    MaskValues,
    RangeViolation,
    RegexPatternViolation,
)
from prismadv.error_injection.managers.sql_query import GeneralErrorInjectionManager


@pytest.fixture
def manager(tmp_path):
    rng = np.random.RandomState(0)
    frame = pd.DataFrame({
        "EmpID": [f"RM{i:03d}" for i in range(100)],
        "Age": rng.randint(20, 60, size=100),
        "YearsAtCompany": rng.randint(0, 10, size=100),
        "TotalWorkingYears": rng.randint(10, 30, size=100),
        "Score": rng.uniform(0, 1, size=100),
    })
    raw = tmp_path / "files"
    raw.mkdir()
    frame.to_csv(raw / "table.csv", index=False)
    return GeneralErrorInjectionManager(
        raw_file_path=raw,
        target_table_name="table",
        processed_data_dir=tmp_path / "processed",
        sample_size=1.0,
    )


def test_label_is_the_config_index_not_a_directory_count(manager, tmp_path):
    (tmp_path / "processed" / "constraints").mkdir(parents=True)
    corrupts = [MaskValues(columns=["Age"], severity=0.1)]
    manager.seed_corrupts(corrupts, base_seed=7)
    manager.error_injection(corrupts)
    path = manager.save_data("7")
    assert path == tmp_path / "processed" / "7"
    assert (path / "error_injection_config.yaml").exists()
    assert manager.check_corrupts_existence([MaskValues(columns=["Age"], severity=0.1)], path)
    assert not manager.check_corrupts_existence([MaskValues(columns=["Age"], severity=0.2)], path)


def test_clean_baseline_has_no_corrupted_side(manager, tmp_path):
    path = manager.save_clean_data("0")
    assert (path / "files_with_clean_new_data" / "new_data.csv").exists()
    assert not (path / "files_with_corrupted_new_data").exists()
    assert not (path / "error_injection_config.yaml").exists()


def test_seeded_injection_is_reproducible(manager):
    def run():
        corrupts = [RangeViolation(columns=["Age"], severity=0.1, strategy="above_max")]
        manager.seed_corrupts(corrupts, base_seed=3)
        manager.error_injection(corrupts)
        assert corrupts[0].random_state == 300
        return manager.post_corruption_new_data

    first, second = run(), run()
    pd.testing.assert_frame_equal(first, second)


def test_noop_corruption_is_rejected(manager):
    # A string pattern on a column that never gets registered is exactly the
    # hr_analytics label-16 mistake; the class now expands it, so provoke the
    # no-op differently: severity 0 samples no rows.
    corrupts = [RangeViolation(columns=["Age"], severity=0.0)]
    with pytest.raises(RuntimeError, match="unchanged"):
        manager.error_injection(corrupts)


def test_integer_column_stays_integer_on_disk(manager, tmp_path):
    corrupts = [MaskValues(columns=["Age"], severity=0.1, random_state=1)]
    manager.error_injection(corrupts)
    path = manager.save_data("1")
    clean = (path / "files_with_clean_new_data" / "new_data.csv").read_text().splitlines()
    corrupted = (path / "files_with_corrupted_new_data" / "new_data.csv").read_text().splitlines()
    assert len(clean) == len(corrupted)
    changed = sum(1 for a, b in zip(clean, corrupted) if a != b)
    assert changed == int(len(manager.new_data) * 0.1)
    assert ".0," not in "\n".join(corrupted)


def test_range_violation_on_integer_column_is_integer_and_outside(manager):
    corrupts = [RangeViolation(columns=["Age"], severity=0.2, strategy="random", random_state=5)]
    manager.error_injection(corrupts)
    corrupted = manager.post_corruption_new_data["Age"]
    assert pd.api.types.is_integer_dtype(corrupted)
    clean = manager.new_data["Age"]
    touched = corrupted != clean
    assert touched.sum() == int(len(clean) * 0.2)
    assert ((corrupted[touched] < clean.min()) | (corrupted[touched] > clean.max())).all()


def test_cross_column_operator_alias_breaks_the_relation(manager):
    corrupts = [CrossColumnInconsistency(
        columns=["YearsAtCompany", "TotalWorkingYears"], severity=0.1,
        column_pairs=[["YearsAtCompany", "TotalWorkingYears", "<="]], random_state=2)]
    manager.error_injection(corrupts)
    out = manager.post_corruption_new_data
    violated = out["YearsAtCompany"] > out["TotalWorkingYears"]
    assert violated.sum() == int(len(out) * 0.1)
    assert pd.api.types.is_integer_dtype(out["TotalWorkingYears"])


def test_regex_string_pattern_applies_to_the_columns(manager):
    corrupts = [RegexPatternViolation(
        columns=["EmpID"], severity=0.1, pattern="^[A-Za-z0-9]+$",
        strategy="character_insertion", random_state=4)]
    manager.error_injection(corrupts)
    out = manager.post_corruption_new_data["EmpID"]
    assert (~out.str.fullmatch("[A-Za-z0-9]+")).sum() == int(len(out) * 0.1)


def test_unknown_parameter_is_rejected():
    with pytest.raises(TypeError, match="truncation_length"):
        MaskValues(columns=["Age"], severity=0.1, truncation_length=3)
    with pytest.raises(ValueError, match="relationship"):
        CrossColumnInconsistency(column_pairs=[["a", "b", "=<"]])
