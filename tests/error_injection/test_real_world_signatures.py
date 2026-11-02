import numpy as np
import pandas as pd
import pytest

from prismadv.error_injection.corrupts import (
    DistributionShift,
    MaskValues,
    RowDropping,
    Scaling,
    ValueReplacement,
)
from prismadv.error_injection.managers.sql_query import GeneralErrorInjectionManager


@pytest.fixture
def frame():
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "city": rng.choice(["Mumbai", "Delhi", "Pune", "Jaipur"], size=200),
        "runs": rng.randint(100, 200, size=200),
        "age": rng.randint(18, 60, size=200),
        "score": rng.uniform(0, 1, size=200),
    })


def test_slice_confines_the_corruption_to_the_slice(frame):
    corrupt = MaskValues(columns=["runs"], severity=0.5, slice_column="city",
                         slice_values=["Mumbai", "Delhi"], random_state=1)
    out = corrupt.transform(frame)
    changed = out["runs"].isna()
    in_slice = frame["city"].isin(["Mumbai", "Delhi"])
    assert changed.any()
    assert not (changed & ~in_slice).any(), "rows outside the slice were touched"
    assert changed.sum() == int(in_slice.sum() * 0.5)


def test_slice_at_full_severity_hits_every_slice_row(frame):
    corrupt = ValueReplacement(columns=["runs"], severity=1, strategy="constant", replacement_values=[-1],
                               slice_column="city", slice_values="Pune", random_state=1)
    out = corrupt.transform(frame)
    in_slice = frame["city"] == "Pune"
    assert (out.loc[in_slice, "runs"] == -1).all()
    assert (out.loc[~in_slice, "runs"] == frame.loc[~in_slice, "runs"]).all()


def test_slice_needs_both_parameters_and_an_existing_column(frame):
    with pytest.raises(ValueError):
        MaskValues(columns=["runs"], severity=0.5, slice_column="city")
    with pytest.raises(ValueError):
        MaskValues(columns=["runs"], severity=0.5, slice_column="region", slice_values=["x"]).transform(frame)
    with pytest.raises(ValueError):
        MaskValues(columns=["runs"], severity=0.5, slice_column="city", slice_values=["Atlantis"]).transform(frame)


def test_slice_is_recorded_in_the_saved_config(frame):
    corrupt = MaskValues(columns=["runs"], severity=0.5, slice_column="city", slice_values=["Mumbai"])
    params = corrupt.to_dict()["MaskValues"]["Params"]
    assert params["slice_column"] == "city"
    assert params["slice_values"] == ["Mumbai"]
    assert "slice_column" not in MaskValues(columns=["runs"], severity=0.5).to_dict()["MaskValues"]["Params"]


def test_block_row_dropping_removes_one_contiguous_run(frame):
    frame = frame.assign(row_id=range(len(frame)))
    corrupt = RowDropping(severity=0.1, strategy="block", random_state=3)
    out = corrupt.transform(frame)
    dropped = sorted(set(frame["row_id"]) - set(out["row_id"]))
    assert len(dropped) == 20
    assert dropped == list(range(dropped[0], dropped[0] + 20))
    assert list(out.columns) == list(frame.columns)


def test_row_dropping_is_seeded_and_random_mode_honours_the_slice(frame):
    frame = frame.assign(row_id=range(len(frame)))
    a = RowDropping(severity=0.1, strategy="block", random_state=3).transform(frame)
    b = RowDropping(severity=0.1, strategy="block", random_state=3).transform(frame)
    assert a.equals(b)
    corrupt = RowDropping(severity=0.5, strategy="random", slice_column="city", slice_values=["Jaipur"], random_state=4)
    out = corrupt.transform(frame)
    lost = frame.loc[~frame["row_id"].isin(out["row_id"])]
    assert (lost["city"] == "Jaipur").all()
    assert len(lost) == int((frame["city"] == "Jaipur").sum() * 0.5)


def test_exact_mean_shift_keeps_integers(frame):
    corrupt = DistributionShift(columns=["age"], severity=1, strategy="mean_shift", shift_value=2)
    out = corrupt.transform(frame)
    assert (out["age"] == frame["age"] + 2).all()
    assert pd.api.types.is_integer_dtype(out["age"])


def test_whole_column_constant_and_scale(frame):
    pinned = ValueReplacement(columns=["score"], severity=1, strategy="constant", replacement_values=[0]).transform(frame)
    assert (pinned["score"] == 0).all()
    scaled = Scaling(columns=["runs"], severity=1, factor=1.1).transform(frame)
    assert np.allclose(scaled["runs"], frame["runs"] * 1.1)


def test_manager_accepts_the_new_parameters_from_a_config(tmp_path, frame):
    raw = tmp_path / "files"
    raw.mkdir()
    frame.to_csv(raw / "table.csv", index=False)
    manager = GeneralErrorInjectionManager(raw_file_path=raw, target_table_name="table",
                                           processed_data_dir=tmp_path / "processed", sample_size=1.0)
    config = tmp_path / "config.yaml"
    config.write_text(
        "- RowDropping:\n    Columns: null\n    Params:\n      severity: 0.1\n      strategy: block\n"
        "- MaskValues:\n    Columns:\n      - runs\n    Params:\n      severity: 0.5\n"
        "      slice_column: city\n      slice_values:\n        - Mumbai\n"
    )
    corrupts = manager.load_error_injection_config(config)
    manager.seed_corrupts(corrupts, base_seed=26)
    manager.error_injection(corrupts)
    out = manager.post_corruption_new_data
    assert len(out) == len(manager.new_data) - int(len(manager.new_data) * 0.1)
    assert out["runs"].isna().any()
    assert not (out["runs"].isna() & (out["city"] != "Mumbai")).any()
