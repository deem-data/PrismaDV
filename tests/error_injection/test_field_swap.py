import pandas as pd
import pytest

from prismadv.error_injection.corrupts import FieldSwap


@pytest.fixture
def df():
    return pd.DataFrame({"a": list(range(20)), "b": list(range(100, 120)), "g": [0, 1] * 10})


def test_swap_exchanges_values(df):
    out = FieldSwap(columns=["a", "b"], severity=1.0, strategy="swap").transform(df)
    assert out["a"].tolist() == df["b"].tolist()
    assert out["b"].tolist() == df["a"].tolist()


def test_copy_keeps_the_source(df):
    out = FieldSwap(columns=["a", "b"], severity=1.0, strategy="copy").transform(df)
    assert out["a"].tolist() == df["b"].tolist()
    assert out["b"].tolist() == df["b"].tolist()


def test_swap_only_in_the_slice(df):
    out = FieldSwap(columns=["a", "b"], severity=1.0, strategy="swap",
                    slice_column="g", slice_values=[1]).transform(df)
    changed = out["a"] != df["a"]
    assert changed.tolist() == (df["g"] == 1).tolist()


def test_needs_two_columns():
    with pytest.raises(ValueError):
        FieldSwap(columns=["a"], strategy="swap")
