import pandas as pd

from prismadv.error_injection.corrupts.string_noise import StringNoise


def test_transpose_keeps_numbers_numeric():
    df = pd.DataFrame({"grade": [142.5, 118.3, 160.0, 127.9] * 10})
    out = StringNoise(columns=["grade"], severity=1.0, sampling="CAR", mode="transpose").transform(df)
    assert pd.api.types.is_float_dtype(out["grade"])
    changed = out["grade"] != df["grade"]
    assert changed.all()
    # Same digits, same place of the decimal point.
    for before, after in zip(df["grade"], out["grade"]):
        assert sorted(str(before)) == sorted(str(after))
        assert str(before).index(".") == str(after).index(".")


def test_transpose_swaps_two_letters_in_strings():
    df = pd.DataFrame({"target": ["Graduate", "Dropout", "Enrolled"] * 10})
    out = StringNoise(columns=["target"], severity=1.0, sampling="CAR", mode="transpose").transform(df)
    for before, after in zip(df["target"], out["target"]):
        assert before != after and sorted(before) == sorted(after)
        assert sum(a != b for a, b in zip(before, after)) == 2
