import numpy as np
import pandas as pd
import pytest

from prismadv.error_injection.corrupts import ColumnRename, FormatChange
from prismadv.error_injection.managers.sql_query import GeneralErrorInjectionManager


@pytest.fixture
def frame():
    rng = np.random.RandomState(0)
    n = 120
    return pd.DataFrame({
        "city": rng.choice(["Mumbai", "Delhi", "Pune"], size=n),
        "path": [f"/page/{i}" for i in range(n)],
        "ip": [f"10.1.2.{i % 250}" for i in range(n)],
        "when": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "amount": rng.uniform(100, 5000, size=n).round(2),
        "count": rng.randint(1, 50, size=n),
    })


def test_prefix_rewrites_every_row_and_keeps_the_meaning(frame):
    out = FormatChange(columns=["path"], severity=1, strategy="prefix", prefix="https://www.gov.uk").transform(frame)
    assert (out["path"] == "https://www.gov.uk" + frame["path"]).all()
    assert out["path"].str.replace("https://www.gov.uk", "", regex=False).equals(frame["path"])


def test_prefix_on_a_slice_touches_only_the_slice(frame):
    corrupt = FormatChange(columns=["ip"], severity=1, strategy="prefix", prefix="::ffff:",
                           slice_column="city", slice_values=["Pune"], random_state=1)
    out = corrupt.transform(frame)
    in_slice = frame["city"] == "Pune"
    assert (out.loc[in_slice, "ip"] == "::ffff:" + frame.loc[in_slice, "ip"]).all()
    assert (out.loc[~in_slice, "ip"] == frame.loc[~in_slice, "ip"]).all()


def test_date_format_renders_every_date_in_the_new_form_and_leaves_nulls(frame):
    frame.loc[3, "when"] = None
    out = FormatChange(columns=["when"], severity=1, strategy="date_format", date_format="%d/%m/%Y").transform(frame)
    assert out.loc[0, "when"] == "01/01/2024"
    assert pd.isna(out.loc[3, "when"])
    back = pd.to_datetime(out["when"].dropna(), format="%d/%m/%Y")
    assert back.equals(pd.to_datetime(frame["when"].dropna()))


def test_case_suffix_and_number_format(frame):
    upper = FormatChange(columns=["city"], severity=1, strategy="case", case="upper").transform(frame)
    assert (upper["city"] == frame["city"].str.upper()).all()
    suffixed = FormatChange(columns=["count"], severity=1, strategy="suffix", suffix=" units").transform(frame)
    assert suffixed.loc[0, "count"] == f"{frame.loc[0, 'count']} units"
    money = FormatChange(columns=["amount"], severity=1, strategy="number_format", number_format=",.2f").transform(frame)
    assert money.loc[0, "amount"] == format(frame.loc[0, "amount"], ",.2f")
    assert money["amount"].map(lambda s: float(s.replace(",", ""))).round(2).equals(frame["amount"].round(2))


def test_format_change_rejects_bad_parameters(frame):
    with pytest.raises(ValueError):
        FormatChange(columns=["path"], strategy="reverse")
    with pytest.raises(ValueError):
        FormatChange(columns=["path"], strategy="prefix")
    with pytest.raises(ValueError):
        FormatChange(columns=["city"], strategy="case", case="random")
    with pytest.raises(ValueError):
        FormatChange(columns=["amount"], strategy="number_format", number_format="not a spec")
    with pytest.raises(TypeError):
        FormatChange(columns=["path"], strategy="prefix", prefix="x", pattern="y")
    with pytest.raises(ValueError):
        FormatChange(columns=["missing"], strategy="prefix", prefix="x").transform(frame)


def test_format_change_is_seeded_at_partial_severity(frame):
    a = FormatChange(columns=["path"], severity=0.3, strategy="prefix", prefix="u:", random_state=5).transform(frame)
    b = FormatChange(columns=["path"], severity=0.3, strategy="prefix", prefix="u:", random_state=5).transform(frame)
    assert a.equals(b)
    assert (a["path"] != frame["path"]).sum() == int(len(frame) * 0.3)


def test_explicit_rename_keeps_values_and_order(frame):
    out = ColumnRename(columns=["count", "amount"], new_names=["n", "total"]).transform(frame)
    assert list(out.columns) == ["city", "path", "ip", "when", "total", "n"]
    assert out["n"].equals(frame["count"])
    assert out["total"].equals(frame["amount"])


def test_rename_strategies(frame):
    frame = frame.rename(columns={"count": "Order Count"})
    assert list(ColumnRename(columns=["Order Count"], strategy="snake_case").transform(frame).columns)[-1] == "order_count"
    assert list(ColumnRename(columns=["Order Count"], strategy="upper").transform(frame).columns)[-1] == "ORDER COUNT"
    assert list(ColumnRename(columns=["city"], strategy="add_prefix").transform(frame).columns)[0] == "new_city"
    assert list(ColumnRename(columns=["city"], strategy="add_prefix", prefix="src_").transform(frame).columns)[0] == "src_city"


def test_rename_rejects_collisions_and_no_ops(frame):
    with pytest.raises(ValueError):
        ColumnRename(columns=["count"], new_names=["amount"]).transform(frame)
    with pytest.raises(ValueError):
        ColumnRename(columns=["count"], new_names=["count"]).transform(frame)
    with pytest.raises(ValueError):
        ColumnRename(columns=["count", "amount"], new_names=["x"])
    with pytest.raises(ValueError):
        ColumnRename(columns=["count", "amount"], new_names=["x", "x"]).transform(frame)
    with pytest.raises(ValueError):
        ColumnRename(columns=["city"], strategy="upper").transform(frame.rename(columns={"path": "CITY"}))
    with pytest.raises(ValueError):
        ColumnRename(columns=["missing"], new_names=["x"]).transform(frame)


def test_saved_config_carries_only_the_parameters_that_were_set(frame):
    params = FormatChange(columns=["path"], severity=1, strategy="prefix", prefix="u:").to_dict()["FormatChange"]["Params"]
    assert params == {"severity": 1, "sampling": "CAR", "strategy": "prefix", "prefix": "u:"}
    params = ColumnRename(columns=["count"], new_names=["n"]).to_dict()["ColumnRename"]["Params"]
    assert params == {"severity": 1, "sampling": "CAR", "strategy": "explicit", "new_names": ["n"]}


def test_manager_loads_both_classes_from_a_config_and_sees_a_change(tmp_path, frame):
    raw = tmp_path / "files"
    raw.mkdir()
    frame.to_csv(raw / "table.csv", index=False)
    manager = GeneralErrorInjectionManager(raw_file_path=raw, target_table_name="table",
                                           processed_data_dir=tmp_path / "processed", sample_size=1.0)
    config = tmp_path / "config.yaml"
    config.write_text(
        "- FormatChange:\n    Columns:\n      - path\n    Params:\n      severity: 1\n"
        "      strategy: prefix\n      prefix: https://www.gov.uk\n"
        "- ColumnRename:\n    Columns:\n      - count\n    Params:\n      new_names:\n        - n\n"
    )
    corrupts = manager.load_error_injection_config(config)
    manager.seed_corrupts(corrupts, base_seed=26)
    manager.error_injection(corrupts)
    out = manager.post_corruption_new_data
    assert "n" in out.columns and "count" not in out.columns
    assert out["path"].str.startswith("https://www.gov.uk").all()
    # a rename alone is a change too
    only_rename = manager.load_error_injection_config(config)[1:]
    manager.seed_corrupts(only_rename, base_seed=27)
    manager.error_injection(only_rename)
    assert "n" in manager.post_corruption_new_data.columns
