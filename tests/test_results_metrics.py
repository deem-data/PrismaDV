import pandas as pd
import pytest

from workflow_prismadv.results.combined_table import cell, number
from workflow_prismadv.results.metrics import binary_scores, ci_half_width, macro_scores, summary

# 6 units: 4 safe, 2 unsafe. Predicted safe (no alarm): safe 1-3 yes, safe 4 no, unsafe 1 yes, unsafe 2 no.
SAFE = [True, True, True, True, False, False]
PRED = [True, True, True, False, True, False]


def test_binary_scores_score_the_safe_class():
    s = binary_scores(SAFE, PRED)
    assert s["precision"] == pytest.approx(3 / 4)  # 3 of 4 predicted safe are safe
    assert s["recall"] == pytest.approx(3 / 4)  # 3 of 4 safe predicted safe
    assert s["f1"] == pytest.approx(3 / 4)


def test_binary_f1_equals_the_safe_f1_of_the_macro_scores():
    assert binary_scores(SAFE, PRED)["f1"] == pytest.approx(macro_scores(SAFE, PRED)["f1_safe"])


def test_binary_scores_are_zero_without_predicted_safe_units():
    assert binary_scores(SAFE, [False] * 6) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_ci_half_width_uses_the_t_quantile():
    assert ci_half_width(1.0, 3) == pytest.approx(4.302652729911275 / 3 ** 0.5)
    assert ci_half_width(0.0, 3) == 0.0


def test_summary_binary_averages_datasets_then_replicates():
    rows = []
    for rep, preds in ((1, PRED), (2, SAFE)):  # r2 predicts perfectly
        for ds in ("a", "b"):
            rows += [dict(benchmark="x", dataset=ds, unit=f"u{i}", arm="m", rep=rep, safe=s, alarm=not p)
                     for i, (s, p) in enumerate(zip(SAFE, preds))]
    s = summary(pd.DataFrame(rows), "binary").loc[("x", "m")]
    assert s["f1_mean"] == pytest.approx((0.75 + 1.0) / 2)
    assert s["f1_sd"] == pytest.approx(pd.Series([0.75, 1.0]).std())
    assert s["reps"] == 2


def test_number_pads_to_the_width_of_100():
    assert number(5.0) == (r"\phantom{00}", "5.0")
    assert number(56.7) == (r"\phantom{0}", "56.7")
    assert number(100.0) == ("", "100.0")


def test_cell_marks_rank_and_prints_the_interval():
    s = pd.DataFrame({"f1_mean": [0.9, 0.8], "f1_sd": [0.01, 0.02], "reps": [3, 3]},
                     index=pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")]))
    ranks = {"x": ["a", "b"]}
    assert cell(s, "x", "a", "f1", ranks) == r"\textbf{\phantom{0}90.0}{\scriptsize$\pm$1.0}"
    assert cell(s, "x", "b", "f1", ranks, wrap="rnone") == r"\rnone{\phantom{0}\underline{80.0}{\scriptsize$\pm$2.0}}"
    ci = 100 * ci_half_width(0.01, 3)
    assert cell(s, "x", "a", "f1", ranks, "ci95") == rf"\textbf{{\phantom{{0}}90.0}}{{\scriptsize$\pm${ci:.1f}}}"
    assert cell(s, "x", "c", "f1", ranks) == "-"
