"""Agent output with a missing column is recovered from the check, not thrown away."""
from workflow_prismadv.eid_bench_real_baselines.single_shot import column_from_code, normalize_raw_constraints

PROFILES = {
    "orders": {"columns": [{"name": "order_id"}, {"name": "amount"}, {"name": "country"}]},
    "customers": {"columns": [{"name": "customer_id"}, {"name": "name"}]},
}


def test_column_is_read_from_the_check_when_the_agent_leaves_it_out(capsys):
    raw = [
        {"table_name": "orders", "code_for_constraint": '.isComplete("amount")'},                  # inferred
        {"table_name": "orders", "code_for_constraint": ".hasUniqueness(['order_id', 'country'], lambda x: x == 1.0)"},
        {"table_name": "orders", "code_for_constraint": ".hasSize(lambda x: x > 0)"},              # table-level: dropped
        {"table_name": "orders", "column_name": "amount", "code_for_constraint": ".isPositive('amount')"},
        {"table_name": "customers", "code_for_constraint": '.isComplete("missing_col")'},          # not a column: dropped
        {"table_name": "shipments", "column_name": "x", "code_for_constraint": ".isComplete('x')"},  # unknown table
    ]
    out = normalize_raw_constraints(raw, PROFILES)
    assert [c["column_name"] for c in out["orders"]] == ["amount", "order_id", "amount"]
    assert "customers" not in out and "shipments" not in out
    report = capsys.readouterr().out
    assert "kept 3 of 6" in report and "2 with the column read from the check" in report
    assert "unknown table 'shipments'" in report


def test_column_from_code_prefers_a_known_column():
    known = {"country", "year"}
    assert column_from_code('.hasCompleteness("cost_category", lambda x: x > 0)', known) is None
    assert column_from_code(".hasUniqueness(['country', 'year'], lambda x: x == 1.0)", known) == "country"
    assert column_from_code(".hasSize(lambda x: x > 0)", known) is None
