from prismadv.data_models.constraints_v2 import AssumptionEntry, CodeEntry, SourceLocation
from prismadv.llm.langchain.models.prismadv_multi_table.orchestrator import MultiTablePrismaLangChainDV
from prismadv.llm.langchain.models.prismadv_multi_table.types import (
    ColumnGroupRef,
    ColumnRef,
    MultiFileCodeContext,
)


def test_constraints_from_table_outputs_preserves_table_and_column_group():
    column_ref = ColumnRef("patients", "Id")
    assumption = AssumptionEntry(
        text="patients.Id should be complete.",
        sources=[SourceLocation(file="adapter/case.py", start_line=10, end_line=12)],
    )
    code = CodeEntry(suggestion='.isComplete("Id")', validity=True, level="error")

    constraints = MultiTablePrismaLangChainDV.constraints_from_table_outputs(
        [column_ref],
        {column_ref.key: [assumption]},
        {column_ref.key: [code]},
    )

    raw = constraints.to_dict()["constraints"]
    assert list(raw) == ["patients.Id"]
    assert raw["patients.Id"]["table_name"] == "patients"
    assert raw["patients.Id"]["column_group"] == "Id"
    assert raw["patients.Id"]["assumptions"][0]["sources"] == [
        {"file": "adapter/case.py", "start_line": 10, "end_line": 12}
    ]


def test_constraints_from_table_outputs_supports_same_table_groups():
    single_ref = ColumnRef("encounters", "START")
    group_ref = ColumnGroupRef(
        table="encounters",
        columns=frozenset({"START", "STOP"}),
        correlation_type="Order dependency",
    )
    single_assumption = AssumptionEntry(text="START not null", sources=[])
    group_assumption = AssumptionEntry(text="START <= STOP", sources=[])
    single_code = CodeEntry(suggestion='.isComplete("START")', validity=True, level="error")
    group_code = CodeEntry(
        suggestion='.satisfies("`START` <= `STOP`", "start_le_stop", lambda x: x == 1)',
        validity=True,
        level="warning",
    )

    constraints = MultiTablePrismaLangChainDV.constraints_from_table_outputs(
        [single_ref],
        {single_ref.key: [single_assumption]},
        {single_ref.key: [single_code]},
        column_groups=[group_ref],
        group_assumptions={group_ref.key: [group_assumption]},
        group_codes={group_ref.key: [group_code]},
    )

    raw = constraints.to_dict()["constraints"]
    assert raw["encounters.START"]["table_name"] == "encounters"
    assert raw["encounters.START"]["column_group"] == "START"
    group_key = '["encounters.START","encounters.STOP"]'
    assert group_key in raw
    assert raw[group_key]["table_name"] == "encounters"
    assert raw[group_key]["column_group"] == '["START","STOP"]'
    assert raw[group_key]["code"][0]["suggestion"].startswith(".satisfies")

    grouped = constraints.group_by_table()
    encounters_constraints = grouped["encounters"]
    assert any(
        key == frozenset({"START", "STOP"}) for key in encounters_constraints.data_map
    )
    assert "START" in encounters_constraints.data_map


def test_column_groups_from_response_filters_cross_table_and_singletons():
    table_profiles = {
        "encounters": {
            "table_name": "encounters",
            "columns": [{"name": "START"}, {"name": "STOP"}, {"name": "PATIENT"}],
        },
        "patients": {
            "table_name": "patients",
            "columns": [{"name": "Id"}, {"name": "BIRTHDATE"}],
        },
    }
    column_refs = [
        ColumnRef("encounters", "START"),
        ColumnRef("encounters", "STOP"),
        ColumnRef("patients", "Id"),
    ]
    response = {
        "groups": [
            {"table": "encounters", "correlated_columns": ["START", "STOP"], "correlation_type": "Order"},
            {"table": "encounters", "correlated_columns": ["START"]},
            {"table": "encounters", "correlated_columns": ["START", "DOES_NOT_EXIST"]},
            {"table": "patients", "correlated_columns": ["Id", "BIRTHDATE"]},
            {"table": "unknown_table", "correlated_columns": ["a", "b"]},
            {"table": "encounters", "correlated_columns": ["START", "STOP"]},
        ]
    }

    groups = MultiTablePrismaLangChainDV._column_groups_from_response(response, column_refs, table_profiles)
    assert len(groups) == 1
    only_group = groups[0]
    assert only_group.table == "encounters"
    assert only_group.columns == frozenset({"START", "STOP"})
    assert only_group.correlation_type == "Order"


class _RecordingChain:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def invoke(self, input_vars):
        self.calls.append(input_vars)
        return self._response


def _make_orchestrator(monkeypatch, chain_responses):
    from prismadv.llm.langchain.models.prismadv_multi_table import orchestrator as orch
    from prismadv.llm.langchain.models.prismadv_multi_table.tasks import MultiTablePrismaDVTasks

    monkeypatch.setattr(orch, "get_langchain_model", lambda model_name, temperature=0.6: object())
    monkeypatch.setattr(orch, "build_chains", lambda model: {})

    instance = orch.MultiTablePrismaLangChainDV(
        model_name="dummy-model",
        downstream_task_description="task",
        use_column_correlation_detection=False,
        with_assumptions=False,
    )
    chains = {
        task: _RecordingChain(chain_responses.get(task, {})) for task in MultiTablePrismaDVTasks
    }
    instance._chains = chains
    instance.runtime = orch.Runtime(chains)
    return instance, chains


def test_invoke_without_assumptions_uses_direct_code_task(monkeypatch, tmp_path):
    from prismadv.llm.langchain.models.prismadv_multi_table.tasks import MultiTablePrismaDVTasks

    table_profiles = {
        "patients": {
            "table_name": "patients",
            "columns": [{"name": "Id"}],
        }
    }
    chain_responses = {
        MultiTablePrismaDVTasks.TABLE_COLUMN_ACCESS_DETECTION: {
            "columns": [{"table": "patients", "column": "Id"}]
        },
        MultiTablePrismaDVTasks.TABLE_COLUMN_DATA_FLOW_INSPECTION: {"sources": []},
        MultiTablePrismaDVTasks.TABLE_COLUMN_DIRECT_CODE_GENERATION: {
            "constraint_code": [
                {"suggestion": '.isComplete("Id")', "level": "error"}
            ]
        },
    }
    instance, chains = _make_orchestrator(monkeypatch, chain_responses)

    input_variables = {
        "code_context": {
            "files": [{"path": "adapter/case.py", "content": "x = 1\n"}],
        },
        "context": {"script": {"reads": ["patients"]}},
        "table_profiles": table_profiles,
        "cfg_use_dataflow": True,
    }
    constraints, locations, cost_summary = instance.invoke(input_variables)

    assert chains[MultiTablePrismaDVTasks.TABLE_COLUMN_ASSUMPTION_GENERATION].calls == []
    assert chains[MultiTablePrismaDVTasks.TABLE_COLUMN_CODE_GENERATION].calls == []
    assert len(chains[MultiTablePrismaDVTasks.TABLE_COLUMN_DIRECT_CODE_GENERATION].calls) == 1
    raw = constraints.to_dict()["constraints"]
    assert "patients.Id" in raw
    assert raw["patients.Id"]["assumptions"] == []
    assert raw["patients.Id"]["code"][0]["suggestion"] == '.isComplete("Id")'


def test_multi_file_code_context_renders_independent_file_line_numbers():
    context = MultiFileCodeContext(
        [
            {"path": "adapter/run_script.py", "content": "import case\ncase.run()\n"},
            {"path": "adapter/case.py", "content": "def run():\n    return 1\n"},
        ]
    )

    rendered = context.add_highlighted_line_numbers(
        [SourceLocation(file="adapter/case.py", start_line=2, end_line=2)]
    )

    assert "# File: adapter/run_script.py" in rendered
    assert "# File: adapter/case.py" in rendered
    assert "      0001: import case" in rendered
    assert "-**-> 0002:     return 1" in rendered
