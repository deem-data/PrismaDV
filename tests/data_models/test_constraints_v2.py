import oyaml as yaml

from prismadv.data_models import AssumptionEntry, CodeEntryV2, ConstraintsWithSources, ColumnConstraintsWithSources


def test_from_yaml(constraints_with_sources_instance, tmp_path):
    constraints_with_sources_instance.save_to_yaml(str(tmp_path / "constraints.yaml"))
    with open(tmp_path / "constraints.yaml", "r") as f:
        raw_constraint_dict = yaml.load(
            f, Loader=yaml.FullLoader
        )
    constraints = ConstraintsWithSources.from_dict(
        raw_constraint_dict
    )

    assert constraints.to_dict() == constraints_with_sources_instance.to_dict()


def test_from_dict():
    data = {
        "constraints": {
            "column1": {
                "code": [{"suggestion": "Code 1", "validity": True, "level": "warning"},
                         {"suggestion": "Code 2", "validity": False, "level": "error"}],
                "assumptions": [
                    {"text": "Assumption 1", "sources": [{"start_line": 1, "end_line": 2}]}
                ]
            },
            "column2": {
                "code": [{"suggestion": "Use a unique constraint", "validity": True, "level": "info"}],
                "assumptions": [
                    {"text": "Assumption 2", "sources": [{"file": "file2.py", "start_line": 3, "end_line": 4}]}
                ]
            }
        }
    }

    constraints = ConstraintsWithSources.from_dict(data)
    assert len(constraints.data_map) == 2
    assert constraints.data_map["column1"].code[0].suggestion == "Code 1"
    assert constraints.data_map["column1"].assumptions[0].text == "Assumption 1"
    assert constraints.data_map["column1"].assumptions[0].sources[0].file == ""
    assert constraints.data_map["column2"].code[0].suggestion == "Use a unique constraint"
    assert constraints.data_map["column2"].assumptions[0].text == "Assumption 2"
    assert constraints.data_map["column2"].assumptions[0].sources[0].file == "file2.py"


def test_save_to_yaml(constraints_with_sources_instance, tmp_path):
    output_path = tmp_path / "constraints.yaml"
    constraints_with_sources_instance.save_to_yaml(str(output_path))

    with open(output_path, "r") as f:
        content = f.read()

    assert "constraints:" in content
    assert "column1:" in content
    assert "Code 1" in content
    assert "Assumption 1" in content
    assert "file1.py" in content


def test_table_qualified_constraints_with_sources_round_trip():
    data = {
        "constraints": {
            "patients.id": {
                "table_name": "patients",
                "column_group": "id",
                "code": [{"suggestion": ".isComplete('id')", "validity": True, "level": "error"}],
                "assumptions": [{"text": "Patient id is required", "sources": []}]
            },
            "encounters.id": {
                "table_name": "encounters",
                "column_group": "id",
                "code": [{"suggestion": ".isUnique('id')", "validity": False, "level": "warning"}],
                "assumptions": [{"text": "Encounter id should be unique", "sources": []}]
            }
        }
    }

    constraints = ConstraintsWithSources.from_dict(data)
    serialized = constraints.to_dict()

    assert serialized["constraints"]["patients.id"]["table_name"] == "patients"
    assert serialized["constraints"]["patients.id"]["column_group"] == "id"
    assert constraints.data_map["patients.id"].table_name == "patients"
    assert constraints.data_map["patients.id"].column_group == "id"
    code_map = constraints.get_suggestions_code_column_map()
    assert code_map[".isComplete('id')"]["table"] == "patients"
    assert code_map[".isComplete('id')"]["column"] == "id"
    assert code_map[".isUnique('id')"]["table"] == "encounters"


def test_table_qualified_constraints_with_sources_group_by_table():
    constraints = ConstraintsWithSources(data_map={
        "patients.id": ColumnConstraintsWithSources(
            assumptions=[AssumptionEntry(text="Patient id is required")],
            code=[CodeEntryV2(suggestion=".isComplete('id')", validity=True, level="error")],
            table_name="patients",
            column_group="id",
        ),
        "encounters.id": ColumnConstraintsWithSources(
            assumptions=[AssumptionEntry(text="Encounter id should be unique")],
            code=[CodeEntryV2(suggestion=".isUnique('id')", validity=False, level="warning")],
            table_name="encounters",
            column_group="id",
        ),
    })

    grouped = constraints.group_by_table()

    assert sorted(grouped.keys()) == ["encounters", "patients"]
    assert list(grouped["patients"].data_map.keys()) == ["id"]
    assert grouped["patients"].data_map["id"].table_name == "patients"
    assert grouped["patients"].get_suggestions_code_column_map()[".isComplete('id')"]["column"] == "id"
