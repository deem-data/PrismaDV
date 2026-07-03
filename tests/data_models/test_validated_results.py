import pytest

from prismadv.data_models import ValidationResults, ValidationCodeEntry, ColumnValidationResults


@pytest.fixture
def validation_results_instance():
    code_entries = [
        ValidationCodeEntry(suggestion=".hasCompleteness('Age', lambda x: x > 0.9)", status="Passed"),
        ValidationCodeEntry(suggestion=".hasUniqueness('Age')", status="Failed"),
    ]
    column_validation_results = ColumnValidationResults(code=code_entries)
    validation_results = ValidationResults(results={"column1": column_validation_results})
    return validation_results


def test_io_yaml(validation_results_instance, tmp_path):
    output_path = tmp_path / "validation_results.yaml"
    validation_results_instance.save_to_yaml(str(output_path))

    loaded_instance = ValidationResults.from_yaml(str(output_path))
    assert loaded_instance.to_dict() == validation_results_instance.to_dict()
