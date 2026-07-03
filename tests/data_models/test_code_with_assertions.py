from prismadv.data_models.code_container import CodeContainer
from prismadv.loader import FileLoader


def test_code_container(resources_path):
    code_with_assertions = CodeContainer(
        FileLoader.load_py_file(resources_path / "code_with_assertions" / "with_assertions.py"))
    code_without_assertions = CodeContainer(
        FileLoader.load_py_file(resources_path / "code_with_assertions" / "without_assertions.py"))
    assertion_removed, assertions = code_with_assertions.extract_assertions()
    assertion_removed_without_blank_lines = str(assertion_removed.without_blank_lines())
    code_without_assertions_without_blank_lines = str(code_without_assertions.without_blank_lines())
    assert assertion_removed_without_blank_lines == code_without_assertions_without_blank_lines
    assertion_back = assertion_removed.insert_assertions(assertions)
    assertion_back_without_blank_lines = str(assertion_back.without_blank_lines())
    code_with_assertions_without_blank_lines = str(code_with_assertions.without_blank_lines())
    assert assertion_back_without_blank_lines == code_with_assertions_without_blank_lines
