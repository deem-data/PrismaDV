import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Query

from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

project_manager = ProjectManager(project_root=get_project_root(), dataset_name="healthcare_dataset")
script_info = project_manager.get_available_script_info()

example_script = script_info["ml_inference"]["classification"][0]
source_code = FileLoader.load_py_file(example_script)
tree = parser.parse(
    bytes(
        source_code,
        "utf8"
    )
)

query_text = """
(subscript
  (_) @df_base
  (string (string_content) @col
    (#eq? @col "Age")))
"""

query = Query(PY_LANGUAGE, query_text)

captures_dict = query.captures(tree.root_node)

print("Captured nodes:")

for name, captures in captures_dict.items():
    for capture in captures:
        start_line = capture.start_point[0]
        end_line = capture.end_point[0]
        print(f"{source_code.splitlines()[start_line:end_line + 1]}")
