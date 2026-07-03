from prismadv.code_inspector.llm_code_inspector.multiple.model import \
    ColumnDataFlowInspector as MultipleColumnDataFlowInspector
from prismadv.data_models import SourceLocation
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

project_manager = ProjectManager(project_root=get_project_root(), dataset_name="healthcare_dataset")
script_info = project_manager.get_available_script_info()

example_script = script_info["ml_inference"]["classification"][0]
source_code = FileLoader.load_py_file(example_script)

target_columns = ["Age", "Admission Type"]
sink_variable = "preds"

multiple_inspector = MultipleColumnDataFlowInspector(model_name="gpt-4.1-mini")

input_variables = {
    "code_script": source_code,
    "target_columns": target_columns,
    "sink_variable": sink_variable
}

res = multiple_inspector.invoke(input_variables=input_variables)
source_locations = [
    SourceLocation(start_line=source["start_line"], end_line=source["end_line"], file=example_script)
    for source in res["sources"]
]
for source in source_locations:
    print(f"Source from line {source.start_line} to {source.end_line}")

focused_code = source_code.focused_code(source_locations)
print("\nFocused Code Snippet:")
print(focused_code)
