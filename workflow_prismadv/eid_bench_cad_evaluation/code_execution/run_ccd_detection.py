import pandas as pd

from prismadv.data_models.code_container import CodeContainer
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root


def run_ccd_detection(dataset_name, subtask_name, processed_data_label,
                      single_script=""):
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    script_results = {}  # script_name -> [correlated columns]

    for script_path in project_manager.get_available_script_path_list_for_subtask(subtask_name):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        clean = True
        raw_input_path = project_manager.get_new_data_path(subtask_name, processed_data_label, clean=clean)
        input_csv_path = raw_input_path / "new_data.csv"
        complete_df = pd.read_csv(input_csv_path)
        column_name_list = list(complete_df.columns)
        code: CodeContainer = FileLoader.load_py_file(script_path)
        code_without_assertions, assertions_sorted = code.extract_assertions()
        correlated_column_groups = correlated_column_detection(assertions_sorted, column_name_list)
        script_results[script_path.name] = correlated_column_groups
    return script_results


def correlated_column_detection(assertions_sorted, column_name_list):
    correlated_column_groups = []
    for assertion in assertions_sorted:
        correlated_columns_in_assertion = set()
        # if column_names coappear in assertion['code'], they are correlated
        assertion_code = assertion['code']
        for column_name in column_name_list:
            if column_name in assertion_code:
                correlated_columns_in_assertion.add(column_name)
        if len(correlated_columns_in_assertion) >= 2:
            # check if this group overlaps with existing groups
            merged = False
            for group in correlated_column_groups:
                if not correlated_columns_in_assertion.isdisjoint(group):
                    group.update(correlated_columns_in_assertion)
                    merged = True
                    break
            if not merged:
                correlated_column_groups.append(correlated_columns_in_assertion)
    # Convert sets to sorted lists for consistency
    correlated_column_groups = [sorted(list(group)) for group in correlated_column_groups]
    return correlated_column_groups
