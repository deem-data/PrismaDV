import tempfile
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from prismadv.data_models.code_container import CodeContainer
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.runtime_environments import PythonExecutor
from prismadv.utils import get_project_root


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.reindex(sorted(df.columns), axis=1)
    if len(df.columns) > 0:
        df = df.sort_values(list(df.columns)).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        assert_frame_equal(normalize_df(a), normalize_df(b), check_like=True, check_dtype=False)
        return True
    except AssertionError:
        return False


def read_outputs(folder: Path):
    outputs = {}
    for file in folder.glob("*"):
        if file.suffix == ".csv":
            outputs[file.name] = pd.read_csv(file)
        elif file.suffix == ".txt":
            outputs[file.name] = file.read_text(encoding="utf-8")
    return outputs


def outputs_equal(baseline: dict, test: dict) -> dict:
    comparison = {}
    all_names = set(baseline.keys()) | set(test.keys())
    for name in all_names:
        if name not in baseline or name not in test:
            comparison[name] = False
            continue
        b, t = baseline[name], test[name]
        if isinstance(b, pd.DataFrame) and isinstance(t, pd.DataFrame):
            comparison[name] = frames_equal(b, t)
        elif isinstance(b, str) and isinstance(t, str):
            comparison[name] = (b.strip() == t.strip())
        else:
            comparison[name] = False
    return comparison


def run_general_python_code_for_accessed_column_detection(dataset_name, subtask_name, processed_data_label,
                                                          single_script=""):
    executor = PythonExecutor()
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    script_results = {}  # script_name -> [necessary columns]

    for script_path in project_manager.get_available_script_path_list_for_subtask(subtask_name):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        clean = True
        raw_input_path = project_manager.get_new_data_path(subtask_name, processed_data_label, clean=clean)
        input_csv_path = raw_input_path / "new_data.csv"
        complete_df = pd.read_csv(input_csv_path)
        column_name_list = list(complete_df.columns)
        code: CodeContainer = FileLoader.load_py_file(script_path)

        # ... helpers: normalize_df, frames_equal, read_outputs, outputs_equal ...

        with tempfile.TemporaryDirectory() as baseline_dir:
            baseline_dir = Path(baseline_dir)
            _ = executor.run_script(
                project_name=dataset_name,
                input_path=raw_input_path,
                script_context=code,
                output_path=baseline_dir,
            )
            baseline_outputs = read_outputs(baseline_dir)

        results = {}
        for column_name in column_name_list:
            one_leave_out_df = complete_df.drop(columns=[column_name])
            with tempfile.TemporaryDirectory() as loo_input_dir, tempfile.TemporaryDirectory() as loo_output_dir:
                loo_input_dir = Path(loo_input_dir)
                loo_output_dir = Path(loo_output_dir)

                temp_csv_path = loo_input_dir / "new_data.csv"
                one_leave_out_df.to_csv(temp_csv_path, index=False)

                _ = executor.run_script(
                    project_name=dataset_name,
                    input_path=loo_input_dir,
                    script_context=code,
                    output_path=loo_output_dir,
                )

                loo_outputs = read_outputs(loo_output_dir)
                comparison = outputs_equal(baseline_outputs, loo_outputs)

                results[column_name] = {
                    "necessary": not all(comparison.values()),
                    "file_comparison": comparison,
                }

        # collect necessary columns for this script
        script_results[script_path.stem] = [c for c, info in results.items() if info["necessary"]]
        print(f"script results: {script_results}")
    return script_results
