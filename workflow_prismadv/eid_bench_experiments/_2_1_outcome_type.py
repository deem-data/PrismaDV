import argparse
import difflib
import json
import re

from prismadv.data_models.code_container import CodeContainer
from prismadv.loader import FileLoader
from prismadv.project_manager import is_clean_label
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_experiments.code_execution.run_general_python_code_without_assertions import \
    get_without_assertions_output_path

FRAME = re.compile(r'File "([^"]+)", line (\d+)')
EXC = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Warning)\b(:|$)")
TEMP_SCRIPT = "script_to_run.py"  # PythonExecutor.run_script writes the code without assertions to this file


def parse_error(text, script_name):
    """Exception name, exception line and the line numbers of the task script in a traceback, outermost first."""
    heads = [line.strip() for line in text.splitlines() if EXC.match(line.strip())]
    exc_line = heads[-1] if heads else ("Timeout" if "Timed out" in text else "unknown error")
    exc = exc_line.split(":", 1)[0].rsplit(".", 1)[-1]
    lines = [int(m.group(2)) for m in FRAME.finditer(text) if m.group(1).endswith("/" + script_name)]
    return exc, exc_line[:300], lines


def assertion_blocks(code_lines):
    blocks, start = [], None
    for i, line in enumerate(code_lines, 1):
        if line.strip() == "# ASSERTION_START":
            start = i
        elif line.strip() == "# ASSERTION_END" and start is not None:
            blocks.append([start, i])
            start = None
    return blocks


def stripped_to_original_lines(code):
    """Line number in the code without assertions -> line number in the original script."""
    original = code.split("\n")
    stripped = CodeContainer(code).extract_assertions()[0].split("\n")
    mapping = {}
    for tag, i1, _, j1, j2 in difflib.SequenceMatcher(None, original, stripped, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                mapping[j1 + k + 1] = i1 + k + 1
    return mapping


def outcome_type(project_manager, subtask_name, label, script_path):
    script_name = script_path.stem
    corrupted_error = project_manager.get_execution_output_path(subtask_name, label, script_name, clean=False) / \
        "error.txt"
    if not corrupted_error.exists():
        return {"outcome": "safe"}
    code = FileLoader.load_py_file(script_path)
    blocks = assertion_blocks(code.split("\n"))
    exc, msg, lines = parse_error(corrupted_error.read_text(errors="replace"), script_path.name)
    # The innermost frame inside an assertion block; a block can fail in a helper function that it calls.
    line, block = (lines[-1] if lines else None), None
    for frame_line in reversed(lines):
        k = next((k for k, (a, b) in enumerate(blocks) if a <= frame_line <= b), None)
        if k is not None:
            line, block = frame_line, k
            break
    result = {"outcome": "silent", "exception": exc, "message": msg, "line": line, "block": block}

    without_dir = get_without_assertions_output_path(project_manager, subtask_name, label, script_name)
    if not without_dir.exists():
        raise FileNotFoundError(f"{without_dir} is missing: run _1_1_code_execution_without_assertions.py first")
    without_error = without_dir / "error.txt"
    if without_error.exists():
        # The run without assertions decides the crash: report its exception at its line in the original script.
        text = without_error.read_text(errors="replace")
        exc, msg, stripped_lines = parse_error(text, TEMP_SCRIPT)
        stripped_line = stripped_lines[-1] if stripped_lines else None
        mapping = stripped_to_original_lines(code)
        tail = "\n".join(text.strip().splitlines()[-6:])
        tail = re.sub(r'File "[^"]*/' + re.escape(TEMP_SCRIPT) + r'", line (\d+)',
                      lambda m: f'File "{script_path.name}", line {mapping.get(int(m.group(1)), "?")}', tail)
        tail = re.sub(r'"[^"]*/site-packages/', '"site-packages/', tail)
        result = {"outcome": "crash", "exception": exc, "message": msg,
                  "line": mapping.get(stripped_line) if stripped_line else None,
                  "block": None, "assertion_block_failed_first": block, "tail": tail}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify each corrupted run as safe, silent failure or crash")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name to process (e.g., 'imdb', 'students'). If not provided, processes all datasets."
    )
    args = parser.parse_args()

    all_datasets = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]
    dataset_name_options = [args.dataset_name] if args.dataset_name else all_datasets

    for dataset_name in dataset_name_options:
        project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        subtask_name = "general_task"
        if subtask_name not in project_manager.get_available_subtasks():
            continue
        script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name)
        counts = {"safe": 0, "silent": 0, "crash": 0}
        for processed_data_label in project_manager.get_available_processed_data_labels_for_subtask(subtask_name):
            if is_clean_label(processed_data_label):
                continue
            for script_path in script_path_list:
                result = outcome_type(project_manager, subtask_name, processed_data_label, script_path)
                counts[result["outcome"]] += 1
                output_file = project_manager.get_execution_output_validation_path(
                    subtask_name, processed_data_label, script_path.stem
                ) / "outcome_type.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with output_file.open("w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"{dataset_name}: {counts}")
