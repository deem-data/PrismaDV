import os
from pathlib import Path
from typing import List, Union


def load_py_file(file_path: str, return_contents: bool = True) -> Union[str, Path]:
    """Load a Python file.

    Args:
        file_path: Path to the Python file
        return_contents: If True, returns file contents as string. If False, returns Path object.

    Returns:
        File contents as string if return_contents=True, otherwise Path object
    """
    if return_contents:
        with open(file_path, "r") as file:
            return file.read()
    else:
        return Path(file_path)


def load_py_files(dir_path: str, return_contents: bool = True) -> List[str]:
    """Load a list of Python files.

    Args:
        dir_path: Directory containing Python files
        return_contents: If True, returns file contents as strings. If False, returns Path objects.

    Returns:
        List of file contents (strings) if return_contents=True, otherwise list of Path objects
    """
    file_path = [f"{dir_path}/{file}" for file in os.listdir(dir_path)]
    return [load_py_file(file, return_contents) for file in file_path]
