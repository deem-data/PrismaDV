"""
Some useful utils for the project
"""
import inspect
import logging
import os
from pathlib import Path

import dotenv
import oyaml as yaml
from attr import dataclass


def suppress_py4j_logging():
    """Suppress py4j error logs that clutter the output."""
    py4j_logger = logging.getLogger("py4j")
    py4j_logger.setLevel(logging.CRITICAL)
    py4j_logger.propagate = False
    py4j_logger.handlers = []

    py4j_cs_logger = logging.getLogger("py4j.clientserver")
    py4j_cs_logger.setLevel(logging.CRITICAL)
    py4j_cs_logger.propagate = False
    py4j_cs_logger.handlers = []


class SuppressIvyOutput:
    """Context manager to suppress Ivy dependency resolution output during Spark initialization."""

    def __init__(self):
        self.null_fd = None
        self.save_stdout = None
        self.save_stderr = None

    def __enter__(self):
        # Open devnull
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        # Save current stdout/stderr
        self.save_stdout = os.dup(1)
        self.save_stderr = os.dup(2)
        # Redirect stdout/stderr to devnull
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)
        return self

    def __exit__(self, *_):
        # Restore stdout/stderr
        os.dup2(self.save_stdout, 1)
        os.dup2(self.save_stderr, 2)
        # Close saved file descriptors
        os.close(self.save_stdout)
        os.close(self.save_stderr)
        os.close(self.null_fd)


class FilteredStream:
    """Filter stdout/stderr to suppress Spark/Ivy callback server messages."""

    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = ""
        self.filtered_patterns = [
            "Python Callback server",
            "PythonCallback server",
            "Setting default log level",
            "To adjust logging level",
            "WARN NativeCodeLoader",
            "WARN SparkStringUtils",
            "WARN DAGScheduler",
            "WARN Utils",
            # "INFO dspy.teleprompt.gepa.gepa",  # Commented out to see SIFTA optimization logs
            "DeprecationWarning",
            "distutils Version classes",
            "is_datetime64tz_dtype",
            "LooseVersion",
            "pyspark/sql/pandas",
            ":: loading settings ::",
            "Ivy Default Cache",
            "The jars for the packages",
            "added as a dependency",
            ":: resolving dependencies ::",
            "confs: [default]",
            "found ",
            ":: resolution report ::",
            ":: modules in use:",
            ":: evicted modules:",
            "---------------------------------------------------------------------",
            "|                  |",
            ":: retrieving ::",
            "artifacts copied",
            "already retrieved",
            "using builtin-java classes",
            "resolves to a loopback address",
            "There was an exception while executing the Python Proxy on the Python Side",
        ]

    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            self.buffer = lines[-1]
            for line in lines[:-1]:
                if line.strip() and not any(pattern in line for pattern in self.filtered_patterns):
                    self.original_stream.write(line + '\n')
            return len(text)
        return len(text)

    def flush(self):
        if self.buffer and not any(pattern in self.buffer for pattern in self.filtered_patterns):
            self.original_stream.write(self.buffer)
            self.buffer = ""
        return self.original_stream.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent.parent


def get_current_folder() -> Path:
    """
    Returns the directory where the calling script is stored.
    """
    # Get the file path of the script that called this function
    caller_file = inspect.stack()[1].filename
    # Get the directory of the caller script
    return Path(caller_file).parent


def load_dotenv():
    """Load the .env file."""
    dotenv.load_dotenv(get_project_root() / ".env")


@dataclass
class TaskInstance:
    script_name: str
    original_script: str
    script_path: Path
    annotations: dict


def get_task_instance(script_path):
    config_file_path = script_path.parent.parent.parent / "annotations" / script_path.parent.stem / f"{script_path.stem}.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    task_instance = TaskInstance(
        script_name=script_path.stem,
        original_script=script_path.read_text(), script_path=script_path,
        annotations=config["annotations"])
    return task_instance
