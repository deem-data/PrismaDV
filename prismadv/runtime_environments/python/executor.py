import platform
import subprocess
import tempfile
import venv
from pathlib import Path
from typing import Union

from prismadv.runtime_environments.basis import ExecutorBase
from prismadv.utils import get_current_folder


class PythonExecutor(ExecutorBase):
    env_path = get_current_folder() / "env"
    requirements_path = get_current_folder() / "requirements.txt"

    def __init__(self):
        super().__init__()
        self.env_path.mkdir(exist_ok=True)
        self.python_executable = self._get_python_executable()

    def _run_with_handling(self, input_path, script_path, output_path: Union[Path, None], start_msg: str, timeout: int):
        tmp_ctx = None
        if output_path is None:
            tmp_ctx = tempfile.TemporaryDirectory()
            output_path = Path(tmp_ctx.name)
        output_path.mkdir(parents=True, exist_ok=True)
        print(start_msg)
        command = [
            str(self.python_executable), str(script_path),
            "--input", str(input_path), "--output", str(output_path)
        ]
        try:
            result = subprocess.run(command, check=True, timeout=timeout, capture_output=True, text=True)
            print(f"Finished successfully\noutput files: {list(output_path.iterdir())}")
            return f"Success: {result}"
        except subprocess.CalledProcessError as e:
            msg = e.stderr or e.stdout or "subprocess failed without stderr/stdout"
            print(f"Error: {msg}")
            try:
                (output_path / "error.txt").write_text(msg)
            except Exception:
                pass
            return f"Error: {msg}"
        except subprocess.TimeoutExpired:
            msg = f"Timed out after {timeout} seconds."
            print(f"Error: {msg}")
            try:
                (output_path / "error.txt").write_text(msg)
            except Exception:
                pass
            return f"Error: {msg}"
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    def run(self, project_name: str, input_path: Path, script_path: Path, output_path: Path, timeout: int = 120):
        start_msg = f"Running {script_path}\n  input={input_path}\n  output={output_path}"

        return self._run_with_handling(input_path, script_path, output_path, start_msg, timeout)

    def run_script(self, project_name: str, input_path: Path, script_context: str, output_path: Union[Path, None],
                   timeout: int = 120):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "script_to_run.py"
            script_path.write_text(script_context, encoding="utf-8")
            start_msg = (
                f"Running script from {script_path}\n"
                f"  input={input_path}\n"
                f"  output={output_path}"
            )
            return self._run_with_handling(input_path, script_path, output_path, start_msg, timeout)

    def _get_python_executable(self):
        self._create_or_update_environment()
        if platform.system() == "Windows":
            python_executable = self.env_path / "Scripts" / "python.exe"
        else:
            python_executable = self.env_path / "bin" / "python"
        return python_executable

    def _create_or_update_environment(self):
        pyvenv_cfg = self.env_path / "pyvenv.cfg"
        if not pyvenv_cfg.exists():
            self._create_environment()
        if not self._check_env_against_requirements():
            self._update_environment()

    def _create_environment(self):
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(self.env_path)
        pip_path = self._get_pip_path()
        print(f"Installing requirements from {self.requirements_path} into {self.env_path}")
        subprocess.check_call([str(pip_path), "install", "--upgrade", "pip"])
        subprocess.check_call([str(pip_path), "install", "-r", str(self.requirements_path)])

    def _update_environment(self):
        pip_path = self._get_pip_path()
        print(f"Updating requirements from {self.requirements_path} into {self.env_path}")
        subprocess.check_call([str(pip_path), "install", "--upgrade", "pip"])
        subprocess.check_call([str(pip_path), "install", "-r", str(self.requirements_path)])

    def _check_env_against_requirements(self):
        reqs = self.requirements_path.read_text().split("\n")
        reqs = [req.strip().split("==") for req in reqs if req.strip()]
        reqs = [tuple(req) if len(req) == 2 else req[0] for req in reqs]
        if not reqs:
            return True
        pip_path = self._get_pip_path()
        installed = subprocess.check_output([str(pip_path), "freeze"]).decode("utf-8").split("\n")
        installed = [req.split("==") for req in installed]
        installed_with_version = [tuple(i) if len(i) == 2 else i[0] for i in installed]
        installed_wo_version = [i[0] for i in installed]
        return all((req in installed_with_version or req in installed_wo_version) for req in reqs)

    def _get_pip_path(self):
        if platform.system() == "Windows":
            pip_path = self.env_path / "Scripts" / "pip.exe"
        else:
            pip_path = self.env_path / "bin" / "pip"
        return pip_path
