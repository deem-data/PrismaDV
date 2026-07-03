import json
from pathlib import Path

from tests.eid_bench_real.test_project_manager import write_example
from workflow_prismadv.eid_bench_real_experiments import _1_run_eid_bench_real, utils


def write_adapter_entrypoint(example_root: Path, *, exit_code: int = 0) -> None:
    (example_root / "adapter" / "run_script.py").write_text(
        f"""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--script-id", required=True)
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
print(f"running {{args.script_id}}")
if {exit_code}:
    print("adapter failed", file=sys.stderr)
    raise SystemExit({exit_code})

(output / "summary.json").write_text(json.dumps({{
    "script_id": args.script_id,
    "input": args.input,
}}) + "\\n")
""".lstrip(),
        encoding="utf-8",
    )


def test_run_eid_bench_real_runs_all_scripts_and_writes_result(tmp_path):
    example_root = write_example(tmp_path)
    write_adapter_entrypoint(example_root)

    results = _1_run_eid_bench_real.run_eid_bench_real(
        project_root=tmp_path,
        example_id="example_a",
        script_id="all",
        variant="clean",
        overwrite=True,
    )

    assert len(results) == 1
    result = results[0]
    assert result["status"] == "passed"
    assert result["script_id"] == "patient_summary"
    assert result["variant"] == "clean"

    output_dir = utils.execution_dir("example_a", "patient_summary", "clean", tmp_path)
    raw_result = json.loads((output_dir / "run_result.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert raw_result["status"] == "passed"
    assert raw_result["summary_path"] == str(output_dir / "summary.json")
    assert summary["script_id"] == "patient_summary"


def test_run_eid_bench_real_records_adapter_failures(tmp_path):
    example_root = write_example(tmp_path)
    write_adapter_entrypoint(example_root, exit_code=3)

    results = _1_run_eid_bench_real.run_eid_bench_real(
        project_root=tmp_path,
        example_id="example_a",
        script_id="patient_summary",
        variant="corrupted",
        labels=["missing_key"],
        overwrite=True,
    )

    assert len(results) == 1
    result = results[0]
    assert result["status"] == "failed"
    assert result["returncode"] == 3
    assert result["corruption_label"] == "missing_key"
    assert "adapter failed" in result["stderr"]

    output_dir = utils.execution_dir(
        "example_a",
        "patient_summary",
        "corrupted",
        tmp_path,
        corruption_label="missing_key",
    )
    raw_result = json.loads((output_dir / "run_result.json").read_text(encoding="utf-8"))
    assert raw_result["status"] == "failed"
    assert raw_result["summary_path"] is None


def test_output_text_decodes_timeout_bytes():
    assert _1_run_eid_bench_real.output_text(b"adapter output") == "adapter output"
    assert _1_run_eid_bench_real.output_text(None) == ""
