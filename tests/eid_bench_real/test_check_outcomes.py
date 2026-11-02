import json
from pathlib import Path

import oyaml as yaml
import pandas as pd

from tests.eid_bench_real.test_project_manager import write_example
from workflow_prismadv.eid_bench_real_experiments import _0_error_injection, _2_check_outcomes, utils

SCRIPT = "patient_summary"


def write_config(example_root: Path, label: str, expected: str) -> None:
    group, number = label.split("-")
    errors_dir = example_root / "errors" / f"error-group-{group}"
    errors_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "table": "patients",
        "corruption": {"class": "ColumnDropping", "columns": ["name"]},
        "expected_current_outcome": expected,
    }
    (errors_dir / f"project_{number}.yaml").write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")


def write_run(tmp_path: Path, label: str, *, status: str, summary: dict | None) -> None:
    out = utils.execution_dir("example_a", SCRIPT, "corrupted", tmp_path, corruption_label=label, create=True)
    (out / "run_result.json").write_text(json.dumps({"status": status, "stderr": "boom\n"}), encoding="utf-8")
    if summary is not None:
        (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def clean_summary(example_root: Path, payload: dict) -> None:
    path = example_root / "expected" / SCRIPT / "clean_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def verdicts(tmp_path: Path) -> dict[str, str]:
    report = _2_check_outcomes.check_example(tmp_path, "example_a")
    return {row["label"]: row["verdict"] for row in report["labels"]}


def test_labels_are_compared_with_execution_and_output_drift(tmp_path):
    example_root = write_example(tmp_path)
    clean_summary(example_root, {"rows": 10, "elapsed_seconds": 1.5, "plots": ["a.png"]})
    write_config(example_root, "1-1", "safe")
    write_run(tmp_path, "1-1", status="passed", summary={"rows": 10, "elapsed_seconds": 9.0, "plots": []})
    write_config(example_root, "1-2", "unsafe")
    write_run(tmp_path, "1-2", status="failed", summary=None)
    write_config(example_root, "2-1", "safe")
    write_run(tmp_path, "2-1", status="failed", summary=None)
    write_config(example_root, "2-2", "unsafe")
    write_run(tmp_path, "2-2", status="passed", summary={"rows": 10, "elapsed_seconds": 2.0, "plots": ["a.png"]})
    write_config(example_root, "10-1", "safe")
    write_run(tmp_path, "10-1", status="passed", summary={"rows": 9, "elapsed_seconds": 1.5, "plots": ["a.png"]})
    write_config(example_root, "10-2", "unsafe")

    result = verdicts(tmp_path)

    assert result["1-1"] == "consistent"  # volatile keys ignored
    assert result["1-2"] == "consistent"
    assert result["2-1"] == "label_contradicted_by_execution"
    assert result["2-2"] == "unsafe_label_without_observable_effect"
    assert result["10-1"] == "safe_label_with_output_drift"
    assert result["10-2"] == "not_run"
    written = yaml.safe_load((utils.outcomes_dir("example_a", tmp_path) / "observed_outcomes.yaml").read_text())
    assert written["verdict_counts"]["consistent"] == 2


def test_changed_paths_names_the_differing_fields():
    expected = {"a": 1, "b": {"c": [1, 2], "d": "x"}}
    actual = {"a": 1, "b": {"c": [1, 3], "d": "x"}, "e": 0}
    assert _2_check_outcomes.changed_paths(expected, actual) == ["b.c[1]", "e"]


def test_numeric_tolerance_and_metric_paths_shape_the_drift(tmp_path):
    example_root = write_example(tmp_path)
    manifest_path = example_root / "manifest.yaml"
    manifest_path.write_text(
        manifest_path.read_text()
        + "outcome_check:\n  numeric_tolerance: 0.05\n  compare_output_files: false\n"
          "  scripts:\n    patient_summary:\n      metric_paths: [best_auc]\n",
        encoding="utf-8",
    )
    clean_summary(example_root, {"best_auc": 0.90, "recall": 0.50})
    write_config(example_root, "1-1", "safe")
    write_run(tmp_path, "1-1", status="passed", summary={"best_auc": 0.88, "recall": 0.10})
    write_config(example_root, "1-2", "unsafe")
    write_run(tmp_path, "1-2", status="passed", summary={"best_auc": 0.70, "recall": 0.50})

    result = verdicts(tmp_path)

    assert result["1-1"] == "consistent"  # 2% AUC move within tolerance; recall not a metric path
    assert result["1-2"] == "consistent"  # 22% AUC drop is drift


def test_lookup_follows_dotted_paths_with_indexes():
    document = {"all_models": [{"auc": 0.9}, {"auc": 0.8}], "best": {"auc": 0.9}}
    assert _2_check_outcomes.lookup(document, "all_models[1].auc") == 0.8
    assert _2_check_outcomes.lookup(document, "best.auc") == 0.9
    assert _2_check_outcomes.lookup(document, "best.missing") is None


def test_directory_assets_are_copied_into_the_bundle(tmp_path):
    clean = tmp_path / "clean"
    (clean / "vocab").mkdir(parents=True)
    (clean / "vocab" / "CONCEPT.csv.gz").write_bytes(b"x")
    (clean / "patients.csv").write_text("Id\n1\n")
    bundle = tmp_path / "bundle"
    (bundle / "input").mkdir(parents=True)
    (bundle / "tables").mkdir()

    copied = _0_error_injection.copy_directory_assets(clean, bundle)

    assert copied == ["vocab"]
    assert (bundle / "input" / "vocab" / "CONCEPT.csv.gz").exists()
    assert (bundle / "tables" / "vocab" / "CONCEPT.csv.gz").exists()


def test_unchanged_tables_are_detected():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", None]})
    assert _0_error_injection.tables_unchanged({"t": frame}, {"t": frame.copy()})
    assert not _0_error_injection.tables_unchanged({"t": frame}, {"t": frame.assign(a=[1, 3])})
