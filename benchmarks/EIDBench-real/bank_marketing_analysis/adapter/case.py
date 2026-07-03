"""Adapter implementation for the Bank Marketing Analysis EIDBench-real case."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ADAPTER_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = ADAPTER_DIR.parent
SOURCE_REPO = ADAPTER_DIR / "source_repo"
ALL_SCRIPTS = ("preprocess", "eda", "fit_classifier", "feature_importance")


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def find_bank_csv(input_dir: Path) -> Path:
    path = input_dir / "bank_marketing.csv"
    if not path.exists():
        raise FileNotFoundError(f"bank_marketing.csv not found under {input_dir}")
    return path


def run_python(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=SOURCE_REPO, check=True)


def value_counts_csv(path: Path, column: str) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        counts: dict[str, int] = {}
        for row in reader:
            value = row[column]
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_preprocess(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    raw_csv = find_bank_csv(input_dir)
    data_dir = output_dir / "data" / "processed"
    model_dir = output_dir / "results" / "models"
    ensure_dirs(data_dir, model_dir)

    run_python([
        "scripts/split_and_process.py",
        "--raw_data",
        str(raw_csv),
        "--save_to",
        str(data_dir),
        "--preprocessor_to",
        str(model_dir),
        "--seed",
        "522",
    ])

    x_train = data_dir / "X_train.csv"
    x_train_trans = data_dir / "X_train_trans.csv"
    y_train = data_dir / "y_train.csv"
    y_test = data_dir / "y_test.csv"
    y_train_resmp = data_dir / "y_train_resmp.csv"
    bank_train = data_dir / "bank_train.csv"
    bank_test = data_dir / "bank_test.csv"
    preprocessor = model_dir / "bank_preprocessor.pickle"

    x_train_header = csv_header(x_train)
    x_train_trans_header = csv_header(x_train_trans)

    summary = {
        "script_id": "preprocess",
        "raw_rows": csv_row_count(raw_csv),
        "raw_columns": csv_header(raw_csv),
        "train_rows": csv_row_count(bank_train),
        "test_rows": csv_row_count(bank_test),
        "x_train_columns": x_train_header,
        "x_train_trans_columns": x_train_trans_header,
        "x_train_column_count": len(x_train_header),
        "x_train_trans_column_count": len(x_train_trans_header),
        "target_counts_train": value_counts_csv(y_train, "y"),
        "target_counts_test": value_counts_csv(y_test, "y"),
        "target_counts_resampled": value_counts_csv(y_train_resmp, "y"),
        "age_in_x_train": "age" in x_train_header,
        "age_in_x_train_trans": "age" in x_train_trans_header,
        "has_unnamed_index_column": any(
            col.startswith("Unnamed") for col in x_train_header + x_train_trans_header
        ),
        "preprocessor_exists": preprocessor.exists(),
    }
    return summary


def run_eda(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    preprocess_summary = run_preprocess(input_dir, output_dir)
    data_dir = output_dir / "data" / "processed"
    figure_dir = output_dir / "results" / "figures"
    ensure_dirs(figure_dir)

    run_python([
        "scripts/eda.py",
        "--training_data",
        str(data_dir / "bank_train.csv"),
        "--save_plot_to",
        str(figure_dir),
    ])

    plots = [
        "eda_categorical_variables.png",
        "eda_continuous_variables.png",
        "eda_log_variables.png",
    ]
    return {
        "script_id": "eda",
        "preprocess": preprocess_summary,
        "plots": {
            name: {
                "exists": (figure_dir / name).exists(),
                "size_bytes": (figure_dir / name).stat().st_size if (figure_dir / name).exists() else 0,
            }
            for name in plots
        },
    }


def run_fit_classifier(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    preprocess_summary = run_preprocess(input_dir, output_dir)
    data_dir = output_dir / "data" / "processed"
    model_dir = output_dir / "results" / "models"
    figure_dir = output_dir / "results" / "figures"
    ensure_dirs(model_dir, figure_dir)

    run_python([
        "scripts/fit_bank_classifier.py",
        "--resampled_training_data",
        str(data_dir / "X_train_resmp.csv"),
        "--resampled_training_response",
        str(data_dir / "y_train_resmp.csv"),
        "--test_data",
        str(data_dir / "X_test.csv"),
        "--test_response",
        str(data_dir / "y_test.csv"),
        "--preprocessor_pipe",
        str(model_dir / "bank_preprocessor.pickle"),
        "--save_pipelines_to",
        str(model_dir),
        "--save_plot_to",
        str(figure_dir),
        "--seed",
        "522",
    ])

    all_models = figure_dir / "all_models.csv"
    model_metrics = pd.read_csv(all_models).to_dict(orient="records")
    return {
        "script_id": "fit_classifier",
        "preprocess": preprocess_summary,
        "model_files": sorted(path.name for path in model_dir.glob("*.pickle")),
        "figure_files": sorted(path.name for path in figure_dir.iterdir() if path.is_file()),
        "all_models_rows": len(model_metrics),
        "all_models": model_metrics,
        "best_model": model_metrics[0]["Model"] if model_metrics else None,
        "best_auc": float(model_metrics[0]["Area_under_curve"]) if model_metrics else None,
    }


def run_feature_importance(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    fit_summary = run_fit_classifier(input_dir, output_dir)
    data_dir = output_dir / "data" / "processed"
    model_dir = output_dir / "results" / "models"
    figure_dir = output_dir / "results" / "figures"

    run_python([
        "scripts/feat_imp.py",
        "--transformed_training_data",
        str(data_dir / "X_train_trans.csv"),
        "--pipeline_model",
        str(model_dir / "logistic_pipeline.pickle"),
        "--save_plot_to",
        str(figure_dir),
        "--seed",
        "522",
    ])

    feature_columns = csv_header(data_dir / "X_train_trans.csv")
    feature_plot = figure_dir / "feat_imp.png"
    return {
        "script_id": "feature_importance",
        "fit_classifier": fit_summary,
        "feature_count": len(feature_columns),
        "age_in_features": "age" in feature_columns,
        "feature_importance_plot_exists": feature_plot.exists(),
        "feature_importance_plot_size_bytes": feature_plot.stat().st_size if feature_plot.exists() else 0,
    }


def run_script(script_id: str, input_dir: Path, output_dir: Path) -> None:
    if script_id not in ALL_SCRIPTS:
        raise ValueError(f"unknown script_id: {script_id}")

    clean_output_dir(output_dir)

    if script_id == "preprocess":
        summary = run_preprocess(input_dir, output_dir)
    elif script_id == "eda":
        summary = run_eda(input_dir, output_dir)
    elif script_id == "fit_classifier":
        summary = run_fit_classifier(input_dir, output_dir)
    elif script_id == "feature_importance":
        summary = run_feature_importance(input_dir, output_dir)
    else:
        raise AssertionError(script_id)

    write_summary(output_dir, summary)
