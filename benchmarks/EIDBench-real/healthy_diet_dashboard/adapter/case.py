"""Adapter implementation for the Healthy Diet Dashboard EIDBench-real case."""

from __future__ import annotations

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
ALL_SCRIPTS = ("clean_data",)

RAW_TABLE = "price_of_healthy_diet.csv"
COUNTRY_CODES_TABLE = "country_codes.csv"
CONTINENT_TABLE = "countries_by_continents.csv"
OUTPUT_CSV = "cleaned_price_of_healthy_diet.csv"
OUTPUT_PARQUET = "cleaned_price_of_healthy_diet.parquet"


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def find_input(input_dir: Path, name: str) -> Path:
    path = input_dir / name
    if not path.exists():
        raise FileNotFoundError(f"{name} not found under {input_dir}")
    return path


def run_python(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=SOURCE_REPO, check=True)


def write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    finite = s.dropna()
    if finite.empty:
        return {"count": 0, "missing": int(series.isna().sum()), "min": None, "max": None, "mean": None}
    return {
        "count": int(finite.size),
        "missing": int(series.size - finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def run_clean_data(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    raw_csv = find_input(input_dir, RAW_TABLE)
    codes_csv = find_input(input_dir, COUNTRY_CODES_TABLE)
    continent_csv = find_input(input_dir, CONTINENT_TABLE)

    data_dir = output_dir / "data" / "processed"
    ensure_dirs(data_dir)

    run_python([
        "src/scripts/clean_data.py",
        "--raw_data", str(raw_csv),
        "--country_codes", str(codes_csv),
        "--continent_lookup", str(continent_csv),
        "--save_to", str(data_dir),
    ])

    output_csv = data_dir / OUTPUT_CSV
    output_parquet = data_dir / OUTPUT_PARQUET

    raw_df = pd.read_csv(raw_csv, encoding="utf-8-sig")
    codes_df = pd.read_csv(codes_csv, encoding="utf-8-sig")
    continent_df = pd.read_csv(continent_csv, encoding="utf-8-sig")
    clean_df = pd.read_csv(output_csv, encoding="utf-8-sig")

    columns = list(clean_df.columns)
    year_summary = _numeric_summary(clean_df["year"]) if "year" in clean_df.columns else {}
    year_range = (
        [int(year_summary["min"]), int(year_summary["max"])]
        if year_summary.get("min") is not None
        else [None, None]
    )

    rows_per_region = (
        clean_df["region"].fillna("<missing>").value_counts().sort_index().to_dict()
        if "region" in clean_df.columns
        else {}
    )
    countries_per_region = (
        clean_df.dropna(subset=["region", "country"])
        .groupby("region")["country"]
        .nunique()
        .sort_index()
        .to_dict()
        if {"region", "country"}.issubset(clean_df.columns)
        else {}
    )
    cost_categories = (
        sorted(clean_df["cost_category"].dropna().unique().tolist())
        if "cost_category" in clean_df.columns
        else []
    )
    regions = (
        sorted(clean_df["region"].dropna().unique().tolist())
        if "region" in clean_df.columns
        else []
    )

    summary = {
        "script_id": "clean_data",
        "inputs": {
            "raw_rows": int(len(raw_df)),
            "raw_columns": list(raw_df.columns),
            "country_codes_rows": int(len(codes_df)),
            "continent_lookup_rows": int(len(continent_df)),
        },
        "outputs": {
            "cleaned_csv_exists": output_csv.exists(),
            "cleaned_parquet_exists": output_parquet.exists(),
            "rows": int(len(clean_df)),
            "columns": columns,
            "column_count": len(columns),
            "alpha3_column_present": "Alpha-3 code" in columns,
            "alpha3_missing_rows": int(clean_df["Alpha-3 code"].isna().sum())
                if "Alpha-3 code" in columns else None,
            "region_missing_rows": int(clean_df["region"].isna().sum())
                if "region" in columns else None,
            "country_missing_rows": int(clean_df["country"].isna().sum())
                if "country" in columns else None,
        },
        "stats": {
            "n_countries": int(clean_df["country"].nunique()) if "country" in columns else 0,
            "regions": regions,
            "cost_categories": cost_categories,
            "year_range": year_range,
            "cost_healthy_diet_ppp_usd": _numeric_summary(clean_df["cost_healthy_diet_ppp_usd"])
                if "cost_healthy_diet_ppp_usd" in columns else {},
            "rows_per_region": {str(k): int(v) for k, v in rows_per_region.items()},
            "countries_per_region": {str(k): int(v) for k, v in countries_per_region.items()},
        },
    }
    return summary


def run_script(script_id: str, input_dir: Path, output_dir: Path) -> None:
    if script_id not in ALL_SCRIPTS:
        raise ValueError(f"unknown script_id: {script_id}")

    clean_output_dir(output_dir)

    if script_id == "clean_data":
        summary = run_clean_data(input_dir, output_dir)
    else:
        raise AssertionError(script_id)

    write_summary(output_dir, summary)
