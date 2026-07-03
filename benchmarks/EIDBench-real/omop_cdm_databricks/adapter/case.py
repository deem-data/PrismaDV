"""Adapter implementation for the OMOP CDM Databricks EIDBench-real case.

The upstream Databricks notebooks live under `adapter/source_repo/` and have
been edited in-place to be locally runnable (no `dbutils`, no `delta.\`path\``,
no `-- COMMAND ----------` cell separators, no Hive-only `CREATE OR REPLACE
TABLE`, no Delta-only `USING DELTA`). This file orchestrates them:

1. boot a local Spark session with Hive support + parquet defaults
2. register the Synthea CSV inputs as bronze temp views
   (`source_repo/1-data-ingest.py::register_bronze_views`)
3. execute the OMOP CDM DDL (`source_repo/2-omop531-cdm-setup.sql`)
4. INSERT OVERWRITE the OMOP vocab tables from the pre-filtered Athena
   subset under `files/clean/tables/vocab/`
5. run the source-to-vocab map building SQL (`source_repo/3-omop-vocab-setup.sql`)
6. execute the Synthea→OMOP ETL (`source_repo/4-omop531-etl-synthea.sql`)
7. (drug_analysis) run the four analytical queries from
   `source_repo/6-drug-analysis.py::run_drug_analysis`
8. (chf_cohort) export OMOP tables to SQLite, shell out to
   `pixi run -- Rscript source_repo/5-CHF-cohort-building.r`
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ADAPTER_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = ADAPTER_DIR.parent
SOURCE_REPO = ADAPTER_DIR / "source_repo"

DATA_INGEST_PY = SOURCE_REPO / "1-data-ingest.py"
CDM_SETUP_SQL = SOURCE_REPO / "2-omop531-cdm-setup.sql"
VOCAB_SETUP_SQL = SOURCE_REPO / "3-omop-vocab-setup.sql"
ETL_SQL = SOURCE_REPO / "4-omop531-etl-synthea.sql"
CHF_R_SCRIPT = SOURCE_REPO / "5-CHF-cohort-building.r"
DRUG_ANALYSIS_PY = SOURCE_REPO / "6-drug-analysis.py"

PIXI_BIN = Path.home() / ".pixi" / "bin" / "pixi"

ALL_SCRIPTS = ("sql_etl", "drug_analysis", "chf_cohort")
OMOP_OUTPUT_TABLES = (
    "cdm_source", "condition_era", "condition_occurrence",
    "drug_era", "drug_exposure", "measurement",
    "observation_period", "person", "procedure_occurrence", "visit_occurrence",
)
VOCAB_TABLES = (
    "CONCEPT", "CONCEPT_RELATIONSHIP", "CONCEPT_ANCESTOR",
    "VOCABULARY", "DOMAIN", "CONCEPT_CLASS", "RELATIONSHIP",
)
CHF_REQUIRED_TABLES = (
    "condition_occurrence", "drug_exposure", "observation_period",
    "visit_occurrence", "person", "concept", "concept_ancestor",
)


# ---- module-load helpers for upstream Python files -----------------------

def _load_source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Upstream Python files are loaded lazily inside the script runners so that
# pyspark import only happens when a script actually runs.

def _load_data_ingest() -> Any:
    return _load_source_module("source_repo_data_ingest", DATA_INGEST_PY)


def _load_drug_analysis() -> Any:
    return _load_source_module("source_repo_drug_analysis", DRUG_ANALYSIS_PY)


# ---- Spark session -------------------------------------------------------

def ensure_local_java_compat() -> None:
    compat_flag = "-Djava.security.manager=allow"
    current = os.environ.get("SPARK_SUBMIT_OPTS", "").strip()
    if compat_flag not in current.split():
        os.environ["SPARK_SUBMIT_OPTS"] = f"{compat_flag} {current}".strip()


def create_spark(app_name: str, output_dir: Path) -> Any:
    ensure_local_java_compat()
    from pyspark.sql import SparkSession

    warehouse = output_dir / "_warehouse"
    derby = output_dir / "_derby"
    warehouse.mkdir(parents=True, exist_ok=True)
    derby.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.sql.sources.default", "parquet")
        .config("spark.sql.legacy.createHiveTableByDefault", "false")
        .config("spark.sql.storeAssignmentPolicy", "LEGACY")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.sql.legacy.parquet.datetimeRebaseModeInWrite", "LEGACY")
        .config("spark.sql.legacy.parquet.datetimeRebaseModeInRead", "LEGACY")
        .config("spark.driver.extraJavaOptions",
                f"-Dderby.system.home={derby}")
        .enableHiveSupport()
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---- SQL execution -------------------------------------------------------

# Match `-- [python cell handled by Python adapter] ...` lines that the
# upstream-edit step left behind in the SQL files.
_PYTHON_PLACEHOLDER_RE = re.compile(r"^\s*--\s*\[python cell.*$", re.MULTILINE)


def split_sql_statements(sql_text: str) -> list[str]:
    """Strip line comments + placeholder markers, then split on `;`."""
    text = _PYTHON_PLACEHOLDER_RE.sub("", sql_text)
    cleaned_lines = []
    for line in text.splitlines():
        if line.strip().startswith("--"):
            continue
        cleaned_lines.append(line)
    return [stmt.strip() for stmt in "\n".join(cleaned_lines).split(";") if stmt.strip()]


def execute_sql_file(spark: Any, path: Path) -> dict[str, Any]:
    statements = split_sql_statements(path.read_text(encoding="utf-8"))
    executed = 0
    failures: list[dict[str, Any]] = []
    for index, stmt in enumerate(statements):
        try:
            spark.sql(stmt)
            executed += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "file": path.name,
                "stmt_index": index,
                "statement_head": stmt.splitlines()[0][:160],
                "error": f"{type(exc).__name__}: {exc}"[:400],
            })
    return {
        "file": path.name,
        "statements_total": len(statements),
        "statements_executed": executed,
        "failures": failures,
    }


# ---- Vocab loading -------------------------------------------------------

def load_vocab_subset(spark: Any, input_dir: Path) -> dict[str, int]:
    from pyspark.sql import functions as F

    vocab_dir = input_dir / "vocab"
    if not vocab_dir.exists():
        raise FileNotFoundError(
            f"vocab subset not found at {vocab_dir}. "
            f"Generate it with `uv run python scripts/filter_omop_vocab.py`."
        )
    counts: dict[str, int] = {}
    for table in VOCAB_TABLES:
        path = vocab_dir / f"{table}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        df = spark.read.csv(str(path), header=True, inferSchema=True)
        for col in ("valid_start_date", "valid_end_date"):
            if col in df.columns:
                df = df.withColumn(col, F.to_date(F.col(col), "yyyy-MM-dd"))
        df.createOrReplaceTempView(f"_load_{table.lower()}")
        column_list = ", ".join(df.columns)
        spark.sql(
            f"INSERT OVERWRITE TABLE {table} ({column_list}) "
            f"SELECT {column_list} FROM _load_{table.lower()}"
        )
        counts[table] = df.count()
    return counts


# ---- OMOP table accounting ----------------------------------------------

def collect_omop_table_counts(spark: Any) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for table in OMOP_OUTPUT_TABLES:
        try:
            counts[table] = int(spark.sql(f"SELECT COUNT(*) AS n FROM {table}").collect()[0]["n"])
        except Exception:
            counts[table] = None
    return counts


# ---- Pipeline orchestration ---------------------------------------------

def _build_omop_dataset(spark: Any, input_dir: Path) -> dict[str, Any]:
    data_ingest = _load_data_ingest()
    bronze_counts = data_ingest.register_bronze_views(spark, input_dir)
    cdm_setup_report = execute_sql_file(spark, CDM_SETUP_SQL)
    vocab_counts = load_vocab_subset(spark, input_dir)
    vocab_map_report = execute_sql_file(spark, VOCAB_SETUP_SQL)
    etl_report = execute_sql_file(spark, ETL_SQL)
    omop_counts = collect_omop_table_counts(spark)
    return {
        "bronze_input_counts": bronze_counts,
        "cdm_setup": cdm_setup_report,
        "vocab_loaded": vocab_counts,
        "vocab_maps": vocab_map_report,
        "etl": etl_report,
        "omop_output_counts": omop_counts,
    }


# ---- Script entry points ------------------------------------------------

def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_sql_etl(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    spark = create_spark("omop_cdm_databricks_sql_etl", output_dir)
    try:
        result = _build_omop_dataset(spark, input_dir)
    finally:
        spark.stop()
    return {"script_id": "sql_etl", **result}


def run_drug_analysis(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    spark = create_spark("omop_cdm_databricks_drug_analysis", output_dir)
    try:
        pipeline = _build_omop_dataset(spark, input_dir)
        drug_analysis_module = _load_drug_analysis()
        analysis = drug_analysis_module.run_drug_analysis(spark)
    finally:
        spark.stop()
    return {"script_id": "drug_analysis", "pipeline": pipeline, "analysis": analysis}


# ---- chf_cohort ---------------------------------------------------------

def export_omop_tables_to_sqlite(spark: Any, sqlite_path: Path,
                                 tables: tuple[str, ...] = CHF_REQUIRED_TABLES) -> dict[str, int]:
    import sqlite3

    if sqlite_path.exists():
        sqlite_path.unlink()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with sqlite3.connect(str(sqlite_path)) as conn:
        for table in tables:
            pdf = spark.sql(f"SELECT * FROM {table}").toPandas()
            for col in pdf.columns:
                dtype_name = str(pdf[col].dtype).lower()
                if "datetime" in dtype_name or "date" in dtype_name:
                    pdf[col] = pdf[col].astype("string").where(pdf[col].notna(), None)
            pdf.to_sql(table, conn, if_exists="replace", index=False)
            counts[table] = int(len(pdf))
    return counts


def run_chf_cohort(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    if not PIXI_BIN.exists():
        raise FileNotFoundError(
            f"pixi binary not found at {PIXI_BIN}; install with "
            "`curl -fsSL https://pixi.sh/install.sh | bash` and run `pixi install` "
            "in this example directory."
        )
    spark = create_spark("omop_cdm_databricks_chf_cohort", output_dir)
    try:
        pipeline = _build_omop_dataset(spark, input_dir)
        sqlite_path = output_dir / "omop_for_r.sqlite"
        table_counts = export_omop_tables_to_sqlite(spark, sqlite_path)
    finally:
        spark.stop()

    cohort_json_path = output_dir / "chf_cohort_summary.json"
    proc = subprocess.run(
        [str(PIXI_BIN), "run", "--manifest-path", str(EXAMPLE_DIR / "pixi.toml"),
         "--", "Rscript", str(CHF_R_SCRIPT), str(sqlite_path), str(cohort_json_path)],
        capture_output=True,
        text=True,
        cwd=str(EXAMPLE_DIR),
    )
    r_stdout = (proc.stdout or "").strip().splitlines()
    r_stderr = (proc.stderr or "").strip().splitlines()
    if proc.returncode != 0:
        return {
            "script_id": "chf_cohort",
            "pipeline": pipeline,
            "sqlite_export_counts": table_counts,
            "status": "rscript_failed",
            "rscript_returncode": proc.returncode,
            "rscript_stdout_tail": r_stdout[-10:],
            "rscript_stderr_tail": r_stderr[-20:],
        }
    cohort_summary = json.loads(cohort_json_path.read_text(encoding="utf-8"))
    return {
        "script_id": "chf_cohort",
        "pipeline": pipeline,
        "sqlite_export_counts": table_counts,
        "rscript_stdout_tail": r_stdout[-5:],
        "cohort": cohort_summary,
    }


def run_script(script_id: str, input_dir: Path, output_dir: Path) -> None:
    if script_id not in ALL_SCRIPTS:
        raise ValueError(f"unknown script_id: {script_id}")
    clean_output_dir(output_dir)
    if script_id == "sql_etl":
        summary = run_sql_etl(input_dir, output_dir)
    elif script_id == "drug_analysis":
        summary = run_drug_analysis(input_dir, output_dir)
    elif script_id == "chf_cohort":
        summary = run_chf_cohort(input_dir, output_dir)
    else:
        raise AssertionError(script_id)
    write_summary(output_dir, summary)
