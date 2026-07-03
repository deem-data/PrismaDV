# OMOP CDM (Databricks Synthea→OMOP)

This example adapts `databricks-industry-solutions/omop-cdm`, a Databricks
Solution Accelerator that maps Synthea synthetic healthcare data into the
OMOP Common Data Model v5.3.1 via a 1,225-line SQL ETL, then runs analytical
downstream scripts in Python (drug exposure analysis) and R (CHF cohort
identification).

It exercises the full Synthea→OMOP SQL ETL — declarative SQL ETL paradigm +
multi-language downstream (Python + R) + 30+ OMOP table schema.

## Layout

```
omop_cdm_databricks/
├── README.md, manifest.yaml
├── pyproject.toml          # uv: pyspark==3.5.1 + pandas
├── pixi.toml               # pixi: r-base + DBI + RSQLite + dplyr + glue + jsonlite
├── adapter/
│   ├── case.py             # Spark orchestration; imports source_repo modules
│   ├── run_script.py       # CLI wrapper
│   └── source_repo/        # gitignored: upstream files, edited in-place (see manifest.yaml)
├── scripts/
│   └── filter_omop_vocab.py            # one-shot Athena → 1.2 MB local subset
├── expected/
│   ├── sql_etl/clean_summary.json
│   ├── drug_analysis/clean_summary.json
│   └── chf_cohort/clean_summary.json
└── files/
    ├── clean/tables/                   # 6 Synthea CSVs + vocab/ (filtered Athena)
    └── observed/tables/                # mirrors clean
```

## Scripts

`uv run python adapter/run_script.py --script-id <id> --input files/clean/tables --output <dir>`

| script_id | what it does | clean-run output |
|---|---|---|
| `sql_etl` | Synthea → OMOP CDM via the upstream SQL files | 10 OMOP tables populated (3K persons, 80K visits, 23K conditions, 108K drug exposures, 456K measurements, 11K drug eras, …) |
| `drug_analysis` | Reuses `sql_etl` then runs 4 analytical queries from upstream `6-drug-analysis.py` (Examples 1, 2, 3, 4.1) | Oxycodone usage 25 rows, top drugs by year 2536 rows, Hydro/Oxy age distro 220 rows, drug-pair cosine 567 rows |
| `chf_cohort` | Reuses `sql_etl`, exports 7 OMOP tables to SQLite, shells out to `pixi run -- Rscript 5-CHF-cohort-building.r` | 196 CHF patients in target cohort across 12 (gender × age) groups; 1,945 ER visits |

## Tables

The benchmark's `manifest.yaml` lists 6 input tables (the Synthea bronze
subset). Constraint inference operates over these tables plus the code
context (SQL ETL + drug analysis + CHF cohort scripts) under
`adapter/source_repo/`.

| table | rows in clean bundle |
|---|---|
| `patients` | 3,000 |
| `encounters` | 83,781 |
| `conditions` | 29,828 |
| `medications` | 115,459 |
| `observations` | 488,950 |
| `procedures` | 28,861 |

Plus the OMOP vocab subset under `files/clean/tables/vocab/` (~1.2 MB
total, 7 gzipped CSVs).

## Environment

Two managers side by side:

```bash
uv sync                       # Python (PySpark)
~/.pixi/bin/pixi install      # R (DBI + RSQLite + dplyr/glue/jsonlite), ~830 MB
```
