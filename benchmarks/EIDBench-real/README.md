# EIDBench-real

- `bank_marketing_analysis`: Python ML workflow over marketing data.
- `healthy_diet_dashboard`: DuckDB/Pandas ETL workflow over diet and country reference tables.
- `omop_cdm_databricks`: PySpark/R healthcare workflow over OMOP-style clinical tables.

## Scope

Each benchmark application includes:

- a clean input bundle used as the sample data,
- corrupted input bundles with task-specific injected errors,
- normalized CSV tables for PrismaDV and constraint validation,
- the runnable task code or adapter wrapper,
- a `manifest.yaml` that declares tables, scripts, runtime expectations, and labels