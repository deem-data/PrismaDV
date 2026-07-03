# EIDBench-real

- `bank_marketing_analysis`: Python ML workflow over marketing data.
- `healthy_diet_dashboard`: DuckDB/Pandas ETL workflow over diet and country reference tables.
- `omop_cdm_databricks`: PySpark/R healthcare workflow over OMOP-style clinical tables.

Together these applications provide 64 labeled input batches: 26 batches that
should pass and 38 batches that should be rejected.

## Scope

Each benchmark application includes:

- a clean input bundle used as the sample data,
- corrupted input bundles with task-specific injected errors,
- normalized CSV tables for PrismaDV and constraint validation,
- the runnable task code or adapter wrapper,
- a `manifest.yaml` that declares tables, scripts, runtime expectations, and labels, and
- stable execution summaries or expected signals for evaluating task outcomes.

Adapters may materialize normalized CSV tables for PrismaDV while preserving
native files for pipeline execution.

## Adapter Contract

Every declared script should run through:

```bash
python adapter/run_script.py --script-id <script_id> --input <input_dir> --output <output_dir>
```