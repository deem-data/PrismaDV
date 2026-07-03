# Bank Marketing Analysis

This example adapts `UBC-MDS/bank-marketing-analysis`, a Python ML pipeline
over the UCI Bank Marketing dataset. The benchmark table is the upstream
full Bank Marketing CSV normalized to `bank_marketing.csv` with 45,211 rows.

## Upstream Repair

The upstream scripts read the raw CSV with `index_col=0`, which turns
`age` into the dataframe index and drops it from the model feature matrix. That
is not intended by the project description or the dataset schema, so the
benchmark uses a repaired source copy where `age` remains a normal column and
processed CSV outputs are written without accidental index columns.

This repair should be reported with the benchmark result because it changes the
feature set while preserving the intended upstream pipeline behavior.

## Environment

The benchmark environment is managed with `uv` from this directory:

```bash
uv sync
uv run python --version
```

Adapter commands should be run through this environment rather than a shared
global Python environment.
