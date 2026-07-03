# Redyuk et al. (EDBT 2021) — Task-Agnostic Novelty-Detection Baseline

A standalone, **non-LLM** batch-level data-quality validator from:

> S. Redyuk, Z. Kaoudi, V. Markl, S. Schelter.
> *Automating Data Quality Validation for Dynamic Data Ingestion.* EDBT 2021.
> Code: https://github.com/sergred/automating-data-quality-validation-data

It computes descriptive statistics per data batch (completeness, uniqueness,
approx-distinct via HyperLogLog, most-frequent-value ratio, numeric
min/mean/max/std/sum, and a text *index of peculiarity*), then fits a **one-class
kNN novelty detector** (mean distance to `k=5` neighbours, `contamination=1%`,
Euclidean, MinMax-scaled) on the statistics of previously-observed batches. A new
batch is flagged **reject** if it is an outlier, **pass** otherwise.

## Scope: New Data Batches only

This method is **purely task-agnostic** — it models the *data distribution* of past
batches and never sees task code or task outcomes. It therefore requires
previously-observed batches of the same dataset to define "acceptable", which only
exists in the **New Data Batches** scenario (`test_1_cross_new_data`). It is
**N/A** for *New Tasks* and *New Data Batches + New Tasks*.

The summary it writes populates only the `test_1_cross_new_data` block; the other
two are marked `"N/A"`.

## Protocol (mirrors the SIFTA `test_1` split)

- Batches (`processed_data_label`) are split 1:1 into **D_train** (observed) and
  **D_new** (held-out). Reuse a SIFTA run's exact splits with
  `--split_strategy_from`, or generate them with the SIFTA default seed/ratios.
- **Train:** fit the detector on the feature vectors of the D_train batches' actually
  ingested data (`files_with_corrupted_new_data/new_data.csv`), all treated as
  "acceptable" (their safety labels are **not** used).
- **Test:** one pass/reject decision per D_new batch.
- The decision is broadcast across every task in (T_train ∪ T_val) =
  `train_eval_script_name_list` to form the same `(task, batch)` cells SIFTA is
  scored on (the decision is identical regardless of task).
- **Metric:** F1 with positive class = "unsafe" (reject), against ground-truth
  `corrupted_data_is_safe` per cell — identical to the other baselines.

## Usage

```bash
poetry install --with baseline   # pyod + hyperloglog (optional group)

# Reuse a SIFTA reference run's splits (recommended for a fair comparison):
poetry run python workflow_sifta/baselines/redyuk/run.py \
    --dataset_name hr_analytics --split_strategy_from run_20260111_155751

# Or generate splits from scratch with the SIFTA defaults (seed=1, 50/50 labels):
poetry run python workflow_sifta/baselines/redyuk/run.py --dataset_name hr_analytics
```

Results are written to `optimization_runs/baseline_redyuk_run_<timestamp>/summary.json`.

## Implementation notes / deviations from the original `demo.py`

- The statistics + kNN detector are vendored faithfully in
  [`data_profiler.py`](data_profiler.py). `dabl` is dropped (computed but unused in
  the original `compute_for`), and `nltk.util.ngrams` is replaced by an identical
  inline helper.
- The per-column metric set ("schema") is fixed from a clean reference batch so every
  batch yields a constant-length vector even when pandas infers a corrupted column's
  dtype differently. Numeric columns are coerced with `pd.to_numeric(errors="coerce")`
  and aggregated with NaN-aware reducers.
