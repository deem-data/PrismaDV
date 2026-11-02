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

In the SIFTA table this is the **stats-novelty** row (program `stats_novelty`). It is
not the MMD stats-novelty row of the EIDBench tables.

## Scope: tests 1 and 3

The method is task-agnostic: it never sees task code or task outcomes. Its verdict on
a batch counts for every task, so it is scored on test 1 (train + eval scripts, test
labels) and test 3 (test scripts, test labels). Test 2 is N/A, because its labels are
the batches the detector is fit on.

## Protocol (split file `benchmarks/EIDBench-synth/<dataset>/splits/split<N>.yaml`)

- **Fit:** on the profiles of the corrupted batches of the `train_eval_labels`
  (`files_with_corrupted_new_data/new_data.csv`), all treated as acceptable. Safety
  labels are **not** used. The clean batch is not in the fit set.
- **Test:** one pass/reject verdict per `test_labels` batch, written per label as
  `validation_results_on_corrupted_test_data__sifta_stats_novelty[--r<N>].yaml`.
- **Metric:** stage 4 (`workflow_sifta._4_scoring`) scores the verdicts with the same
  macro F1 as the other rows; split N is replicate N.

## Usage

```bash
SPARK_VERSION=3.5 uv run python -m workflow_sifta.baselines.redyuk.run --datasets students --splits 1 2 3
```

A summary per split is written to `optimization_runs/stats_novelty/<dataset>_split<N>.json`.
