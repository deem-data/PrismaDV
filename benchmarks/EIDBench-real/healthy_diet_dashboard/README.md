# Healthy Diet Dashboard

This example adapts `UBC-MDS/DSCI-532_2026_29_healthy-diet`, a Python ETL +
Shiny dashboard over the FAO/World Bank "Cost of a Healthy Diet" dataset
(2017-2024). The benchmark exposes the cleaning step
(`src/scripts/clean_data.py`) as a single ETL script that joins the raw price
data with a country-code lookup and a country-to-continent lookup, normalizes
country names, and writes `cleaned_price_of_healthy_diet.{csv,parquet}`.

## Upstream adaptation

The upstream `clean_data.py` hard-codes paths under
`src/data/{raw,lookups,processed}`. The benchmark copy parameterizes I/O via
Click options:

```
python src/scripts/clean_data.py \
  --raw_data <path/to/price_of_healthy_diet.csv> \
  --country_codes <path/to/country_codes.csv> \
  --continent_lookup <path/to/countries_by_continents.csv> \
  --save_to <output/dir>
```

`src/app.py` (Shiny dashboard) is kept and rebuilt to run on the trimmed
benchmark environment: the AI Chatbot panel and its `querychat` / `chatlas` /
`anthropic` / `python-dotenv` server handlers have been stripped, and the
on-startup `download_dataset()` + `clean_dataset()` calls (which depended on
Kaggle and the original no-arg `clean_dataset` signature) have been removed.
The dashboard's charts, filters, KPI cards, and click-to-filter handlers are
unchanged. `src/scripts/download_data.py` is kept in-tree for reference but
is not imported or invoked by the benchmark or by `app.py`. The Kaggle,
Anthropic, dotenv, chatlas, and querychat dependencies are dropped from
the benchmark `pyproject.toml`.

## Tables

| table | rows | notes |
|---|---|---|
| `price_of_healthy_diet` | 1,379 | raw FAO/World Bank rows, joined on `country_code` |
| `country_codes` | 249 | ISO numeric/alpha-2/alpha-3 lookup (BOM in header) |
| `countries_by_continents` | 195 | country → continent lookup (BOM in header) |

## Cleaned output

`cleaned_price_of_healthy_diet.csv` has 1,379 rows × 12 columns (raw columns +
`Alpha-3 code`). `region` is replaced with the joined `Continent` where the
country-name canonicalization + manual territory overrides find a match;
otherwise the raw `region` is kept.

## Environment

The benchmark environment is managed with `uv` from this directory:

```bash
uv sync
uv run python --version
uv run python adapter/run_script.py \
  --script-id clean_data \
  --input files/clean/tables \
  --output /tmp/healthy_diet_clean_run
```

Adapter commands should be run through this environment rather than a shared
global Python environment.
