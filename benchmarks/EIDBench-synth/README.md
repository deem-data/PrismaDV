# EIDBench-synth: End-to-End Error Impact Detection Benchmark

## Datasets

| Dataset            | Source | Domain               | # Columns | # Tasks |
|--------------------|--------|----------------------|-----------|---------|
| students           | UCI ML | Academic performance | 37        | 10      |
| hr_analytics       | Kaggle | Employee attrition   | 38        | 12      |
| sleep_health       | Kaggle | Sleep & lifestyle    | 13        | 13      |
| IPL_win_prediction | Kaggle | Cricket matches      | 20        | 15      |
| imdb               | Kaggle | Movies & TV shows    | 16        | 10      |

## Error Configs and Labels

The corruption configs are grouped by error type. Each dataset has one folder per error group,
`<dataset>/errors/error-group-<k>/general_task_<n>.yaml`, and `n` counts from 1 in each group. The config is the
processed-data label `<k>-<n>` under `data_processed/<dataset>/general/<k>-<n>/`:

## Dataset Generation

The 60 tasks in EIDBench-synth were generated using an **LLM-assisted, human-in-the-loop pipeline**.

**Documentation**:

- **Generation pipeline**: [workflow_prismadv/eid_bench_building/](../../workflow_prismadv/eid_bench_building/) -
  Scripts and methodology
- **Intermediate outputs**: [eid_bench_gen/](../../eid_bench_gen/) - raw generated tasks before final selection
  (undocumented; see the generation pipeline above for the format)
