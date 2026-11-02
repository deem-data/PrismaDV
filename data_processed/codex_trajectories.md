# Codex trajectories

Every Codex run stores its trajectory in its constraint artifact, under the `trajectory:` key.

| Benchmark | Artifacts |
|---|---|
| EIDBench-synth | `data_processed/<dataset>/general/constraints/<task>/codex--gpt-6-sol--medium[--r<N>].yaml` |
| EIDBench-real | `data_processed/eid_bench_real/<application>/constraints/project/codex--gpt-6-sol[--r<N>]--<timestamp>.yaml` |
| ICDBench | `data_processed/icd_bench/<case>/constraints/codex--gpt-6-sol--medium[--r<N>].yaml` |
