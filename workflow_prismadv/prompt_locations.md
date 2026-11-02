# Prompt locations

Paths are from the repository root.

| Method | Prompt |
|---|---|
| Zero-shot | `workflow_prismadv/eid_bench_baselines/pipelines/run_single_shot.py`|
| Few-shot | `workflow_prismadv/eid_bench_baselines/pipelines/run_few_shot.py`|
| PrismaDV, data flow | `prismadv/code_inspector/llm_code_inspector/{single,multiple}/prompt.py` |
| PrismaDV, other stages | `prismadv/llm/langchain/prompts/prismadv/`|
| PrismaDV, EIDBench-real | `prismadv/llm/langchain/models/prismadv_multi_table/prompts.py` |
| SIFTA proposer | `sifta/strategies/instruction_proposal.py` (`GlobalInstructionProposalSignature`) |
| GEPA proposer | `InstructionProposalSignature` of the `gepa` package 0.0.7 (`gepa/strategies/instruction_proposal.py`) |
| SIFTA / GEPA start program | `prismadv/llm/dspy/models/column_wise_module/signatures.py`; GEPA[S]: `workflow_sifta/baselines/gepa/script_level_module.py` |
| Codex | `workflow_prismadv/eid_bench_baselines/prompts/codex.md` |