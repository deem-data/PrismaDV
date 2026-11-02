# Task generation prompts

The prompts we used with gemini-2.5-pro to synthesize the EIDBench-synth tasks:

- `table_summarization/`: summarize the input table
- `data_generation/`: generate the tasks, their data assumptions and code
- `data_verification/`: check the generated assertions and remove bad ones

The generated tasks are in [`eid_bench_gen/`](../../../../eid_bench_gen), one folder per dataset.
