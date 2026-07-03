# SIFTA: Selective Informative Feedback for Task Adaptation

SIFTA is a prompt optimization framework that adapts PrismaDV from observed task and test outcomes. It selects
informative constraint failures using failure precision, traces them back to source assumptions and code locations, and
uses that context to update PrismaDV prompts.

## Overview

SIFTA enhances PrismaDV's constraint generation by:

- **Failure selection**: Identifying informative constraint failures from validation and task outcomes
- **Backtracing**: Linking failing constraints to source assumptions and code locations
- **Prompt refinement**: Updating PrismaDV prompts from task-specific feedback
- **Task adaptation**: Improving pass/reject decisions on new data batches and tasks

## Architecture

The SIFTA implementation includes:

- **[`core/`](./core)** - Optimization Engine and necessary data models, mainly from existing GEPA project
- **[`strategies/`](./strategies)** - Task-aware data validation-related optimization instructions
- **[`samplers/`](./samplers)** - fail precision based batch sampler for training examples
- **[`proposer/`](./proposer)** - Prompt proposal logic
- **[`dspy_sifta/`](./dspy_sifta)** - SIFTA adaptation to DSPy
- **[`api.py`](./api.py)** - The optimization loop
- **[`sifta_utils.py`](./sifta_utils.py)** - Utility functions

## Usage

See the [SIFTA workflow documentation](../workflow_sifta/) for examples and experimental results.

## Attribution

SIFTA is written based on [GEPA (Genetic Prompt Algorithm)](https://github.com/gepa-ai/gepa), an MIT-licensed framework
for
reflective prompt evolution.

### Original Work

- **Project**: GEPA - Reflective Prompt Evolution Can Outperform Reinforcement Learning
- **Authors**: Lakshya A Agrawal and collaborators
- **License**: MIT License
- **Repository**: https://github.com/gepa-ai/gepa
- **Paper**: arXiv:2507.19457

### Modifications

This implementation extends GEPA's core concepts for the specific domain of data unit tests generation,
including:

- Integration with PrismaDV's constraint generation pipeline
- Specialized failure-precision-based example selection for data quality rules
- Global prompt proposing logic

## License

The original GEPA code is licensed under the MIT License. See [NOTICE](./NOTICE) for the complete license text and
copyright information.
