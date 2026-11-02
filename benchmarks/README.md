# PrismaDV Benchmarks

This directory contains complementary benchmarks for evaluating task-aware data validation systems. These benchmarks
test different aspects of the data validation problem described in the paper.

## Benchmark Overview

| Benchmark                 | Focus                   | Scale                |
|---------------------------|-------------------------|----------------------|
| **[ICDBench](ICDBench/)** | Constraint Discovery    | 63 test cases        |
| **[EIDBench-synth](EIDBench-synth/)** | End-to-End Error Impact | 5 datasets, 60 tasks, 370 error labels, 4,448 labeled batches |
| **[EIDBench-real](EIDBench-real/)** | Real Repo-Derived Error Impact Detection | 3 applications, 203 labeled batches |

The error types of both EIDBench benchmarks come from public incident reports and engineering blog posts, listed in
[incidents.md](incidents.md).
