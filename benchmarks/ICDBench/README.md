# ICDBench: Individual Constraint Discovery from Data-Code Pairs

## Overview

ICDBench is a benchmark for evaluating the ability of data validation systems to discover implicit data constraints from
code. It tests whether a system can identify hidden data assumptions embedded in downstream task code and translate them
into executable validation constraints.

## Benchmark Characteristics

- **Total Cases**: 63
- **Constraint Format**: PyDeequ syntax
- **Domains**: Payment processing, cricket sports rules, in-game auctions, and more
- **Sources**: Hand-crafted cases + real-world data-code pairs from public GitHub repositories