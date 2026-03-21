# Flagship Results

This page is the one-glance technical summary for engineering/research reviewers.

## Project Thesis

QAI-Chain demonstrates a unified architecture where decentralized infrastructure can integrate:

- adaptive AI governance signals,
- post-quantum transaction integrity,
- and quantum model experimentation,

while staying testable and reproducible as a software system.

## Reliability Signals

- CI quality gates: lint + type checks + tests + healthcheck
- Automated integration and contract tests for RPC transaction and mining flows
- Deterministic project healthcheck script for syntax/import/runtime smoke
- Reproducible benchmark artifact generation into docs and experiments

## Benchmark Snapshot (CPU)

Command:

```bash
python scripts/generate_benchmark_report.py
```

Artifacts:

- [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md)
- [experiments/benchmarks/latest.json](experiments/benchmarks/latest.json)

Latest local sample:

| Component | avg (ms) | p50 (ms) | p90 (ms) |
|---|---:|---:|---:|
| MetricsEncoder | 0.047 | 0.040 | 0.061 |
| QNN | 70.967 | 70.705 | 72.065 |
| QTransformer | 418.068 | 416.360 | 423.010 |

These values are machine-dependent and should be treated as reproducibility anchors, not universal claims.

## What Is Proven Today

- Core modules import cleanly
- Node RPC paths are integration-tested
- AI + quantum code paths run in smoke checks
- End-to-end test suite passes locally

## What Comes Next

- add reproducible experiment reports with confidence intervals
- define and test one flagship claim against strong baselines
- publish ablation study for AI-only vs hybrid AI+quantum paths