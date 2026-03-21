# Benchmark Report

Generated (UTC): 2026-03-21T12:57:03.620677+00:00

| Component | avg (ms) | p50 (ms) | p90 (ms) | 95% CI (ms) |
|---|---:|---:|---:|---:|
| MetricsEncoder | 0.042 | 0.038 | 0.056 | [0.037, 0.047] |
| QNN | 69.690 | 69.699 | 70.475 | [69.186, 70.194] |
| QTransformer | 410.003 | 409.672 | 415.292 | [407.334, 412.672] |

Notes:
- Results are hardware-dependent and intended as reproducibility anchors.
- Rerun using `python scripts/generate_benchmark_report.py`.
