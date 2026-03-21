# Constrained Baseline Suite

Generated (UTC): 2026-03-21T12:57:09.249639+00:00

## SCALED Environment
| Method | Reward (mean±std) | Violations | Target MAE | Fallback | Latency |
|---|---:|---:|---:|---:|---:|
| Random | -1733.24 ± 73.74 | 75.53 | 4.102 | 0.000 | 1.190 |
| Heuristic | -1672.96 ± 30.29 | 74.38 | 2.871 | 0.000 | 1.231 |
| CPO | -1780.92 ± 17.52 | 75.47 | 2.807 | 0.000 | 1.245 |
| P3O | -1823.24 ± 10.49 | 75.75 | 2.788 | 0.000 | 1.191 |
| PPO-Lagrangian | -1800.07 ± 11.84 | 75.56 | 2.807 | 0.000 | 1.226 |
| Robust QGate+Shield | -1728.10 ± 24.55 | 75.62 | 3.013 | 0.000 | 1.108 |

## DEFI Environment
| Method | Reward (mean±std) | Violations | Target MAE | Fallback | Latency |
|---|---:|---:|---:|---:|---:|
| Random | -126.54 ± 7.02 | 60.22 | 3.039 | 0.000 | 1.153 |
| Heuristic | -115.34 ± 3.57 | 59.22 | 3.410 | 0.000 | 1.003 |
| CPO | -113.90 ± 3.55 | 59.22 | 3.447 | 0.000 | 1.114 |
| P3O | -101.72 ± 3.46 | 59.22 | 3.725 | 0.000 | 1.118 |
| PPO-Lagrangian | -110.39 ± 3.57 | 59.22 | 3.528 | 0.000 | 1.115 |
| Robust QGate+Shield | -141.29 ± 8.37 | 59.22 | 2.527 | 0.000 | 1.119 |

