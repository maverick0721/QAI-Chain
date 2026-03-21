# Real-Trace Replay Experiment

Generated (UTC): 2026-03-20T15:01:56.542936+00:00

Trace source: Etherscan Ethereum daily gas price history

## SCALED Environment
| Method | Reward (mean±std) | Violations | Target MAE | Fallback | Latency |
|---|---:|---:|---:|---:|---:|
| Heuristic | -3108.62 ± 25.78 | 116.11 | 2.924 | 0.000 | 1.258 |
| PPO-Lagrangian | -3265.31 ± 30.61 | 116.36 | 2.922 | 0.000 | 1.253 |
| Robust QGate+Shield | -3138.95 ± 26.25 | 116.40 | 3.029 | 0.000 | 1.103 |

## DEFI Environment
| Method | Reward (mean±std) | Violations | Target MAE | Fallback | Latency |
|---|---:|---:|---:|---:|---:|
| Heuristic | 177.19 ± 2.53 | 50.97 | 11.338 | 0.000 | 1.002 |
| PPO-Lagrangian | 206.38 ± 2.50 | 50.97 | 11.903 | 0.000 | 1.109 |
| Robust QGate+Shield | 219.79 ± 24.02 | 50.97 | 11.587 | 0.000 | 1.118 |

