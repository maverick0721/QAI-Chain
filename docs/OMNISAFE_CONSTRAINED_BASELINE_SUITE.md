# OmniSafe Official Constrained Baselines

Generated (UTC): 2026-03-21T04:00:09.006368+00:00

Algorithms: CPO, P3O, PPOLag, SACLag, FOCOPS (official OmniSafe implementations)

## SCALED Environment
| Baseline | Final Return (mean±std) | Final Cost (mean±std) | Final Ep Len (mean±std) | p vs best return |
|---|---:|---:|---:|---:|
| CPO | -1620.45 ± 177.45 | 74.793 ± 0.991 | 80.00 ± 0.00 | 0.000 (**) |
| P3O | -1578.88 ± 276.81 | 73.939 ± 3.481 | 80.00 ± 0.00 | 0.000 (**) |
| PPOLag | -1446.77 ± 250.63 | 73.567 ± 2.506 | 80.00 ± 0.00 | 0.098 (ns) |
| SACLag | -1685.74 ± 21.02 | 75.462 ± 0.147 | 80.00 ± 0.00 | 0.000 (**) |
| FOCOPS | -1345.44 ± 348.35 | 71.632 ± 4.526 | 80.00 ± 0.00 | --- |

## DEFI Environment
| Baseline | Final Return (mean±std) | Final Cost (mean±std) | Final Ep Len (mean±std) | p vs best return |
|---|---:|---:|---:|---:|
| CPO | -112.42 ± 25.65 | 59.509 ± 0.726 | 80.00 ± 0.00 | 0.000 (**) |
| P3O | -115.68 ± 33.93 | 59.509 ± 0.726 | 80.00 ± 0.00 | 0.000 (**) |
| PPOLag | -91.60 ± 35.52 | 59.509 ± 0.726 | 80.00 ± 0.00 | 0.378 (ns) |
| SACLag | -118.33 ± 6.40 | 59.302 ± 0.678 | 80.00 ± 0.00 | 0.000 (**) |
| FOCOPS | -85.97 ± 27.54 | 59.509 ± 0.726 | 80.00 ± 0.00 | --- |

