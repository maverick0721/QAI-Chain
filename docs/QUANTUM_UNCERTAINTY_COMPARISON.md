# Quantum vs Classical Uncertainty

Generated (UTC): 2026-03-21T12:43:44.753553+00:00

| Method | Raw ECE | Raw Failure-ECE | Calibrated Failure-ECE | Precision | Recall | Trigger Rate | Cost Units |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_sigma | 0.8710 | 0.2178 | 0.0000 | 1.000 | 0.113 | 0.100 | 1.0 |
| mc_dropout | 1.3835 | 0.4228 | 0.0000 | 0.833 | 0.094 | 0.100 | 10.0 |
| ensemble | 1.3834 | 0.3371 | 0.0000 | 0.814 | 0.092 | 0.100 | 5.0 |
| quantum_param_perturb | 1.3834 | 0.3830 | 0.0000 | 0.942 | 0.106 | 0.100 | 10.0 |

Failure-ECE scope: triggered_only_q90

## Hard-Regime (Intensity=1.0) Trigger Diagnostics
| Method | Precision | Recall | Trigger Rate |
|---|---:|---:|---:|
| gaussian_sigma | 1.000 | 0.105 | 0.093 |
| mc_dropout | 0.743 | 0.100 | 0.120 |
| ensemble | 0.792 | 0.089 | 0.100 |
| quantum_param_perturb | 0.964 | 0.101 | 0.093 |
