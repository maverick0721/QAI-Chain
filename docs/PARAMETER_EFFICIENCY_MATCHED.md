# Matched-Capacity Parameter Efficiency

Generated (UTC): 2026-03-21T03:12:28.348520+00:00

| Policy | Parameters | Mean Reward ± std | Safety Violations ± std | Inference ms/step | FLOPs (forward est.) |
|---|---:|---:|---:|---:|---:|
| MLP (28 params) | 28 | -1800.08 ± 81.53 | 75.46 ± 0.30 | 0.0837 | 56 |
| VQC (6q, 4l) | 28 | -1729.94 ± 25.69 | 75.43 ± 0.33 | 9.3438 | 56 |
| LoRA-MLP (trainable) | 396 | -1785.27 ± 102.43 | 75.47 ± 0.29 | 0.1464 | 10138 |
| Distilled Student | 97 | -1778.46 ± 74.12 | 75.44 ± 0.30 | 0.0799 | 194 |
| MLP (4673 params) | 4673 | -1770.52 ± 92.57 | 75.47 ± 0.29 | 0.0953 | 9346 |

