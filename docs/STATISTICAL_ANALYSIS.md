# Statistical Analysis

Generated (UTC): 2026-03-21T12:56:52.550905+00:00

## 95% CI (bootstrap) for Mean Reward (Last5)

| Method | Mean | Std | 95% CI Low | 95% CI High | n |
|---|---:|---:|---:|---:|---:|
| random | 9.484 | 6.338 | 6.397 | 12.546 | 16 |
| heuristic_target | 23.000 | 0.000 | 23.000 | 23.000 | 16 |
| ppo_full | 16.209 | 6.718 | 12.695 | 19.194 | 16 |
| ppo_no_entropy_anneal | 16.209 | 6.718 | 12.766 | 19.224 | 16 |
| ppo_no_adv_norm | 18.658 | 4.681 | 16.193 | 20.796 | 16 |

## Permutation Tests (Mean Reward Last5)

| Comparison | p-value | Cohen's d |
|---|---:|---:|
| ppo_vs_random | 0.0085 | 1.030 |
| ppo_vs_heuristic | 0.0000 | -1.430 |
