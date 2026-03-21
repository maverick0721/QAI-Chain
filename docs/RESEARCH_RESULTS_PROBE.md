# Research Results

Generated (UTC): 2026-03-19T18:13:26.147855+00:00

## Configuration

- episodes: 45
- steps_per_episode: 30
- seeds: [3, 7, 11, 17, 21, 42, 84, 126, 5, 9, 13, 19, 23, 31, 57, 99]
- start_difficulty: 7

## Main Comparison

| Method | Mean Reward (Last5) | Std Across Seeds | Final Abs Target Error |
|---|---:|---:|---:|
| random | 9.484 | 6.338 | 2.250 |
| heuristic_target | 23.000 | 0.000 | 0.000 |
| ppo_full | 16.209 | 6.718 | 1.688 |

## PPO Ablation

| Variant | Mean Reward (Last5) | Std Across Seeds |
|---|---:|---:|
| ppo_full | 16.209 | 6.718 |
| ppo_no_entropy_anneal | 16.209 | 6.718 |
| ppo_no_adv_norm | 18.658 | 4.681 |
