# Standard Constrained Transfer

Generated (UTC): 2026-03-21T09:30:29.704140+00:00
Status: ok
Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
Total steps per run: 3000
Steps per epoch: 1000
Environments: ['SafetyPointGoal1-v0', 'SafetyHalfCheetahVelocity-v1']
Algorithms: ['CPO', 'P3O', 'PPOLag']

## Results
### SafetyPointGoal1-v0
| Algo | Seeds | Return mean±std | Cost mean±std | EpLen mean±std |
|---|---:|---:|---:|---:|
| CPO | 30 | -0.42 +- 0.89 | 65.77 +- 70.45 | 1000.00 +- 0.00 |
| P3O | 30 | -0.14 +- 1.01 | 65.69 +- 69.28 | 1000.00 +- 0.00 |
| PPOLag | 30 | -0.16 +- 1.15 | 58.08 +- 60.34 | 1000.00 +- 0.00 |

### SafetyHalfCheetahVelocity-v1
| Algo | Seeds | Return mean±std | Cost mean±std | EpLen mean±std |
|---|---:|---:|---:|---:|
| CPO | 30 | -622.28 +- 51.90 | 0.00 +- 0.00 | 1000.00 +- 0.00 |
| P3O | 30 | -659.60 +- 54.03 | 0.00 +- 0.00 | 1000.00 +- 0.00 |
| PPOLag | 30 | -647.55 +- 48.17 | 0.01 +- 0.06 | 1000.00 +- 0.00 |

