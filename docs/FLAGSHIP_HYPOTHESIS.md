# Flagship Hypothesis

## Hypothesis

Under non-stationary and adversarial governance dynamics, robust PPO with uncertainty-gated fallback achieves higher tail-resilience than vanilla PPO, heuristic, and random baselines while maintaining comparable nominal reward.

## Primary Metrics

- Mean reward (last-k episodes)
- Tail-risk proxy under stress sweep
- Constraint violation/error-to-target under regime shifts

## Test Protocol

- Paired-seed evaluation across baseline and robust variants
- Multi-seed confidence intervals and permutation testing
- Separate transfer validation on a second governance environment
