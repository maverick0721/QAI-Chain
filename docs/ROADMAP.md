# QAI-Chain Roadmap

## Phase 1: Reliability Foundation (Now)

- [x] Fix critical import/runtime failures
- [x] Convert script checks into executable pytest tests
- [x] Add CI and healthcheck automation
- [x] Add quick benchmark utility

## Phase 2: Engineering Depth

- [x] Add integration tests for RPC + mempool + mining pipeline
- [x] Add property-style tests for transaction and block invariants
- [x] Add static type checks and stricter linting gate
- [x] Add seed-controlled reproducibility harness for RL experiments

## Phase 3: Research Credibility

- [x] Define one flagship hypothesis (throughput/latency/security metric impact)
- [x] Implement baseline comparisons and ablations
- [x] Publish benchmark table with confidence intervals
- [x] Add experiment tracking and artifact versioning

## Phase 4: Production Signaling

- [x] Add API schema docs and compatibility policy
- [x] Add threat model and security review notes
- [x] Add deployment template (single-node + local testnet)
- [x] Add observability dashboard hooks (logs/metrics/traces)

## One-Glance Recruiter Checklist

- clear technical thesis
- passing CI
- measurable benchmark outputs
- reproducible experiments
- explicit limitations and future work