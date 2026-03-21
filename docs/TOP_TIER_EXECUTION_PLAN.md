# Top-Tier Upgrade Execution Plan

This file maps the requested roadmap to implemented code and the remaining long-run experiment execution.

## Phase 1: Delete and Rewrite

Completed:
- Removed the old runtime-latency subsection centered on QNN/QTransformer from paper results.
- Removed implementation-details section from the compiled manuscript path.
- Rewrote Introduction and Discussion in direct, claim-first style.
- Rewrote Ethics to one paragraph.
- Trimmed appendix artifact-list filler to a single availability statement.
- Replaced old quantum-module prose with VQC policy + quantum uncertainty gating methodology.

## Phase 2: New Components Implemented

Implemented modules:
- `ai/models/vqc_policy.py`
  - 6-qubit VQC policy with AngleEmbedding, BasicEntanglerLayers, data re-uploading, expval(Z0/Z1) outputs.
- `ai/governance/quantum_uncertainty.py`
  - K-pass parameter perturbation uncertainty estimator.
- `ai/governance/safety_shield.py`
  - Hard bounds, rate limiting, anomaly-based reject/clamp/accept decisions.
- `core/blockchain/audit_trail.py`
  - Audit record generation and hashing.
- `core/blockchain/policy_audit.py`
  - Policy hash and epoch verification utility.
- `core/blockchain/blockchain.py`
  - On-chain audit trail storage via `commit_audit_record`.
- `crypto/pqc/keypair.py`, `crypto/pqc/signature.py`
  - Dilithium-3 path via `pqcrypto` with deterministic fallback.
- `ai/rl/scaled_environment.py`
  - 12D state, 3D action, 200-step episodes, partial observability, adversarial scenarios.
- `ai/rl/defi_environment.py`
  - Second independent governance environment (DeFi liquidity).

## Phase 3: Experiment Runners Implemented

- `experiments/run_full_ablation_matrix.py`
- `experiments/run_stress_sweep_scaled.py`
- `experiments/run_quantum_uncertainty_comparison.py`
- `experiments/run_parameter_efficiency.py`
- `experiments/run_dual_environment_transfer.py`

Generated artifacts (smoke-tested):
- `experiments/results/full_ablation_matrix.json`
- `experiments/results/stress_sweep_scaled.json`
- `experiments/results/quantum_uncertainty_comparison.json`
- `experiments/results/parameter_efficiency.json`
- `experiments/results/dual_environment_transfer.json`

## Phase 4: Paper Upgrades

Completed now:
- New title and abstract structure.
- Updated method/system/theory framing around quantum-gated safe governance.
- Added theorem-style statements for safety and variance claims.

Completed for venue-ready draft:
- Rebuilt tables/figures from the new experiment suite and replaced old sections.
- Reduced manuscript to target page budget with figure consolidation.
- Added baseline and constrained-comparison framing in method/results artifacts.

## Full-Scale Execution Commands

Long-run experiments (recommended overnight runs):

```bash
cd /home/ubuntu/QAI-Chain
PYTHONPATH=. .venv/bin/python experiments/run_full_ablation_matrix.py --seeds 3,7,11,17,21,42,84,126,5,9,13,19,23,31,57,99 --episodes 100 --steps 200
PYTHONPATH=. .venv/bin/python experiments/run_stress_sweep_scaled.py --episodes 50 --steps 200
PYTHONPATH=. .venv/bin/python experiments/run_quantum_uncertainty_comparison.py
PYTHONPATH=. .venv/bin/python experiments/run_parameter_efficiency.py
PYTHONPATH=. .venv/bin/python experiments/run_dual_environment_transfer.py
```

Manuscript build:

```bash
cd /home/ubuntu/QAI-Chain/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
