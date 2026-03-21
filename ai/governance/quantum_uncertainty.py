from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class UncertaintyResult:
    mean_action: float
    variance: float
    fallback: bool


def estimate_quantum_uncertainty(
    policy: torch.nn.Module,
    state: torch.Tensor,
    k: int = 10,
    noise_std: float = 0.01,
    tau_q: float = 0.02,
) -> UncertaintyResult:
    """MC-style uncertainty via parameter perturbation over quantum policy weights."""
    if k < 2:
        raise ValueError("k must be >= 2")

    if state.ndim == 1:
        state = state.unsqueeze(0)

    named_params = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    backups = {n: p.detach().clone() for n, p in named_params}

    preds: list[float] = []
    with torch.no_grad():
        for _ in range(k):
            for _, p in named_params:
                p.add_(noise_std * torch.randn_like(p))

            mean, _ = policy(state)
            preds.append(float(mean.squeeze().item()))

            for n, p in named_params:
                p.copy_(backups[n])

    pred_t = torch.tensor(preds, dtype=torch.float32)
    variance = float(torch.var(pred_t, unbiased=True).item())
    mean_action = float(torch.mean(pred_t).item())
    return UncertaintyResult(mean_action=mean_action, variance=variance, fallback=variance > tau_q)
