from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


class VQCPolicyNetwork(nn.Module):
    """6-qubit variational policy network for governance control.

    Structure:
    - AngleEmbedding on 6 qubits
    - BasicEntanglerLayers (layers 1-2)
    - Data re-uploading (AngleEmbedding)
    - BasicEntanglerLayers (layers 3-4)
    - expval(Z0) -> mean, expval(Z1) -> log_std proxy
    """

    def __init__(self, n_qubits: int = 6, n_layers: int = 4):
        super().__init__()
        if n_qubits != 6:
            raise ValueError("VQCPolicyNetwork is configured for 6 qubits.")
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2 to support data re-uploading.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        self.weights = nn.Parameter(0.02 * torch.randn(n_layers, n_qubits))
        self.mean_scale = nn.Parameter(torch.tensor(1.0))
        self.mean_bias = nn.Parameter(torch.tensor(0.0))
        self.log_std_scale = nn.Parameter(torch.tensor(0.5))
        self.log_std_bias = nn.Parameter(torch.tensor(-0.7))

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(x: torch.Tensor, w: torch.Tensor):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(w[:2], wires=range(n_qubits), rotation=qml.RY)
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(w[2:], wires=range(n_qubits), rotation=qml.RY)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        self._circuit = circuit

    def _forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.n_qubits:
            raise ValueError(f"Expected input dim {self.n_qubits}, got {x.shape[-1]}.")
        z0, z1 = self._circuit(x, self.weights)
        mean = self.mean_scale * z0 + self.mean_bias
        log_std = self.log_std_scale * z1 + self.log_std_bias
        std = torch.exp(torch.clamp(log_std, min=-4.0, max=2.0))
        return mean.unsqueeze(0), std.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        outputs = [self._forward_single(row) for row in x]
        means = torch.stack([m.squeeze(0) for m, _ in outputs]).unsqueeze(-1)
        stds = torch.stack([s.squeeze(0) for _, s in outputs]).unsqueeze(-1)
        return means, stds


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
