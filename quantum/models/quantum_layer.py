import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.*")

from quantum.models.qnode import QuantumCircuit


class QuantumLayer(nn.Module):

    def __init__(self, n_qubits=4):
        super().__init__()

        self.n_qubits = n_qubits

        self.qc = QuantumCircuit(n_qubits)

        # trainable parameters
        self.weights = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x):

        outputs = []

        for i in range(x.shape[0]):

            out = self.qc.forward(x[i], self.weights)

            # Robustly convert QNode output to float or numpy before as_tensor
            import numpy as np
            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
            if isinstance(out, (np.generic, np.ndarray)) and out.shape == ():
                out = out.item()
            outputs.append(torch.as_tensor(out, dtype=x.dtype))

        return torch.stack(outputs)