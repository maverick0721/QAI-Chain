import torch
import torch.nn as nn
import warnings

warnings.filterwarnings(
    "ignore",
    message=(
        "Converting a tensor with requires_grad=True to a scalar may lead to "
        "unexpected behavior.*"
    ),
)

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

            # PennyLane can return a tensor, tuple, or list depending on the qnode output.
            if isinstance(out, torch.Tensor):
                out_tensor = out
            elif isinstance(out, (list, tuple)):
                values = []
                for value in out:
                    values.append(value if isinstance(value, torch.Tensor) else torch.tensor(value))
                out_tensor = torch.stack(values)
            else:
                out_tensor = torch.tensor(out)

            outputs.append(out_tensor.to(dtype=torch.float32, device=x.device))

        return torch.stack(outputs)