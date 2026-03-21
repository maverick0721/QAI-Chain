import torch.nn as nn

from quantum.models.quantum_layer import QuantumLayer


class QNN(nn.Module):

    def __init__(self, input_dim=5, n_qubits=4):
        super().__init__()

        self.fc = nn.Linear(input_dim, n_qubits)

        self.q_layer = QuantumLayer(n_qubits)

        self.output = nn.Linear(n_qubits, 1)

    def forward(self, x):

        x = self.fc(x)

        x = self.q_layer(x)

        x = self.output(x)

        return x