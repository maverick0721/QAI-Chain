import pennylane as qml
import torch

from quantum.devices.device import get_device

n_qubits = 4
dev = get_device(n_qubits)


def _fit_features_to_qubits(x, qubits):

    # Ensure a 1D feature vector with exactly `qubits` entries for AngleEmbedding.
    x = torch.flatten(x)

    if x.shape[0] > qubits:
        return x[:qubits]

    if x.shape[0] < qubits:
        pad = torch.zeros(qubits - x.shape[0], dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=0)

    return x


@qml.qnode(dev, interface="torch")
def kernel_circuit(x1, x2):

    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))

    return qml.probs(wires=range(n_qubits))


def quantum_kernel(x1, x2):

    x1 = _fit_features_to_qubits(x1, n_qubits)
    x2 = _fit_features_to_qubits(x2, n_qubits)

    probs = kernel_circuit(x1, x2)

    # similarity score = probability of |000...>
    return probs[0]