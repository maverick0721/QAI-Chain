import pennylane as qml
import torch

from quantum.devices.device import get_device

n_qubits = 4
dev = get_device(n_qubits)


@qml.qnode(dev, interface="torch")
def kernel_circuit(x1, x2):

    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))

    return qml.probs(wires=range(n_qubits))


def quantum_kernel(x1, x2):

    probs = kernel_circuit(x1, x2)

    # similarity score = probability of |000...>
    return probs[0]