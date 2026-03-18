import pennylane as qml
import torch

from quantum.devices.device import get_device
from quantum.encodings.angle_encoding import angle_encoding
from quantum.circuits.variational_circuit import variational_layer


class QuantumCircuit:

    def __init__(self, n_qubits=4):

        self.n_qubits = n_qubits
        self.dev = get_device(n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, weights):

            angle_encoding(x, wires=range(n_qubits))

            variational_layer(weights, wires=range(n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x, weights):

        return self.circuit(x, weights)