import numpy as np
import pennylane as qml


# NumPy: represent the states directly.
ket0 = np.array([1.0, 0.0], dtype=complex)
ket1 = np.array([0.0, 1.0], dtype=complex)
plus_numpy = (ket0 + ket1) / np.sqrt(2)


# PennyLane: prepare the same state with a circuit.
device = qml.device("default.qubit", wires=1, shots=None)


@qml.qnode(device)
def prepare_plus():
    qml.Hadamard(wires=0)
    return qml.state()


plus_pennylane = prepare_plus()


# Convert amplitudes into measurement probabilities.
probabilities = np.abs(plus_numpy) ** 2

print("NumPy |0>:", ket0)
print("NumPy |1>:", ket1)
print("NumPy |+>:", plus_numpy)
print("PennyLane |+>:", plus_pennylane)
print("Measurement probabilities:", probabilities)
print("Sum of probabilities:", probabilities.sum())

print(
    "States agree:",
    np.allclose(plus_numpy, plus_pennylane),
)