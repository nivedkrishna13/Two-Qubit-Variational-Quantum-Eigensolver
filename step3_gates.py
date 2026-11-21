import numpy as np
import pennylane as qml


# NumPy: initial state |00>.
ket0 = np.array([1.0, 0.0], dtype=complex)
initial_state = np.kron(ket0, ket0)

# Single-qubit gates.
identity = np.eye(2, dtype=complex)
hadamard = np.array(
    [[1, 1],
     [1, -1]],
    dtype=complex,
) / np.sqrt(2)

# Two-qubit gates, using basis order 00, 01, 10, 11.
hadamard_wire0 = np.kron(hadamard, identity)

cnot = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
    dtype=complex,
)

after_h_numpy = hadamard_wire0 @ initial_state
bell_numpy = cnot @ after_h_numpy


# PennyLane: reproduce both stages.
device = qml.device("default.qubit", wires=2, shots=None)


@qml.qnode(device)
def prepare_plus_zero():
    qml.Hadamard(wires=0)
    return qml.state()


@qml.qnode(device)
def prepare_bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()


after_h_pennylane = prepare_plus_zero()
bell_pennylane = prepare_bell()

print("After Hadamard:")
print("NumPy:    ", after_h_numpy)
print("PennyLane:", after_h_pennylane)
print("States agree:", np.allclose(after_h_numpy, after_h_pennylane))

print("\nAfter CNOT (Bell state):")
print("NumPy:    ", bell_numpy)
print("PennyLane:", bell_pennylane)
print("States agree:", np.allclose(bell_numpy, bell_pennylane))

probabilities_numpy = np.abs(bell_numpy) ** 2
probabilities_pennylane = np.abs(bell_pennylane) ** 2

print("\nOutcome probabilities:")
for label, p_numpy, p_pennylane in zip(
    ["00", "01", "10", "11"],
    probabilities_numpy,
    probabilities_pennylane,
):
    print(
        f"{label}: NumPy={p_numpy:.3f}, "
        f"PennyLane={p_pennylane:.3f}"
    )

expected_bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

print("\nMatches expected Bell state:", np.allclose(bell_numpy, expected_bell))
print("Normalized:", np.isclose(probabilities_numpy.sum(), 1.0))
print("CNOT twice restores input:", np.allclose(cnot @ bell_numpy, after_h_numpy))