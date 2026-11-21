import numpy as np
import pennylane as qml


# Single-qubit basis states.
ket0 = np.array([1.0, 0.0], dtype=complex)
ket1 = np.array([0.0, 1.0], dtype=complex)

# NumPy: build the four two-qubit basis states.
basis_states = {
    "00": np.kron(ket0, ket0),
    "01": np.kron(ket0, ket1),
    "10": np.kron(ket1, ket0),
    "11": np.kron(ket1, ket1),
}

for label, state in basis_states.items():
    print(f"|{label}>:", state)


# NumPy: prepare |+> on wire 0 and |0> on wire 1.
plus = (ket0 + ket1) / np.sqrt(2)
state_numpy = np.kron(plus, ket0)


# PennyLane: prepare the same two-qubit state.
device = qml.device("default.qubit", wires=2, shots=None)


@qml.qnode(device)
def prepare_plus_zero():
    qml.Hadamard(wires=0)
    return qml.state()


state_pennylane = prepare_plus_zero()

probabilities_numpy = np.abs(state_numpy) ** 2
probabilities_pennylane = np.abs(state_pennylane) ** 2

print("\nNumPy |+0>:", state_numpy)
print("PennyLane |+0>:", state_pennylane)

print("\nOutcome probabilities:")
for label, p_numpy, p_pennylane in zip(
    basis_states,
    probabilities_numpy,
    probabilities_pennylane,
):
    print(
        f"{label}: NumPy={p_numpy:.3f}, "
        f"PennyLane={p_pennylane:.3f}"
    )

print("\nStates agree:", np.allclose(state_numpy, state_pennylane))
print(
    "Normalized:",
    np.isclose(probabilities_numpy.sum(), 1.0),
)