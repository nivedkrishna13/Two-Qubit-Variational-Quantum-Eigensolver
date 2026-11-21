import numpy as np
import pennylane as qml


# NumPy: construct the Hamiltonian.
identity = np.eye(2, dtype=complex)
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

hamiltonian_numpy = (
    np.kron(pauli_z, identity)
    + 0.5 * np.kron(identity, pauli_z)
    + 0.7 * np.kron(pauli_x, pauli_x)
)

# PennyLane: construct the same Hamiltonian.
hamiltonian_pennylane = qml.Hamiltonian(
    [1.0, 0.5, 0.7],
    [
        qml.PauliZ(wires=0),
        qml.PauliZ(wires=1),
        qml.PauliX(wires=0) @ qml.PauliX(wires=1),
    ],
)


# Prepare familiar states directly as NumPy vectors.
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
plus = (ket0 + ket1) / np.sqrt(2)

energies, eigenvectors = np.linalg.eigh(hamiltonian_numpy)
ground_energy = energies[0]

states = {
    "00": np.kron(ket0, ket0),
    "+0": np.kron(plus, ket0),
    "Bell": (
        np.kron(ket0, ket0) + np.kron(ket1, ket1)
    ) / np.sqrt(2),
    "Ground": eigenvectors[:, 0],
}


# PennyLane: load a given state and calculate its expected energy.
device = qml.device("default.qubit", wires=2, shots=None)


@qml.qnode(device)
def expected_energy_pennylane(state):
    qml.StatePrep(state, wires=[0, 1])
    return qml.expval(hamiltonian_pennylane)


expected_values = {
    "00": 1.5,
    "+0": 0.5,
    "Bell": 0.7,
    "Ground": ground_energy,
}

numpy_results = []
pennylane_results = []

print("State       NumPy energy    PennyLane energy")

for label, state in states.items():
    energy_complex = np.vdot(
        state,
        hamiltonian_numpy @ state,
    )

    # A Hermitian operator should have a real expectation value.
    if not np.isclose(energy_complex.imag, 0.0, atol=1e-12):
        raise ValueError("Unexpected imaginary energy.")

    energy_numpy = float(energy_complex.real)
    energy_pennylane = float(expected_energy_pennylane(state))

    numpy_results.append(energy_numpy)
    pennylane_results.append(energy_pennylane)

    print(
        f"{label:<10} "
        f"{energy_numpy: .10f}    "
        f"{energy_pennylane: .10f}"
    )

print(
    "\nNumPy and PennyLane agree:",
    np.allclose(numpy_results, pennylane_results),
)

print(
    "Match reference values:",
    np.allclose(
        numpy_results,
        list(expected_values.values()),
    ),
)