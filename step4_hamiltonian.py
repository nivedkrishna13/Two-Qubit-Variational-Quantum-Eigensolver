import numpy as np
import pennylane as qml


# NumPy: define the single-qubit matrices.
identity = np.eye(2, dtype=complex)

pauli_x = np.array(
    [[0, 1],
     [1, 0]],
    dtype=complex,
)

pauli_z = np.array(
    [[1, 0],
     [0, -1]],
    dtype=complex,
)

# Build each two-qubit term.
z0 = np.kron(pauli_z, identity)
z1 = np.kron(identity, pauli_z)
xx = np.kron(pauli_x, pauli_x)

# Add the weighted terms.
hamiltonian_numpy = z0 + 0.5 * z1 + 0.7 * xx


# PennyLane: describe the same Hamiltonian.
coefficients = [1.0, 0.5, 0.7]

operators = [
    qml.PauliZ(wires=0),
    qml.PauliZ(wires=1),
    qml.PauliX(wires=0) @ qml.PauliX(wires=1),
]

hamiltonian_pennylane = qml.Hamiltonian(coefficients, operators)

matrix_pennylane = qml.matrix(
    hamiltonian_pennylane,
    wire_order=[0, 1],
)


# Independently write the matrix we derived by hand.
expected_matrix = np.array(
    [[1.5, 0, 0, 0.7],
     [0, 0.5, 0.7, 0],
     [0, 0.7, -0.5, 0],
     [0.7, 0, 0, -1.5]],
    dtype=complex,
)

print("NumPy Hamiltonian:")
print(hamiltonian_numpy)

print("\nPennyLane Hamiltonian matrix:")
print(matrix_pennylane)

print("\nMatrix shape:", hamiltonian_numpy.shape)

print(
    "NumPy and PennyLane agree:",
    np.allclose(hamiltonian_numpy, matrix_pennylane),
)

print(
    "Matches hand-derived matrix:",
    np.allclose(hamiltonian_numpy, expected_matrix),
)

print(
    "Hermitian:",
    np.allclose(
        hamiltonian_numpy,
        hamiltonian_numpy.conj().T,
    ),
)