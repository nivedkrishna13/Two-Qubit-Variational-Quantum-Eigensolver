import numpy as np
import pennylane as qml


# NumPy: construct our Hamiltonian.
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

hamiltonian_numpy = (
    np.kron(pauli_z, identity)
    + 0.5 * np.kron(identity, pauli_z)
    + 0.7 * np.kron(pauli_x, pauli_x)
)


# NumPy: solve the full matrix.
energies, eigenvectors = np.linalg.eigh(hamiltonian_numpy)

ground_energy = energies[0]
ground_state = eigenvectors[:, 0]


# PennyLane: construct the same Hamiltonian and get its eigenvalues.
hamiltonian_pennylane = qml.Hamiltonian(
    [1.0, 0.5, 0.7],
    [
        qml.PauliZ(wires=0),
        qml.PauliZ(wires=1),
        qml.PauliX(wires=0) @ qml.PauliX(wires=1),
    ],
)

energies_pennylane = np.sort(
    np.asarray(qml.eigvals(hamiltonian_pennylane))
)


# Verify the defining eigenvector equation: H|psi> = E|psi>.
left_side = hamiltonian_numpy @ ground_state
right_side = ground_energy * ground_state

# The residual measures the mismatch between the two sides.
residual = np.linalg.norm(left_side - right_side)

probabilities = np.abs(ground_state) ** 2


print("NumPy energies:", energies)
print("PennyLane energies:", energies_pennylane)

print(
    "Energy lists agree:",
    np.allclose(energies, energies_pennylane),
)

print(f"\nGround-state energy: {ground_energy:.10f}")
print("Ground-state vector:", ground_state)

print("\nGround-state basis probabilities:")
for label, probability in zip(
    ["00", "01", "10", "11"],
    probabilities,
):
    print(f"{label}: {probability:.6f}")

print(
    "\nNormalized:",
    np.isclose(probabilities.sum(), 1.0),
)

print(
    "Eigenvector equation satisfied:",
    np.allclose(left_side, right_side),
)

print(f"Eigenvector residual: {residual:.3e}")