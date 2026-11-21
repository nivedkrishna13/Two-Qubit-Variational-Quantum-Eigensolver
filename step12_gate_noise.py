import numpy as np
import pennylane as qml


# Exact reference Hamiltonian.
identity = np.eye(2, dtype=complex)
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

hamiltonian = (
    np.kron(pauli_z, identity)
    + 0.5 * np.kron(identity, pauli_z)
    + 0.7 * np.kron(pauli_x, pauli_x)
)

eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

ground_energy = float(eigenvalues[0])
ground_state = eigenvectors[:, 0]

# Keep the ideal optimized angle fixed.
theta = -2.7049654938

noise_probabilities = [0.0, 0.001, 0.01, 0.03, 0.05, 0.1]

device = qml.device(
    "default.mixed",
    wires=2,
    shots=None,
)


@qml.qnode(device)
def noisy_state(angle, p):
    qml.RY(angle, wires=0)

    # Model an error after the single-qubit gate.
    qml.DepolarizingChannel(p, wires=0)

    qml.CNOT(wires=[0, 1])

    # Model independent errors after the two-qubit gate.
    qml.DepolarizingChannel(p, wires=0)
    qml.DepolarizingChannel(p, wires=1)

    return qml.density_matrix(wires=[0, 1])


print(f"Exact ground energy: {ground_energy:.10f}")
print(f"Fixed angle: {theta:.10f} rad")
print("Shots: None (exact noisy-state expectations)")

print(
    "\nNoise p    True energy     Energy error"
    "    Fidelity      Purity"
)

energies = []
valid_states = []

for p in noise_probabilities:
    rho = np.asarray(
        noisy_state(theta, p),
        dtype=complex,
    )

    energy = float(
        np.trace(rho @ hamiltonian).real
    )

    # Fidelity with the pure reference ground state.
    fidelity = float(
        np.vdot(ground_state, rho @ ground_state).real
    )

    purity = float(
        np.trace(rho @ rho).real
    )

    energy_error = energy - ground_energy

    energies.append(energy)

    # A physical density matrix is Hermitian,
    # has trace 1, and has nonnegative eigenvalues.
    valid_states.append(
        np.allclose(rho, rho.conj().T)
        and np.isclose(np.trace(rho), 1.0)
        and np.min(np.linalg.eigvalsh(rho)) >= -1e-10
    )

    print(
        f"{p:>7.3f}    "
        f"{energy: .8f}    "
        f"{energy_error:.3e}    "
        f"{fidelity:.8f}    "
        f"{purity:.8f}"
    )


print(
    "\nZero-noise energy matches reference:",
    np.isclose(
        energies[0],
        ground_energy,
        atol=1e-8,
        rtol=0.0,
    ),
)

print(
    "All density matrices are valid:",
    all(valid_states),
)

print(
    "All true energies respect variational bound:",
    np.all(
        np.asarray(energies) >= ground_energy - 1e-10
    ),
)