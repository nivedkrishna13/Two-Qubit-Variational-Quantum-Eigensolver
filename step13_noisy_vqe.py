import numpy as np
import pennylane as qml


# Reference Hamiltonian.
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

ideal_theta = -2.7049654938
noise_probabilities = [0.0, 0.01, 0.03, 0.05, 0.1]

device = qml.device(
    "default.mixed",
    wires=2,
    shots=None,
)


@qml.qnode(device)
def noisy_state(theta, p):
    qml.RY(theta, wires=0)
    qml.DepolarizingChannel(p, wires=0)

    qml.CNOT(wires=[0, 1])

    qml.DepolarizingChannel(p, wires=0)
    qml.DepolarizingChannel(p, wires=1)

    return qml.density_matrix(wires=[0, 1])


def energy(theta, p):
    rho = np.asarray(noisy_state(theta, p), dtype=complex)
    return float(np.trace(rho @ hamiltonian).real)


def fidelity(theta, p):
    rho = np.asarray(noisy_state(theta, p), dtype=complex)
    return float(
        np.vdot(ground_state, rho @ ground_state).real
    )


def optimize(p):
    # Start from the angle that was optimal without noise.
    theta = ideal_theta

    for _ in range(150):
        energy_plus = energy(theta + np.pi / 2, p)
        energy_minus = energy(theta - np.pi / 2, p)

        gradient = (energy_plus - energy_minus) / 2

        theta = theta - 0.2 * gradient

    return theta


print(f"Exact ground energy: {ground_energy:.10f}")
print(f"Ideal optimized angle: {ideal_theta:.8f} rad")

print(
    "\nNoise p   Fixed energy   Reoptimized energy"
    "   Improvement   New angle"
)

fixed_energies = []
optimized_energies = []

for p in noise_probabilities:
    fixed_energy = energy(ideal_theta, p)

    optimized_theta = optimize(p)
    optimized_energy = energy(optimized_theta, p)

    improvement = fixed_energy - optimized_energy

    fixed_energies.append(fixed_energy)
    optimized_energies.append(optimized_energy)

    print(
        f"{p:>7.3f}   "
        f"{fixed_energy: .8f}   "
        f"{optimized_energy: .8f}      "
        f"{improvement:.3e}    "
        f"{optimized_theta: .6f}"
    )

    if p == noise_probabilities[-1]:
        print(
            f"\nAt p={p:.2f}:"
            f"\n  Fidelity before: {fidelity(ideal_theta, p):.8f}"
            f"\n  Fidelity after:  {fidelity(optimized_theta, p):.8f}"
            f"\n  Remaining energy error: "
            f"{optimized_energy - ground_energy:.6f}"
        )


print(
    "\nReoptimization never increased energy:",
    np.all(
        np.asarray(optimized_energies)
        <= np.asarray(fixed_energies) + 1e-10
    ),
)

print(
    "All optimized energies respect variational bound:",
    np.all(
        np.asarray(optimized_energies)
        >= ground_energy - 1e-10
    ),
)