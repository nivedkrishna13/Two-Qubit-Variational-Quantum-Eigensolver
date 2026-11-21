import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# NumPy: exact reference solution.
identity = np.eye(2, dtype=complex)
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

hamiltonian_numpy = (
    np.kron(pauli_z, identity)
    + 0.5 * np.kron(identity, pauli_z)
    + 0.7 * np.kron(pauli_x, pauli_x)
)

exact_energies, exact_states = np.linalg.eigh(
    hamiltonian_numpy
)

ground_energy = float(exact_energies[0])
ground_state = exact_states[:, 0]


# PennyLane: Hamiltonian and trial circuit.
hamiltonian_pennylane = qml.Hamiltonian(
    [1.0, 0.5, 0.7],
    [
        qml.PauliZ(wires=0),
        qml.PauliZ(wires=1),
        qml.PauliX(wires=0) @ qml.PauliX(wires=1),
    ],
)

device = qml.device(
    "default.qubit",
    wires=2,
    shots=None,
)


def ansatz(theta):
    qml.RY(theta, wires=0)
    qml.CNOT(wires=[0, 1])


@qml.qnode(
    device,
    interface="autograd",
    diff_method="parameter-shift",
)
def energy(theta):
    ansatz(theta)
    return qml.expval(hamiltonian_pennylane)


@qml.qnode(device)
def prepared_state(theta):
    ansatz(theta)
    return qml.state()


# Trainable starting angle and classical optimizer.
theta = pnp.array(0.37, requires_grad=True)

optimizer = qml.GradientDescentOptimizer(
    stepsize=0.2
)

number_of_steps = 100

initial_energy = float(energy(theta))
energy_history = [initial_energy]

print("Step       Angle (rad)       Energy")
print(
    f"{0:>4}       "
    f"{float(theta): .8f}       "
    f"{initial_energy: .10f}"
)

for step in range(1, number_of_steps + 1):
    theta = optimizer.step(energy, theta)

    current_energy = float(energy(theta))
    energy_history.append(current_energy)

    if step % 10 == 0:
        print(
            f"{step:>4}       "
            f"{float(theta): .8f}       "
            f"{current_energy: .10f}"
        )


# Evaluate the final result.
final_energy = float(energy(theta))

final_state = np.asarray(
    prepared_state(float(theta)),
    dtype=complex,
)

energy_error = abs(final_energy - ground_energy)

fidelity = float(
    abs(np.vdot(ground_state, final_state)) ** 2
)

print(f"\nInitial energy: {initial_energy:.10f}")
print(f"Final VQE energy: {final_energy:.10f}")
print(f"Exact ground energy: {ground_energy:.10f}")
print(f"Absolute energy error: {energy_error:.3e}")
print(f"Ground-state fidelity: {fidelity:.10f}")
print(f"Final angle: {float(theta):.10f} rad")

print(
    "\nEnergy decreased:",
    final_energy < initial_energy,
)

print(
    "Matches exact ground energy:",
    np.isclose(
        final_energy,
        ground_energy,
        atol=1e-8,
        rtol=0.0,
    ),
)

print(
    "Matches ground state up to global phase:",
    np.isclose(
        fidelity,
        1.0,
        atol=1e-8,
        rtol=0.0,
    ),
)

print(
    "All recorded energies respect variational bound:",
    np.all(
        np.asarray(energy_history)
        >= ground_energy - 1e-10
    ),
)