import numpy as np
import pennylane as qml


# NumPy Hamiltonian.
identity = np.eye(2, dtype=complex)
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

hamiltonian_numpy = (
    np.kron(pauli_z, identity)
    + 0.5 * np.kron(identity, pauli_z)
    + 0.7 * np.kron(pauli_x, pauli_x)
)

# PennyLane Hamiltonian.
hamiltonian_pennylane = qml.Hamiltonian(
    [1.0, 0.5, 0.7],
    [
        qml.PauliZ(wires=0),
        qml.PauliZ(wires=1),
        qml.PauliX(wires=0) @ qml.PauliX(wires=1),
    ],
)

initial_state = np.array([1, 0, 0, 0], dtype=complex)

cnot = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
    dtype=complex,
)


def ansatz_numpy(theta):
    cosine = np.cos(theta / 2)
    sine = np.sin(theta / 2)

    rotation_y = np.array(
        [[cosine, -sine],
         [sine, cosine]],
        dtype=complex,
    )

    rotation_wire0 = np.kron(rotation_y, identity)

    return cnot @ rotation_wire0 @ initial_state


device = qml.device("default.qubit", wires=2, shots=None)


def ansatz(theta):
    qml.RY(theta, wires=0)
    qml.CNOT(wires=[0, 1])


@qml.qnode(device)
def state_pennylane(theta):
    ansatz(theta)
    return qml.state()


@qml.qnode(device)
def energy_pennylane(theta):
    ansatz(theta)
    return qml.expval(hamiltonian_pennylane)


angles = [
    ("0", 0.0),
    ("pi/2", np.pi / 2),
    ("pi", np.pi),
    ("3pi/2", 3 * np.pi / 2),
    ("0.37", 0.37),
]

states_agree = True
energies_agree = True
normalized = True

print("Angle       NumPy energy    PennyLane energy    Formula")

for label, theta in angles:
    state_numpy = ansatz_numpy(theta)
    state_pl = state_pennylane(theta)

    energy_numpy = float(
        np.vdot(
            state_numpy,
            hamiltonian_numpy @ state_numpy,
        ).real
    )

    energy_pl = float(energy_pennylane(theta))

    energy_formula = (
        1.5 * np.cos(theta)
        + 0.7 * np.sin(theta)
    )

    states_agree = states_agree and np.allclose(
        state_numpy,
        state_pl,
    )

    energies_agree = energies_agree and np.allclose(
        [energy_numpy, energy_pl],
        energy_formula,
    )

    normalized = normalized and np.isclose(
        np.vdot(state_numpy, state_numpy),
        1.0,
    )

    print(
        f"{label:<10} "
        f"{energy_numpy: .10f}    "
        f"{energy_pl: .10f}    "
        f"{energy_formula: .10f}"
    )

print("\nNumPy and PennyLane states agree:", states_agree)
print("All three energy calculations agree:", energies_agree)
print("All tested states normalized:", normalized)

print("\nCircuit:")
print(qml.draw(energy_pennylane)(0.37))