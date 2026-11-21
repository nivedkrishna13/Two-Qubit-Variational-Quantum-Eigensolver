import numpy as np
import pennylane as qml


# Fix the angle found by VQE.
theta = -2.7049654938

exact_ground_energy = -np.sqrt(1.5**2 + 0.7**2)

# One device for each measurement setting.
# Each circuit call uses 1000 shots.
shots = 1000

device_z = qml.device(
    "default.qubit",
    wires=2,
    shots=shots,
    seed=42,
)

device_x = qml.device(
    "default.qubit",
    wires=2,
    shots=shots,
    seed=123,
)


def ansatz(angle):
    qml.RY(angle, wires=0)
    qml.CNOT(wires=[0, 1])


@qml.qnode(device_z)
def sample_z(angle):
    ansatz(angle)
    return qml.sample(wires=[0, 1])


@qml.qnode(device_x)
def sample_x(angle):
    ansatz(angle)

    # Rotate X measurements into computational-basis measurements.
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    return qml.sample(wires=[0, 1])


def estimate_energy(angle):
    bits_z = np.asarray(sample_z(angle))
    bits_x = np.asarray(sample_x(angle))

    # Convert measurement bits to Pauli eigenvalues:
    # bit 0 -> +1, bit 1 -> -1.
    signs_z = 1 - 2 * bits_z
    signs_x = 1 - 2 * bits_x

    z0_estimate = np.mean(signs_z[:, 0])
    z1_estimate = np.mean(signs_z[:, 1])

    # Multiply signs from the SAME shot.
    xx_products = signs_x[:, 0] * signs_x[:, 1]
    xx_estimate = np.mean(xx_products)

    energy_estimate = (
        z0_estimate
        + 0.5 * z1_estimate
        + 0.7 * xx_estimate
    )

    # Z0 and Z1 share shots, so combine their contributions
    # before calculating the variance.
    z_contributions = (
        signs_z[:, 0] + 0.5 * signs_z[:, 1]
    )

    # The two measurement settings use independent shot batches.
    standard_error = np.sqrt(
        np.var(z_contributions, ddof=1) / shots
        + 0.7**2 * np.var(xx_products, ddof=1) / shots
    )

    return (
        z0_estimate,
        z1_estimate,
        xx_estimate,
        energy_estimate,
        standard_error,
    )


print(f"Exact ground energy: {exact_ground_energy:.10f}")
print(f"Shots per setting: {shots}")
print(f"Total shots per energy estimate: {2 * shots}")

print(
    "\nRun      <Z0>      <Z1>      <XX>"
    "       Energy       Std. error"
)

estimates = []

for run in range(1, 11):
    z0, z1, xx, energy, error = estimate_energy(theta)
    estimates.append(energy)

    print(
        f"{run:>3}    "
        f"{z0: .4f}    "
        f"{z1: .4f}    "
        f"{xx: .4f}    "
        f"{energy: .6f}    "
        f"{error:.6f}"
    )

print(
    f"\nMean of 10 energy estimates: "
    f"{np.mean(estimates):.6f}"
)

print(
    "Estimates below exact ground energy:",
    np.count_nonzero(
        np.asarray(estimates) < exact_ground_energy
    ),
)