import numpy as np
import pennylane as qml


theta = -2.7049654938

exact_ground_energy = -np.sqrt(1.5**2 + 0.7**2)

# Analytical expectation at the actual angle being measured.
true_energy = (
    1.5 * np.cos(theta)
    + 0.7 * np.sin(theta)
)

shot_counts = [100, 400, 1600, 6400]
repetitions = 100


def ansatz(angle):
    qml.RY(angle, wires=0)
    qml.CNOT(wires=[0, 1])


def make_measurement_circuits(shots):
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

    @qml.qnode(device_z)
    def sample_z(angle):
        ansatz(angle)
        return qml.sample(wires=[0, 1])

    @qml.qnode(device_x)
    def sample_x(angle):
        ansatz(angle)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        return qml.sample(wires=[0, 1])

    return sample_z, sample_x


def estimate_energy(sample_z, sample_x):
    signs_z = 1 - 2 * np.asarray(sample_z(theta))
    signs_x = 1 - 2 * np.asarray(sample_x(theta))

    z_contributions = (
        signs_z[:, 0] + 0.5 * signs_z[:, 1]
    )

    xx_products = signs_x[:, 0] * signs_x[:, 1]

    return float(
        np.mean(z_contributions)
        + 0.7 * np.mean(xx_products)
    )


print(f"Exact ground energy: {exact_ground_energy:.10f}")
print(f"Repetitions per shot count: {repetitions}")

print(
    "\nShots/setting    Mean energy    Bias estimate"
    "    Observed SD    Predicted SD"
)

observed_spreads = []

for shots in shot_counts:
    sample_z, sample_x = make_measurement_circuits(shots)

    # Reuse the devices so their random generators advance.
    estimates = np.array([
        estimate_energy(sample_z, sample_x)
        for _ in range(repetitions)
    ])

    mean_energy = np.mean(estimates)
    bias_estimate = mean_energy - true_energy

    observed_sd = np.std(estimates, ddof=1)

    # For our ansatz:
    # Var(Z contribution) = 1.5^2 * sin(theta)^2
    # Var(0.7 * XX) = 0.7^2 * cos(theta)^2
    #
    # Independent batches make their variances add.
    predicted_sd = np.sqrt(
        (
            1.5**2 * np.sin(theta)**2
            + 0.7**2 * np.cos(theta)**2
        ) / shots
    )

    observed_spreads.append(observed_sd)

    print(
        f"{shots:>13}    "
        f"{mean_energy: .6f}    "
        f"{bias_estimate: .6f}    "
        f"{observed_sd:.6f}       "
        f"{predicted_sd:.6f}"
    )

print("\nObserved spread ratios (previous / next):")

for index in range(1, len(shot_counts)):
    ratio = (
        observed_spreads[index - 1]
        / observed_spreads[index]
    )

    print(
        f"{shot_counts[index - 1]} -> "
        f"{shot_counts[index]} shots: "
        f"{ratio:.3f} (expected about 2)"
    )