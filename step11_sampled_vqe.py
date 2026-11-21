import numpy as np
import pennylane as qml


shots = 1000
number_of_steps = 100
initial_theta = 0.37

ground_energy = -np.sqrt(1.5**2 + 0.7**2)


def analytical_energy(theta):
    return 1.5 * np.cos(theta) + 0.7 * np.sin(theta)


def ansatz(theta):
    qml.RY(theta, wires=0)
    qml.CNOT(wires=[0, 1])


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
def sample_z(theta):
    ansatz(theta)
    return qml.sample(wires=[0, 1])


@qml.qnode(device_x)
def sample_x(theta):
    ansatz(theta)
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    return qml.sample(wires=[0, 1])


def sampled_energy(theta):
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


theta = initial_theta

print(f"Shots per measurement setting: {shots}")
print(f"Exact ground energy: {ground_energy:.10f}")

print(
    "\nStep      Angle       Sampled gradient"
    "    True energy     True error"
)

for step in range(1, number_of_steps + 1):
    # Two independently sampled energy estimates.
    energy_plus = sampled_energy(theta + np.pi / 2)
    energy_minus = sampled_energy(theta - np.pi / 2)

    gradient_estimate = (
        energy_plus - energy_minus
    ) / 2

    # Reduce the update size as optimization progresses.
    learning_rate = 0.2 / (1 + step / 50)

    theta = theta - learning_rate * gradient_estimate

    # Diagnostics only: these values do not choose the update.
    true_energy = analytical_energy(theta)
    true_error = true_energy - ground_energy

    if step == 1 or step % 10 == 0:
        print(
            f"{step:>4}    "
            f"{theta: .6f}    "
            f"{gradient_estimate: .6f}         "
            f"{true_energy: .8f}    "
            f"{true_error:.3e}"
        )


# Fresh sampled evaluation of the final angle.
final_sampled_energy = sampled_energy(theta)
final_true_energy = analytical_energy(theta)

# For our ansatz, E(theta) = R*cos(theta - phi).
# Its ground-state fidelity is (1 - E(theta)/R) / 2.
energy_scale = np.sqrt(1.5**2 + 0.7**2)
fidelity = (1 - final_true_energy / energy_scale) / 2

optimization_shots = (
    number_of_steps
    * 2       # Shifted energies per gradient.
    * 2       # Measurement settings per energy.
    * shots
)

final_evaluation_shots = 2 * shots

print(f"\nFinal angle: {theta:.8f} rad")
print(f"Final sampled energy: {final_sampled_energy:.10f}")
print(f"Final true energy: {final_true_energy:.10f}")
print(f"Exact ground energy: {ground_energy:.10f}")

print(
    f"True energy error: "
    f"{final_true_energy - ground_energy:.3e}"
)

print(f"Ground-state fidelity: {fidelity:.10f}")

print(
    "\nTrue energy improved:",
    final_true_energy < analytical_energy(initial_theta),
)

print(
    "True energy respects variational bound:",
    final_true_energy >= ground_energy - 1e-10,
)

print(f"\nOptimization shots: {optimization_shots:,}")
print(f"Final evaluation shots: {final_evaluation_shots:,}")
print(
    f"Total shots: "
    f"{optimization_shots + final_evaluation_shots:,}"
)