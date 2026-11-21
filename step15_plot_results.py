import numpy as np
import pennylane as qml
from pathlib import Path
import matplotlib.pyplot as plt

p = 0.05
shots = 1000
number_of_steps = 150
initial_theta = 0.37

ground_energy = -np.sqrt(1.5**2 + 0.7**2)

# Exact noisy landscape, used only for diagnostics.
noise_factor = 1 - 4 * p / 3
a = 1.5 * noise_factor**2
b = 0.7 * noise_factor**3

best_noisy_energy = -np.hypot(a, b)
best_noisy_theta = np.arctan2(b, a) - np.pi


def true_noisy_energy(theta):
    return a * np.cos(theta) + b * np.sin(theta)


def noisy_ansatz(theta):
    qml.RY(theta, wires=0)
    qml.DepolarizingChannel(p, wires=0)

    qml.CNOT(wires=[0, 1])

    qml.DepolarizingChannel(p, wires=0)
    qml.DepolarizingChannel(p, wires=1)


device_z = qml.device(
    "default.mixed",
    wires=2,
    shots=shots,
    seed=42,
)

device_x = qml.device(
    "default.mixed",
    wires=2,
    shots=shots,
    seed=123,
)


@qml.qnode(device_z)
def sample_z(theta):
    noisy_ansatz(theta)
    return qml.sample(wires=[0, 1])


@qml.qnode(device_x)
def sample_x(theta):
    noisy_ansatz(theta)

    # Ideal basis rotations for this experiment.
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    return qml.sample(wires=[0, 1])


def sampled_energy(theta):
    signs_z = 1 - 2 * np.asarray(sample_z(theta))
    signs_x = 1 - 2 * np.asarray(sample_x(theta))

    z_contributions = (
        signs_z[:, 0] + 0.5 * signs_z[:, 1]
    )
    x_contributions = (
        0.7 * signs_x[:, 0] * signs_x[:, 1]
    )

    estimate = (
        np.mean(z_contributions)
        + np.mean(x_contributions)
    )

    # Treat each Z shot's combined contribution as one sample.
    # This includes covariance between Z0 and Z1.
    standard_error = np.sqrt(
        np.var(z_contributions, ddof=1) / shots
        + np.var(x_contributions, ddof=1) / shots
    )

    return float(estimate), float(standard_error)


theta = initial_theta

print(f"Noise probability per channel: {p}")
print(f"Shots per setting: {shots}")
print(f"Ideal ground energy: {ground_energy:.10f}")
print(f"Best noisy energy: {best_noisy_energy:.10f}")
print(f"Best noisy angle: {best_noisy_theta:.8f} rad")

print(
    "\nStep      Angle       Sampled gradient"
    "    True noisy energy    Optimization error"
)

# Include the initial state as step zero.
steps_history = [0]
energy_history = [true_noisy_energy(theta)]
error_history = [
    true_noisy_energy(theta) - best_noisy_energy
]
for step in range(1, number_of_steps + 1):
    energy_plus, _ = sampled_energy(theta + np.pi / 2)
    energy_minus, _ = sampled_energy(theta - np.pi / 2)

    gradient = (energy_plus - energy_minus) / 2

    learning_rate = 0.2 / (1 + step / 50)
    theta = theta - learning_rate * gradient

    # Diagnostics only: neither value chooses the update.
    true_energy = true_noisy_energy(theta)
    optimization_error = true_energy - best_noisy_energy
    steps_history.append(step)
    energy_history.append(true_energy)
    error_history.append(optimization_error)
    if step == 1 or step % 15 == 0:
        print(
            f"{step:>4}    "
            f"{theta: .6f}    "
            f"{gradient: .6f}         "
            f"{true_energy: .8f}          "
            f"{optimization_error:.3e}"
        )


# Fresh measurements after optimization.
final_estimate, final_se = sampled_energy(theta)
final_true_energy = true_noisy_energy(theta)

optimization_error = final_true_energy - best_noisy_energy
noise_penalty = best_noisy_energy - ground_energy
total_true_error = final_true_energy - ground_energy
measurement_error = final_estimate - final_true_energy

total_shots = (
    number_of_steps * 2 * 2 * shots
    + 2 * shots
)

print(f"\nFinal angle: {theta:.8f} rad")
print(f"Final sampled energy: {final_estimate:.10f}")
print(f"Estimated standard error: {final_se:.6f}")
print(f"Final true noisy energy: {final_true_energy:.10f}")

print(f"\nOptimization error: {optimization_error:.6e}")
print(f"Noise penalty:      {noise_penalty:.6e}")
print(f"Total true error:   {total_true_error:.6e}")
print(f"Measurement error: {measurement_error:+.6e}")

print(
    "\nError decomposition agrees:",
    np.isclose(
        total_true_error,
        optimization_error + noise_penalty,
        atol=1e-10,
        rtol=0.0,
    ),
)

print(
    "True energy respects variational bound:",
    final_true_energy >= ground_energy - 1e-10,
)

print(
    "True energy improved:",
    final_true_energy < true_noisy_energy(initial_theta),
)

print(f"\nTotal shots: {total_shots:,}")
output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

steps_array = np.asarray(steps_history)
energies_array = np.asarray(energy_history)
errors_array = np.asarray(error_history)

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4.5),
    constrained_layout=True,
)

# Plot 1: Energy convergence.
axes[0].plot(
    steps_array,
    energies_array,
    color="tab:blue",
    linewidth=2,
    label="True noisy energy",
)

axes[0].axhline(
    best_noisy_energy,
    color="tab:orange",
    linestyle="--",
    label="Best noisy energy",
)

axes[0].axhline(
    ground_energy,
    color="black",
    linestyle=":",
    label="Ideal ground energy",
)

# Show the fresh sampled evaluation and its uncertainty.
axes[0].errorbar(
    number_of_steps,
    final_estimate,
    yerr=final_se,
    fmt="o",
    color="tab:red",
    capsize=5,
    label="Final estimate ± 1 SE",
    zorder=5,
)

axes[0].set(
    xlabel="Optimization step",
    ylabel="Energy",
    title=f"VQE convergence: p={p}, {shots} shots/setting",
)

axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.25)

# Plot 2: Distance from the noisy circuit's minimum.
# Clip only for plotting: logarithmic axes cannot display zero.
axes[1].semilogy(
    steps_array,
    np.maximum(errors_array, 1e-12),
    color="tab:purple",
    linewidth=1.5,
)

axes[1].set(
    xlabel="Optimization step",
    ylabel="True energy − best noisy energy",
    title="Optimization error",
)

axes[1].grid(
    True,
    which="both",
    alpha=0.25,
)

figure_path = output_dir / "noisy_sampled_vqe_convergence.png"
fig.savefig(figure_path, dpi=200)

# Save the underlying values for later analysis.
data_path = output_dir / "noisy_sampled_vqe_history.csv"

np.savetxt(
    data_path,
    np.column_stack([
        steps_array,
        energies_array,
        errors_array,
    ]),
    delimiter=",",
    header="step,true_noisy_energy,optimization_error",
    comments="",
    fmt=["%d", "%.12e", "%.12e"],
)

print(f"\nSaved figure: {figure_path}")
print(f"Saved data:   {data_path}")

plt.show()