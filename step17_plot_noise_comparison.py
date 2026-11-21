from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Your Step 13 results, rounded to the printed precision.
noise = np.array([0.0, 0.01, 0.03, 0.05, 0.10])

fixed_energies = np.array([
    -1.65529454,
    -1.60760524,
    -1.51460697,
    -1.42475438,
    -1.21366428,
])

optimized_energies = np.array([
    -1.65529454,
    -1.60762633,
    -1.51478747,
    -1.42523054,
    -1.21532573,
])

ground_energy = -np.hypot(1.5, 0.7)
improvements = fixed_energies - optimized_energies

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4.5),
    constrained_layout=True,
)

axes[0].plot(
    noise,
    fixed_energies,
    "o-",
    label="Fixed ideal angle",
    color="tab:blue",
)

axes[0].plot(
    noise,
    optimized_energies,
    "s--",
    label="Reoptimized under noise",
    color="tab:orange",
)

axes[0].axhline(
    ground_energy,
    color="black",
    linestyle=":",
    label="Ideal ground energy",
)

axes[0].set(
    xlabel="Noise probability per channel, p",
    ylabel="True energy",
    title="Gate noise raises the achievable energy",
)

axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)

axes[1].plot(
    noise,
    improvements,
    "o-",
    color="tab:green",
)

axes[1].set(
    xlabel="Noise probability per channel, p",
    ylabel="Fixed energy − reoptimized energy",
    title="Energy recovered by reoptimization",
)

axes[1].ticklabel_format(
    axis="y",
    style="sci",
    scilimits=(0, 0),
)

axes[1].grid(alpha=0.25)

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

figure_path = output_dir / "noise_comparison.png"
fig.savefig(figure_path, dpi=200)

data_path = output_dir / "noise_comparison.csv"
np.savetxt(
    data_path,
    np.column_stack([
        noise,
        fixed_energies,
        optimized_energies,
        improvements,
    ]),
    delimiter=",",
    header="p,fixed_energy,reoptimized_energy,improvement",
    comments="",
    fmt="%.10f",
)

original_error = fixed_energies[-1] - ground_energy
recovered_percent = 100 * improvements[-1] / original_error

print(f"At p={noise[-1]:.2f}:")
print(f"Energy improvement: {improvements[-1]:.8f}")
print(f"Original energy error recovered: {recovered_percent:.3f}%")
print(
    "Remaining energy error: "
    f"{optimized_energies[-1] - ground_energy:.8f}"
)

print(f"\nSaved figure: {figure_path}")
print(f"Saved data:   {data_path}")

plt.show()