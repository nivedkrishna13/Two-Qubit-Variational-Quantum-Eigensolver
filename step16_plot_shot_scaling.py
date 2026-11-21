from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Observed results from your Step 10 experiment:
# 100 independent energy estimates at each shot count.
shots = np.array([100, 400, 1600, 6400])

observed_sd = np.array([
    0.084092,
    0.039917,
    0.021229,
    0.011407,
])

theta = -2.7049654938

variance_factor = (
    1.5**2 * np.sin(theta)**2
    + 0.7**2 * np.cos(theta)**2
)

# Smooth theoretical curve.
shot_grid = np.geomspace(shots.min(), shots.max(), 200)
predicted_curve = np.sqrt(variance_factor / shot_grid)

# Fit SD ≈ constant * shots**slope.
slope, intercept = np.polyfit(
    np.log(shots),
    np.log(observed_sd),
    deg=1,
)

fig, ax = plt.subplots(
    figsize=(7, 5),
    constrained_layout=True,
)

ax.loglog(
    shot_grid,
    predicted_curve,
    linestyle="--",
    color="black",
    label=r"Prediction: $\sigma_E \propto N^{-1/2}$",
)

ax.loglog(
    shots,
    observed_sd,
    "o-",
    color="tab:blue",
    markersize=7,
    label="Observed SD (100 repetitions)",
)

ax.set(
    xlabel="Shots per measurement setting",
    ylabel="Standard deviation of energy estimates",
    title="Measurement uncertainty decreases with more shots",
)

ax.set_xticks(shots)
ax.set_xticklabels([str(value) for value in shots])

ax.grid(True, which="both", alpha=0.25)
ax.legend()

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

figure_path = output_dir / "shot_scaling.png"
fig.savefig(figure_path, dpi=200)

print(f"Observed fitted exponent: {slope:.3f}")
print("Theoretical exponent:     -0.500")
print(f"Saved figure: {figure_path}")

plt.show()