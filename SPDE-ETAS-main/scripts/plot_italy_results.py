"""
just plotting
Change RUN_DIR below if another Julia chain/run should be plotted, then run:

    python3 scripts/plot_italy_results.py

The script only reads existing outputs and writes PNG files into RUN_DIR/plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "mcmc_results" / "italy_full_100iter"
GRID_FILE = PROJECT_ROOT / "data" / "italy_mc3_comparison" / "count_grids.npz"
CATALOG_FILE = PROJECT_ROOT / "data" / "italy_mc3_sofiane_mapping.csv"
PLOT_DIR = RUN_DIR / "plots"

SHOW_PLOTS = False
FIGURE_DPI = 160


# ============================================================
# Load shared grid, catalogue, mesh, and Julia chains
# ============================================================

PLOT_DIR.mkdir(parents=True, exist_ok=True)

with np.load(GRID_FILE) as grid:
    counts_all = grid["counts_all"]
    counts_nnd = grid["counts_nnd_kept"]
    x_edges = grid["x_edges"]
    y_edges = grid["y_edges"]

catalog = np.genfromtxt(CATALOG_FILE, delimiter=",", names=True, encoding="utf-8")
x_model = catalog["x_model"]
y_model = catalog["y_model"]

mesh_points = np.loadtxt(RUN_DIR / "mesh_points.tsv")
mesh_cells = np.loadtxt(RUN_DIR / "mesh_cells.tsv", dtype=int) - 1
triangulation = mtri.Triangulation(mesh_points[:, 0], mesh_points[:, 1], mesh_cells)

with (RUN_DIR / "chains_parameters.csv").open("r", encoding="utf-8") as stream:
    parameter_names = stream.readline().strip().split(",")
parameters = np.loadtxt(RUN_DIR / "chains_parameters.csv", delimiter=",", skiprows=1)
parameters = np.atleast_2d(parameters)

nbg = np.loadtxt(RUN_DIR / "chains_nbg.csv", skiprows=1)
nbg = np.atleast_1d(nbg)

with np.load(RUN_DIR / "julia_expected_counts_on_pg_grid.npz") as projected:
    expected_samples = projected["expected_count_samples"]
    expected_mean = projected["expected_count_mean"]
    expected_sd = projected["expected_count_sd"]

print(f"Run directory: {RUN_DIR}")
print(f"MCMC samples in parameter file: {len(parameters)}")
print(f"Spatial samples on PG grid: {len(expected_samples)}")
if len(parameters) < 100:
    print("WARNING: fewer than 100 samples; plots are technical diagnostics only.")


# ============================================================
# 1. Observed counts: all events and NND-kept events
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
count_vmax = max(float(counts_all.max()), float(counts_nnd.max()))

image = axes[0].pcolormesh(
    x_edges, y_edges, counts_all, shading="flat", cmap="viridis", vmin=0, vmax=count_vmax
)
axes[0].set_title(f"All Mc=3.0 events (n={counts_all.sum()})")
axes[1].pcolormesh(
    x_edges, y_edges, counts_nnd, shading="flat", cmap="viridis", vmin=0, vmax=count_vmax
)
axes[1].set_title(f"NND-kept Mc=3.0 events (n={counts_nnd.sum()})")

for axis in axes:
    axis.set_xlabel("rotated x [km]")
    axis.set_ylabel("rotated y [km]")
    axis.set_aspect("equal")
fig.colorbar(image, ax=axes, label="observed events per 20 km cell")
fig.savefig(PLOT_DIR / "01_observed_counts.png", dpi=FIGURE_DPI)


# ============================================================
# 2. Sofiane's triangular SPDE mesh and catalogue locations
# ============================================================

fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
ax.triplot(triangulation, color="0.45", linewidth=0.55, label="SPDE mesh")
ax.scatter(x_model, y_model, s=3, color="C3", alpha=0.28, label="earthquakes")
ax.scatter(mesh_points[:, 0], mesh_points[:, 1], s=7, color="black", zorder=3, label="mesh nodes")
ax.set_title(
    f"Italy SPDE mesh: {len(mesh_points)} nodes, {len(mesh_cells)} triangles"
)
ax.set_xlabel("model x [1 unit = 100 km]")
ax.set_ylabel("model y [1 unit = 100 km]")
ax.set_aspect("equal")
ax.legend(loc="upper right", markerscale=2)
fig.savefig(PLOT_DIR / "02_spde_mesh.png", dpi=FIGURE_DPI)


# ============================================================
# 3. Julia posterior background mean and uncertainty
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
mean_image = axes[0].pcolormesh(
    x_edges, y_edges, expected_mean, shading="flat", cmap="magma", vmin=0
)
sd_image = axes[1].pcolormesh(
    x_edges, y_edges, expected_sd, shading="flat", cmap="cividis", vmin=0
)
axes[0].set_title("Julia expected background count: posterior mean")
axes[1].set_title("Julia expected background count: posterior SD")

for axis in axes:
    axis.set_xlabel("rotated x [km]")
    axis.set_ylabel("rotated y [km]")
    axis.set_aspect("equal")
fig.colorbar(mean_image, ax=axes[0], label="expected events per grid cell")
fig.colorbar(sd_image, ax=axes[1], label="posterior standard deviation")
fig.savefig(PLOT_DIR / "03_julia_background_mean_sd.png", dpi=FIGURE_DPI)


# ============================================================
# 4. One spatial posterior sample, as in the PG experiment
# ============================================================

fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
sample_image = ax.pcolormesh(
    x_edges, y_edges, expected_samples[0], shading="flat", cmap="magma", vmin=0
)
ax.set_title("One Julia background-field sample")
ax.set_xlabel("rotated x [km]")
ax.set_ylabel("rotated y [km]")
ax.set_aspect("equal")
fig.colorbar(sample_image, ax=ax, label="expected events per grid cell")
fig.savefig(PLOT_DIR / "04_julia_background_sample.png", dpi=FIGURE_DPI)


# ============================================================
# 5. Trace plots for all scalar model parameters
# ============================================================

n_parameter = len(parameter_names)
n_columns = 2
n_rows = int(np.ceil(n_parameter / n_columns))
fig, axes = plt.subplots(
    n_rows, n_columns, figsize=(12, 2.35 * n_rows), constrained_layout=True
)
axes = np.asarray(axes).ravel()

for index, name in enumerate(parameter_names):
    axes[index].plot(np.arange(1, len(parameters) + 1), parameters[:, index], linewidth=0.9)
    axes[index].set_title(name)
    axes[index].set_xlabel("retained iteration")
    axes[index].set_ylabel("value")
for axis in axes[n_parameter:]:
    axis.set_visible(False)

fig.suptitle("SPDE-ETAS parameter traces")
fig.savefig(PLOT_DIR / "05_parameter_traces.png", dpi=FIGURE_DPI)


# ============================================================
# 6. Background-event count trace and histogram
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
axes[0].plot(np.arange(1, len(nbg) + 1), nbg, color="C0", linewidth=0.9)
axes[0].axhline(counts_nnd.sum(), color="C3", linestyle="--", label="NND-kept count")
axes[0].set_title("Background-event count trace")
axes[0].set_xlabel("retained iteration")
axes[0].set_ylabel("number of background events")
axes[0].legend()

axes[1].hist(nbg, bins=min(30, max(1, len(nbg))), color="0.45", edgecolor="white")
axes[1].axvline(counts_nnd.sum(), color="C3", linestyle="--", label="NND-kept count")
axes[1].set_title("Background-event count distribution")
axes[1].set_xlabel("number of background events")
axes[1].set_ylabel("frequency")
axes[1].legend()
fig.savefig(PLOT_DIR / "06_background_count.png", dpi=FIGURE_DPI)


# ============================================================
# 7. Posterior distribution in the cell with largest mean
# ============================================================

selected_flat_index = int(np.nanargmax(expected_mean))
selected_iy, selected_ix = np.unravel_index(selected_flat_index, expected_mean.shape)
selected_samples = expected_samples[:, selected_iy, selected_ix]
selected_samples = selected_samples[np.isfinite(selected_samples)]

fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
ax.hist(
    selected_samples,
    bins=min(25, max(1, len(selected_samples))),
    color="C0",
    edgecolor="white",
)
ax.axvline(selected_samples.mean(), color="black", linestyle="--", label="sample mean")
ax.set_title(f"Highest-mean grid cell: ix={selected_ix}, iy={selected_iy}")
ax.set_xlabel("expected background events in cell")
ax.set_ylabel("frequency")
ax.legend()
fig.savefig(PLOT_DIR / "07_selected_cell_histogram.png", dpi=FIGURE_DPI)
plt.show()


if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")

print(f"Plots saved in: {PLOT_DIR}")
