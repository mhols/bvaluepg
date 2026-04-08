"""
Small playground script to visualize the prior on an adaptively built quadtree grid.

What it does
------------
- loads earthquake points from GeoJSON or CSV
- builds a quadtree directly from the raw points
- computes a Gaussian prior on cell centers
- visualizes:
    * prior mean of latent field f
    * prior sd of latent field f
    * a few prior samples of f
    * the implied rate field softplus(f)

This can be changed to show the prior for the sigmoid-transformed field. Only need to change softplus for sigmoid.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from polyagammadensity import softplus, inv_softplus, sigmoid, inv_sigmoid
from quadtree import QuadTree
from exp_gibbs_pg_quadtree2 import build_quadtree_grid, gaussian_covariance_from_coords, plot_quadtree_values, load_earthquake_data

# ============================================================
# PARAMETERS
# ============================================================
COUNTMAX = 25
MAX_DEPTH = 12

RHO = 1
JITTER = 1e-8

N_SAMPLES = 3
SEED = 0
POWER_GAMMA_RATE = 0.6

# plotting / sampling
N_SAMPLES = 3
SEED = 0
POWER_GAMMA_RATE = 0.6


# ============================================================
# PATHS
# ============================================================
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "data"

JSON_FILE = DATA_DIR / "earthquakes_3point5_cl_2010-2020.json"
CSV_FILE = DATA_DIR / "earthquakes_3point5_cl_2010-2020.csv"



# ============================================================
# PRIOR VISUALIZATION
# ============================================================
def visualize_prior(grid_df, prior_mean, prior_cov, n_samples=3, seed=0):
    rng = np.random.default_rng(seed)

    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_sd = np.sqrt(np.maximum(np.diag(prior_cov), 0.0))

    print()
    print("=== Prior summary ===")
    print(f"nbins             = {len(prior_mean)}")
    print(f"prior mean range  = [{prior_mean.min():.4f}, {prior_mean.max():.4f}]")
    print(f"prior sd range    = [{prior_sd.min():.4f}, {prior_sd.max():.4f}]")

    # Plot prior mean
    plot_quadtree_values(
        grid_df,
        prior_mean,
        title="Prior mean of latent field f",
        cmap="viridis",
        power_gamma=None,
    )

    # Plot prior sd
    plot_quadtree_values(
        grid_df,
        prior_sd,
        title="Prior sd of latent field f",
        cmap="magma",
        power_gamma=None,
    )

    # Draw samples
    samples = rng.multivariate_normal(
        mean=prior_mean,
        cov=prior_cov,
        size=n_samples,
        method="cholesky",
    )

    # Plot latent samples
    for k in range(n_samples):
        plot_quadtree_values(
            grid_df,
            samples[k],
            title=f"Prior sample #{k+1} of latent field f",
            cmap="viridis",
            power_gamma=None,
        )

    # Plot prior-induced rate samples
    for k in range(n_samples):
        rate = softplus(samples[k])
        plot_quadtree_values(
            grid_df,
            rate,
            title=f"Prior sample #{k+1} of rate softplus(f)",
            cmap="viridis",
            power_gamma=POWER_GAMMA_RATE,
        )


    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading earthquake data...")
    gdf = load_earthquake_data()
    print(f"Loaded {len(gdf)} earthquake events")
    print(f"CRS: {gdf.crs}")

    print()
    print("Building quadtree...")
    qt, grid_df = build_quadtree_grid(
        gdf,
        countmax=COUNTMAX,
        maxdepth=MAX_DEPTH,
    )

    nobs = grid_df["count"].to_numpy(dtype=int)
    x_center = grid_df["x_center"].to_numpy(dtype=float)
    y_center = grid_df["y_center"].to_numpy(dtype=float)
    area = (
        (grid_df["xmax"].to_numpy(dtype=float) - grid_df["xmin"].to_numpy(dtype=float))
        * (grid_df["ymax"].to_numpy(dtype=float) - grid_df["ymin"].to_numpy(dtype=float))
    )

    lam = max(1, int(np.max(nobs) + 2 * np.sqrt(np.max(nobs))) + 1)
    V2 = 5 / lam

    nbins = len(nobs)
    print(f"Constructed {len(grid_df)} quadtree cells")
    print(f"Total observed events    = {nobs.sum()}")
    print(f"Min/Max counts per cell  = {nobs.min()} / {nobs.max()}")
    print(f"Min/Max cell area        = {area.min():.6f} / {area.max():.6f}")

    # Prior on latent field f
    baseline_rate = np.clip(5.0, 1e-6, lam - 1e-6)
    prior_mean = inv_softplus((baseline_rate / lam) * np.ones(nbins))
    prior_cov = gaussian_covariance_from_coords(
        x_center,
        y_center,
        rho=RHO,
        v2=V2,
        jitter=JITTER,
    )
    

    print()
    print("=== User settings ===")
    print(f"COUNTMAX = {COUNTMAX}")
    print(f"MAX_DEPTH = {MAX_DEPTH}")
    print(f"RHO = {RHO}")
    print(f"V2 = {V2}")
    print(f"LAM0 = {lam}")
    print(f"N_SAMPLES = {N_SAMPLES}")

    # Observed counts for reference
    plot_quadtree_values(
        grid_df,
        nobs,
        title="Observed counts on quadtree grid",
        cmap="viridis",
        power_gamma=0.6,
    )

    # Prior visualization
    visualize_prior(
        grid_df,
        prior_mean,
        prior_cov,
        n_samples=N_SAMPLES,
        seed=SEED,
    )

    return qt, grid_df, prior_mean, prior_cov


if __name__ == "__main__":
    qt, grid_df, prior_mean, prior_cov = main()