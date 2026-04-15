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
    * the prior-induced rate field softplus(f)

"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import math
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from polyagammadensity import  inv_softplus, RampDensity
from quadtree import QuadTree

# ============================================================
# PARAMETERS
# ============================================================
COUNTMAX = 25
MAX_DEPTH = 12

#RHO = 1.5
JITTER = 1e-8
#V2 = 10.0
#baseline_rate = 5.0


# plotting / sampling
SEED = 0
POWER_GAMMA_RATE = 0.6
N_PRIOR_SAMPLES = 60
N_RATE_SAMPLES_TO_PLOT = 6
N_COUNT_SAMPLES_TO_PLOT = 6
N_SELECTED_CELLS = 6

# ============================================================
# PATHS
# ============================================================
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "data"

JSON_FILE = DATA_DIR / "earthquakes_3point5_cl_2010-2020.json"
CSV_FILE = DATA_DIR / "earthquakes_3point5_cl_2010-2020.csv"


# ============================================================
# HELPERS
# ============================================================
def load_earthquake_data() -> gpd.GeoDataFrame:
    if JSON_FILE.exists():
        gdf = gpd.read_file(JSON_FILE)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        return gdf

    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )
        return gdf

    raise FileNotFoundError(
        "Keine Erdbebendaten gefunden. Erwartet wird eine der Dateien:\n"
        f"- {JSON_FILE}\n"
        f"- {CSV_FILE}"
    )

def gaussian_covariance_from_coords(
    x: np.ndarray,
    y: np.ndarray,
    rho: float,
    v2: float,
    jitter: float = 1e-8,
    ) -> np.ndarray:
    coords = np.column_stack([x, y])
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    cov = v2 * np.exp(-dist2 / (2.0 * rho**2))
    cov += jitter * np.eye(len(x))
    return cov

def build_quadtree_grid(
    gdf: gpd.GeoDataFrame,
    countmax: int,
    maxdepth: int,
) -> tuple[QuadTree, pd.DataFrame]:
    xs = gdf.geometry.x.to_numpy(dtype=float)
    ys = gdf.geometry.y.to_numpy(dtype=float)
    minx, miny, maxx, maxy = gdf.total_bounds

    qt = QuadTree(minx, maxx, miny, maxy, xs, ys, countmax=countmax, maxdepth=maxdepth)
    grid_df = pd.DataFrame(qt.lop).copy()
    grid_df["x_center"] = 0.5 * (grid_df["xmin"] + grid_df["xmax"])
    grid_df["y_center"] = 0.5 * (grid_df["ymin"] + grid_df["ymax"])
    grid_df["width"] = grid_df["xmax"] - grid_df["xmin"]
    grid_df["height"] = grid_df["ymax"] - grid_df["ymin"]
    grid_df["area"] = grid_df["width"] * grid_df["height"]
    return qt, grid_df


def plot_quadtree_values(
    grid_df: pd.DataFrame,
    values: np.ndarray,
    title: str,
    cmap: str = "viridis",
    power_gamma: float | None = None,
    ax: plt.Axes | None = None,
):
    values = np.asarray(values, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    if np.allclose(values.min(), values.max()):
        norm = plt.Normalize(vmin=values.min() - 1e-12, vmax=values.max() + 1e-12)
    elif power_gamma is not None:
        norm = PowerNorm(gamma=power_gamma, vmin=float(values.min()), vmax=float(values.max()))
    else:
        norm = plt.Normalize(vmin=float(values.min()), vmax=float(values.max()))

    cmap_obj = plt.get_cmap(cmap)

    for i, row in grid_df.iterrows():
        rect = Rectangle(
            (row["xmin"], row["ymin"]),
            row["width"],
            row["height"],
            facecolor=cmap_obj(norm(values[i])),
            edgecolor="black",
            linewidth=0.3,
            alpha=0.95,
        )
        ax.add_patch(rect)

    minx = float(grid_df["xmin"].min())
    maxx = float(grid_df["xmax"].max())
    miny = float(grid_df["ymin"].min())
    maxy = float(grid_df["ymax"].max())
    pad_x = 0.03 * (maxx - minx)
    pad_y = 0.03 * (maxy - miny)

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax)
    return ax


def make_prior_model(
    grid_df: pd.DataFrame, 
    nobs: np.ndarray,
    rho: float,
    v2: float,
    baseline_rate: float
    ) -> tuple[RampDensity, np.ndarray, np.ndarray]:
    # catch cell centers as floats for covariance construction
    x_center = grid_df["x_center"].to_numpy(dtype=float)
    y_center = grid_df["y_center"].to_numpy(dtype=float)

 
    
    prior_mean = inv_softplus(baseline_rate * np.ones(len(nobs)))
    Sigma0 = gaussian_covariance_from_coords(
        x_center,
        y_center,
        rho=rho,
        v2=v2,
        jitter=JITTER,
    )

    rd = RampDensity(
        prior_mean=prior_mean,
        prior_covariance=Sigma0
    )
    rd.set_data(nobs)
    return rd, prior_mean, Sigma0


def draw_prior_samples(rd: RampDensity, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    L = np.asarray(rd.Lprior, dtype=float)
    z = rng.normal(size=(n_samples, rd.nbins))
    return rd.prior_mean[None, :] + z @ L.T



def choose_cells(grid_df: pd.DataFrame, n_cells: int) -> np.ndarray:
    # Deterministic spread over area quantiles for diagnostics.
    order = np.argsort(grid_df["area"].to_numpy())
    idx = np.linspace(0, len(order) - 1, min(n_cells, len(order)), dtype=int)
    return order[idx]



# ============================================================
# PRIOR VISUALIZATION
# ============================================================
def plot_selected_cell_distributions(
    f_samples: np.ndarray,
    rate_samples: np.ndarray,
    grid_df: pd.DataFrame,
    cell_indices: np.ndarray,
) -> None:
    nrows = len(cell_indices)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 3.0 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for r, idx in enumerate(cell_indices):
        area = grid_df.iloc[idx]["area"]
        axes[r, 0].hist(f_samples[:, idx], bins=25)
        axes[r, 0].set_title(f"Cell {idx}: prior f, area={area:.4g}")
        axes[r, 0].set_xlabel("f")
        axes[r, 0].set_ylabel("frequency")

        axes[r, 1].hist(rate_samples[:, idx], bins=25)
        axes[r, 1].set_title(f"Cell {idx}: prior-induced rate")
        axes[r, 1].set_xlabel("rate = softplus(f)")
        axes[r, 1].set_ylabel("frequency")

    fig.suptitle("Marginal prior distributions for selected quadtree cells", y=0.995)
    fig.tight_layout()


def plot_sample_panels(
    grid_df: pd.DataFrame,
    samples: np.ndarray,
    title_prefix: str,
    cmap: str,
    power_gamma: float | None = None,
    max_plots: int = 6,
) -> None:
    n = min(max_plots, len(samples))
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        plot_quadtree_values(
            grid_df,
            samples[i],
            title=f"{title_prefix} #{i+1}",
            cmap=cmap,
            power_gamma=power_gamma,
            ax=axes[i],
        )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()


def plot_covariance_and_correlation(
    Sigma0: np.ndarray,
    grid_df: pd.DataFrame,
    selected_cells: np.ndarray,
) -> None:
    subcov = Sigma0[np.ix_(selected_cells, selected_cells)]
    sd = np.sqrt(np.diag(subcov))
    corr = subcov / np.outer(sd, sd)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(subcov)
    axes[0].set_title("Prior covariance (selected cells)")
    axes[0].set_xlabel("cell index")
    axes[0].set_ylabel("cell index")
    axes[0].set_xticks(range(len(selected_cells)), labels=selected_cells)
    axes[0].set_yticks(range(len(selected_cells)), labels=selected_cells)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(corr, vmin=-1, vmax=1)
    axes[1].set_title("Prior correlation (selected cells)")
    axes[1].set_xlabel("cell index")
    axes[1].set_ylabel("cell index")
    axes[1].set_xticks(range(len(selected_cells)), labels=selected_cells)
    axes[1].set_yticks(range(len(selected_cells)), labels=selected_cells)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()


# ============================================================
# MAIN
# ============================================================
def main(rho = 1.5, v2 = 10, baseline_rate = 5):
    # build quadtree and prior model
    rng = np.random.default_rng(SEED)

    gdf = load_earthquake_data()
    qt, grid_df = build_quadtree_grid(gdf, countmax=COUNTMAX, maxdepth=MAX_DEPTH)

    nobs = grid_df["count"].to_numpy(dtype=int)
    rd, prior_mean, Sigma0 = make_prior_model(
        grid_df, 
        nobs, 
        rho, 
        v2, 
        baseline_rate
        )
   
    print(f"rho={rho}, v2={v2}, baseline_rate={baseline_rate}")


    f_prior_sd = np.sqrt(np.diag(Sigma0))
    rate_prior_mean = rd.field_from_f(prior_mean)

    print(f"Loaded {len(gdf)} earthquake events")
    print(f"Constructed {len(grid_df)} quadtree cells")
    print(f"prior mean in f-space: min={prior_mean.min():.4f}, max={prior_mean.max():.4f}")
    print(f"prior sd in f-space: min={f_prior_sd.min():.4f}, max={f_prior_sd.max():.4f}")
    print(f"prior mean rate: min={rate_prior_mean.min():.4f}, max={rate_prior_mean.max():.4f}")

    # Base quadtree diagnostics
    plot_quadtree_values(grid_df, nobs, "Observed counts on quadtree grid", cmap="viridis", power_gamma=0.6)
    plot_quadtree_values(grid_df, grid_df["area"].to_numpy(), "Quadtree cell areas", cmap="cividis", power_gamma=0.5)

    # Prior-Pipeline
    # Prior mean / sd in latent space and induced mean rate
    plot_quadtree_values(grid_df, prior_mean, "Prior mean of latent field f", cmap="coolwarm")
    plot_quadtree_values(grid_df, f_prior_sd, "Prior standard deviation of latent field f", cmap="magma")
    plot_quadtree_values(
        grid_df, 
        rate_prior_mean, 
        f"Prior mean rate | rho={rho}, v2={v2}, baseline={baseline_rate}",
        cmap="viridis",
        power_gamma=0.7,
        )
    # Samples from the prior in latent and rate space
    f_samples = draw_prior_samples(rd, N_PRIOR_SAMPLES, rng)
    rate_samples = rd.field_from_f(f_samples)
    count_samples = rng.poisson(rate_samples)

    f_sample_mean = f_samples.mean(axis=0)
    f_sample_sd = f_samples.std(axis=0)
    rate_sample_mean = rate_samples.mean(axis=0)
    rate_sample_sd = rate_samples.std(axis=0)
    count_sample_mean = count_samples.mean(axis=0)

    plot_quadtree_values(grid_df, f_sample_mean, "Monte-Carlo mean of prior samples in f-space", cmap="coolwarm")
    plot_quadtree_values(grid_df, f_sample_sd, "Monte-Carlo sd of prior samples in f-space", cmap="magma")
    plot_quadtree_values(grid_df, rate_sample_mean, "Monte-Carlo mean of prior-induced rates", cmap="viridis", power_gamma=0.7)
    plot_quadtree_values(grid_df, rate_sample_sd, "Monte-Carlo sd of prior-induced rates", cmap="plasma", power_gamma=0.7)
    plot_quadtree_values(grid_df, count_sample_mean, "Prior-predictive mean counts", cmap="viridis", power_gamma=0.6)

    plot_sample_panels(
        grid_df,
        f_samples,
        title_prefix="Latent prior sample f",
        cmap="coolwarm",
        power_gamma=None,
        max_plots=min(6, N_PRIOR_SAMPLES),
    )
    plot_sample_panels(
        grid_df,
        rate_samples,
        title_prefix="Prior-induced rate sample",
        cmap="viridis",
        power_gamma=0.7,
        max_plots=min(N_RATE_SAMPLES_TO_PLOT, N_PRIOR_SAMPLES),
    )
    plot_sample_panels(
        grid_df,
        count_samples,
        title_prefix="Prior-predictive count sample",
        cmap="viridis",
        power_gamma=0.6,
        max_plots=min(N_COUNT_SAMPLES_TO_PLOT, N_PRIOR_SAMPLES),
    )

    # Plot marginal distributions for selected cells and covariance/correlation diagnostics
    selected_cells = choose_cells(grid_df, N_SELECTED_CELLS)
    plot_selected_cell_distributions(f_samples, rate_samples, grid_df, selected_cells)
    plot_covariance_and_correlation(Sigma0, grid_df, selected_cells)

    plt.show()

    return {
        "gdf": gdf,
        "qt": qt,
        "grid_df": grid_df,
        "nobs": nobs,
        "rd": rd,
        "prior_mean": prior_mean,
        "Sigma0": Sigma0,
        "f_samples": f_samples,
        "rate_samples": rate_samples,
        "count_samples": count_samples,
        "selected_cells": selected_cells,
    }



if __name__ == "__main__":
    rho_values = [0.5]
    v2_values = [1.0, 5.0, 10.0]
    baseline_values = [5.0]

    for rho in rho_values:
        for v2 in v2_values:
            for baseline_rate in baseline_values:
                print("\n" + "=" * 70)
                main(rho=rho, v2=v2, baseline_rate=baseline_rate)
