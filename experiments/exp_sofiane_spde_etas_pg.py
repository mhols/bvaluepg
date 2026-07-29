"""PG-Experiment auf Sofianes SPDE-ETAS-Synthetic-Katalogen.

Figures mit plt.show():
- Sofiane-Raumkatalog mit Background/Triggered-Status.
- Count-Grids fuer alle Ereignisse und nur Background-Ereignisse.
- PG posterior mean/sd fuer alle Ereignisse und nur Background-Ereignisse.
- Differenz: all-events posterior mean minus background-only posterior mean.
- Einzelne f-Samples und Histogramm der f-Samples fuer eine ausgewaehlte Zelle.

Das Skript arbeitet direkt auf Sofianes synthetischen x/y-Koordinaten im
Quadrat [0, 5] x [0, 5]. Die Originaldateien werden nicht veraendert.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from covariance_kernels import precision_matern
from polyagammadensity import PolyaGammaDensity2D, inv_sigmoid


# ============================================================
# PARAMETERS TO CHANGE
# ============================================================

# INPUT_FILE = REPO_ROOT / "data" / "sofiane_spde_etas" / "synthetic_data_case_01_patches_bvaluepg.csv"
INPUT_FILE = REPO_ROOT / "data" / "sofiane_spde_etas" / "synthetic_data_case02_three_faults_bvaluepg.csv"


NX = 50
NY = 50
X_MIN = 0.0
X_MAX = 5.0
Y_MIN = 0.0
Y_MAX = 5.0

N_ITER = 240
BURN_IN = 40
THIN = 2
RANDOM_SEED = 0

RHO = 3.0
PRIOR_VARIANCE = 1.0
BOUNDARY = "symmetric"
LAMBDA_SCALE = None

SAMPLE_TO_PLOT = 0
HIST_BIN_IY = NY // 2
HIST_BIN_IX = NX // 2


def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    required = {"Longitude", "Latitude", "Magnitude", "sofiane_parent_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def make_count_grid(df: pd.DataFrame, background_only: bool) -> np.ndarray:
    selected = df.copy()
    if background_only:
        selected = selected[selected["sofiane_parent_id"] == 0].copy()

    x_edges = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, NY + 1)
    counts, _, _ = np.histogram2d(
        selected["Latitude"].to_numpy(float),
        selected["Longitude"].to_numpy(float),
        bins=[y_edges, x_edges],
    )
    return counts.astype(int)


def choose_lambda_scale(counts: np.ndarray) -> float:
    if LAMBDA_SCALE is not None:
        return float(LAMBDA_SCALE)
    p995 = float(np.percentile(counts, 99.5))
    return float(max(int(counts.max()) + 2, np.ceil(1.35 * p995), 1.0))


def run_pg(counts: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ny, nx = counts.shape
    lam = choose_lambda_scale(counts)
    mean_count = float(counts.mean())
    prior_probability = float(np.clip(mean_count / lam, 1e-6, 1.0 - 1e-6))
    prior_mean_scalar = float(inv_sigmoid(prior_probability))
    prior_precision = precision_matern(
        n=ny,
        m=nx,
        rho=RHO,
        v2=PRIOR_VARIANCE,
        boundary=BOUNDARY,
    )

    model = PolyaGammaDensity2D(
        prior_mean=np.full(ny * nx, prior_mean_scalar),
        prior_precision=prior_precision,
        sparse=True,
        lam=lam,
        n=ny,
        m=nx,
    )
    model.set_data(counts.ravel(order="C"))

    samples_f = np.asarray(
        list(
            model.sample_posterior(
                n_iter=N_ITER,
                burn_in=BURN_IN,
                thin=THIN,
                initial_f=np.full(ny * nx, prior_mean_scalar),
                random_seed=RANDOM_SEED,
            )
        )
    )
    if samples_f.size == 0:
        raise ValueError(f"{label}: no samples retained; adjust N_ITER, BURN_IN, THIN.")

    rate_samples = model.field_from_f(samples_f)
    posterior_mean = rate_samples.mean(axis=0).reshape(ny, nx, order="C")
    posterior_sd = rate_samples.std(axis=0).reshape(ny, nx, order="C")

    print(
        f"{label}: events={int(counts.sum())}, nonzero_cells={int((counts > 0).sum())}, "
        f"lambda={lam:g}, retained_samples={len(samples_f)}"
    )
    return posterior_mean, posterior_sd, samples_f, lam


def plot_input_catalog(df: pd.DataFrame) -> None:
    background = df["sofiane_parent_id"] == 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].scatter(df.loc[background, "Longitude"], df.loc[background, "Latitude"], s=8, alpha=0.7, label="background")
    axes[0].scatter(df.loc[~background, "Longitude"], df.loc[~background, "Latitude"], s=8, alpha=0.5, label="triggered")
    axes[0].set_title("Sofiane catalogue")
    axes[0].set_xlabel("x / Longitude")
    axes[0].set_ylabel("y / Latitude")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend()

    axes[1].hist(df["sofiane_generation"], bins=np.arange(-0.5, df["sofiane_generation"].max() + 1.5, 1))
    axes[1].set_title("Trigger generations")
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("count")
    plt.show()


def plot_counts(counts_all: np.ndarray, counts_background: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = (
        (axes[0], counts_all, "all-event counts"),
        (axes[1], counts_background, "background-only counts"),
        (axes[2], counts_all - counts_background, "triggered counts"),
    )
    for ax, image, title in panels:
        im = ax.imshow(image, origin="lower", extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.show()


def plot_pg_results(
    all_mean: np.ndarray,
    all_sd: np.ndarray,
    background_mean: np.ndarray,
    background_sd: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], all_mean, "PG mean rate: all events"),
        (axes[0, 1], background_mean, "PG mean rate: background only"),
        (axes[0, 2], all_mean - background_mean, "mean rate difference"),
        (axes[1, 0], all_sd, "PG rate SD: all events"),
        (axes[1, 1], background_sd, "PG rate SD: background only"),
        (axes[1, 2], all_sd - background_sd, "SD difference"),
    )
    for ax, image, title in panels:
        im = ax.imshow(image, origin="lower", extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.show()


def plot_f_sample_and_histogram(
    all_f_samples: np.ndarray,
    background_f_samples: np.ndarray,
) -> None:
    sample_idx = min(SAMPLE_TO_PLOT, len(all_f_samples) - 1, len(background_f_samples) - 1)
    flat_idx = HIST_BIN_IY * NX + HIST_BIN_IX

    all_f_image = all_f_samples[sample_idx].reshape(NY, NX, order="C")
    background_f_image = background_f_samples[sample_idx].reshape(NY, NX, order="C")
    all_bin_samples = all_f_samples[:, flat_idx]
    background_bin_samples = background_f_samples[:, flat_idx]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    im0 = axes[0, 0].imshow(all_f_image, origin="lower", extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
    axes[0, 0].scatter(
        [X_MIN + (HIST_BIN_IX + 0.5) * (X_MAX - X_MIN) / NX],
        [Y_MIN + (HIST_BIN_IY + 0.5) * (Y_MAX - Y_MIN) / NY],
        c="red",
        s=35,
    )
    axes[0, 0].set_title(f"single f sample: all events #{sample_idx}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(background_f_image, origin="lower", extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
    axes[0, 1].scatter(
        [X_MIN + (HIST_BIN_IX + 0.5) * (X_MAX - X_MIN) / NX],
        [Y_MIN + (HIST_BIN_IY + 0.5) * (Y_MAX - Y_MIN) / NY],
        c="red",
        s=35,
    )
    axes[0, 1].set_title(f"single f sample: background #{sample_idx}")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    axes[1, 0].hist(all_bin_samples, bins=20, color="0.35", edgecolor="white")
    axes[1, 0].axvline(float(all_bin_samples.mean()), color="red", linewidth=1.5, label="mean")
    axes[1, 0].set_title(f"f samples in bin iy={HIST_BIN_IY}, ix={HIST_BIN_IX}: all")
    axes[1, 0].set_xlabel("f")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend()

    axes[1, 1].hist(background_bin_samples, bins=20, color="0.35", edgecolor="white")
    axes[1, 1].axvline(float(background_bin_samples.mean()), color="red", linewidth=1.5, label="mean")
    axes[1, 1].set_title(f"f samples in bin iy={HIST_BIN_IY}, ix={HIST_BIN_IX}: background")
    axes[1, 1].set_xlabel("f")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend()

    plt.show()


def main() -> None:
    df = load_catalog(INPUT_FILE)
    counts_all = make_count_grid(df, background_only=False)
    counts_background = make_count_grid(df, background_only=True)

    print(f"input: {INPUT_FILE}")
    print(f"rows: {len(df)}")
    print(f"background rows: {int((df['sofiane_parent_id'] == 0).sum())}")
    print(f"triggered rows: {int((df['sofiane_parent_id'] != 0).sum())}")

    all_mean, all_sd, all_f_samples, _ = run_pg(counts_all, label="all events")
    background_mean, background_sd, background_f_samples, _ = run_pg(counts_background, label="background only")

    plot_input_catalog(df)
    plot_counts(counts_all, counts_background)
    plot_pg_results(all_mean, all_sd, background_mean, background_sd)
    plot_f_sample_and_histogram(all_f_samples, background_f_samples)


if __name__ == "__main__":
    main()
