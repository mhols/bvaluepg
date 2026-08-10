"""PG-Experiment auf dem aktuellen Italy-Preprocessing-Output.

Figures mit plt.show():
- Beobachtete declustered Counts aus preprocess_nnd_rot_cut_bin.py.
- PG posterior mean rate und PG posterior rate SD.
- Ein einzelnes f-Sample aus dem PG-Sampler.
- Histogramm der f-Samples fuer eine ausgewaehlte Grid-Zelle.

"""

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from covariance_kernels import precision_matern
from polyagammadensity import PolyaGammaDensity2D, inv_sigmoid


# ============================================================
# PARAMETERS TO CHANGE
# ============================================================

INPUT_PREFIX = REPO_ROOT / "data" / "preprocess_nnd_rot_cut_bin_Mc_2.5_eta_-4.60_dkm_20"

N_ITER = 240
BURN_IN = 40
THIN = 2
RANDOM_SEED = 0

RHO = 3.0
PRIOR_VARIANCE = 1.0
BOUNDARY = "symmetric"

# None means: choose lambda from the count grid.
LAMBDA_SCALE = None

SAMPLE_TO_PLOT = 0
HIST_BIN_IY = None
HIST_BIN_IX = None


def related_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    counts_path = related_path(INPUT_PREFIX, "_counts.npz")
    meta_path = related_path(INPUT_PREFIX, "_meta.json")

    with np.load(counts_path) as data:
        counts = data["counts"].astype(int)
        x_edges = data["x_edges"].astype(float)
        y_edges = data["y_edges"].astype(float)

    with meta_path.open("r", encoding="utf-8") as stream:
        meta = json.load(stream)

    return counts, x_edges, y_edges, meta


def choose_lambda_scale(counts: np.ndarray) -> float:
    if LAMBDA_SCALE is not None:
        return float(LAMBDA_SCALE)
    p995 = float(np.percentile(counts, 99.5))
    return float(max(int(counts.max()) + 2, np.ceil(1.35 * p995), 1.0))


def make_model(counts: np.ndarray) -> tuple[PolyaGammaDensity2D, float, float]:
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
    return model, lam, prior_mean_scalar


def sample_pg(model: PolyaGammaDensity2D, initial_f: np.ndarray) -> np.ndarray:
    samples_f = np.asarray(
        list(
            model.sample_posterior(
                n_iter=N_ITER,
                burn_in=BURN_IN,
                thin=THIN,
                initial_f=initial_f,
                random_seed=RANDOM_SEED,
            )
        )
    )
    if samples_f.size == 0:
        raise ValueError("No PG samples retained; adjust N_ITER, BURN_IN, THIN.")
    return samples_f


def image(model: PolyaGammaDensity2D, values: np.ndarray) -> np.ndarray:
    return model.scanorder_to_image(values)


def plot_counts(counts: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> None:
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(counts, origin="lower", extent=extent, interpolation="nearest")
    axes[0].set_title("Observed declustered counts")
    axes[0].set_xlabel("x_rot_km")
    axes[0].set_ylabel("y_rot_km")
    fig.colorbar(im0, ax=axes[0], label="events per bin")

    max_count = int(counts.max())
    axes[1].hist(counts.ravel(), bins=np.arange(-0.5, max_count + 1.5, 1), color="0.45", edgecolor="white")
    axes[1].set_yscale("log")
    axes[1].set_title("Count histogram")
    axes[1].set_xlabel("events per bin")
    axes[1].set_ylabel("number of bins")

    plt.show()


def plot_pg_rate(
    posterior_mean_rate: np.ndarray,
    posterior_sd_rate: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    panels = (
        (axes[0], posterior_mean_rate, "PG posterior mean rate"),
        (axes[1], posterior_sd_rate, "PG posterior rate SD"),
    )

    for ax, values, title in panels:
        im = ax.imshow(values, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x_rot_km")
        ax.set_ylabel("y_rot_km")
        fig.colorbar(im, ax=ax)

    plt.show()


def plot_f_diagnostics(
    model: PolyaGammaDensity2D,
    samples_f: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    ny, nx = model.n, model.m
    sample_idx = min(SAMPLE_TO_PLOT, len(samples_f) - 1)
    bin_iy = ny // 2 if HIST_BIN_IY is None else int(HIST_BIN_IY)
    bin_ix = nx // 2 if HIST_BIN_IX is None else int(HIST_BIN_IX)
    bin_iy = int(np.clip(bin_iy, 0, ny - 1))
    bin_ix = int(np.clip(bin_ix, 0, nx - 1))
    flat_idx = bin_iy * nx + bin_ix

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    f_image = image(model, samples_f[sample_idx])
    f_bin_samples = samples_f[:, flat_idx]
    x_marker = 0.5 * (x_edges[bin_ix] + x_edges[bin_ix + 1])
    y_marker = 0.5 * (y_edges[bin_iy] + y_edges[bin_iy + 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im = axes[0].imshow(f_image, origin="lower", extent=extent, interpolation="nearest")
    axes[0].scatter([x_marker], [y_marker], c="red", s=35)
    axes[0].set_title(f"single f sample #{sample_idx}")
    axes[0].set_xlabel("x_rot_km")
    axes[0].set_ylabel("y_rot_km")
    fig.colorbar(im, ax=axes[0])

    axes[1].hist(f_bin_samples, bins=25, color="0.45", edgecolor="white")
    axes[1].axvline(float(f_bin_samples.mean()), color="red", linewidth=1.5, label="mean")
    axes[1].set_title(f"f samples in bin iy={bin_iy}, ix={bin_ix}")
    axes[1].set_xlabel("f")
    axes[1].set_ylabel("sample count")
    axes[1].legend()

    plt.show()


def main() -> None:
    counts, x_edges, y_edges, meta = load_inputs()
    ny, nx = counts.shape

    expected_shape = meta["grid"]["shape_ny_nx"]
    if [ny, nx] != expected_shape:
        raise ValueError(f"Counts shape {counts.shape} does not match metadata shape {expected_shape}.")

    model, lam, prior_mean_scalar = make_model(counts)
    initial_f = np.full(ny * nx, prior_mean_scalar)
    samples_f = sample_pg(model, initial_f)
    rate_samples = model.field_from_f(samples_f)

    posterior_mean_rate = image(model, rate_samples.mean(axis=0))
    posterior_sd_rate = image(model, rate_samples.std(axis=0))
    posterior_mean_f = image(model, samples_f.mean(axis=0))
    posterior_sd_f = image(model, samples_f.std(axis=0))

    print("Italy PG experiment")
    print(f"input prefix: {INPUT_PREFIX}")
    print(f"grid: {nx} x {ny}")
    print(f"events: {int(counts.sum())}")
    print(f"nonzero cells: {int((counts > 0).sum())}")
    print(f"lambda: {lam:g}")
    print(f"rho: {RHO:g}; prior variance: {PRIOR_VARIANCE:g}; boundary: {BOUNDARY}")
    print(f"retained samples: {len(samples_f)}")

    plot_counts(counts, x_edges, y_edges)
    plot_pg_rate(posterior_mean_rate, posterior_sd_rate, x_edges, y_edges)
    plot_pg_rate(posterior_mean_f, posterior_sd_f, x_edges, y_edges)
    plot_f_diagnostics(model, samples_f, x_edges, y_edges)


if __name__ == "__main__":
    main()
