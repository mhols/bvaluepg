"""Softplus-Experiment auf dem aktuellen Italy-Preprocessing-Output.

Figures mit plt.show():
- Beobachtete declustered Counts aus preprocess_nnd_rot_cut_bin.py.
- Posterior mean rate und posterior rate SD.
- Ein einzelnes f-Sample aus dem sampler.
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
from polyagammadensity import RampDensity2D, inv_softplus


# ============================================================
# PARAMETERS TO CHANGE
# ============================================================

INPUT_PREFIX = REPO_ROOT / "data" / "preprocess_nnd_rot_cut_bin_Mc_2.5_eta_-4.60_dkm_20"

N_ITER = 240
BURN_IN = 40
THIN = 2
RANDOM_SEED = 0

RHO_KM = 60.0
PRIOR_VARIANCE = 15.0
BOUNDARY = "symmetric"
SOFTPLUS_K = 1.0
NMAX_MIX_EXTRA = 10


SAMPLE_TO_PLOT = 0
HIST_BIN_IY = None
HIST_BIN_IX = None


def related_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)

def get_cell_size_km_from_filename(prefix: Path) -> float:
    """
    Extract the grid-cell size from a filename containing `_dkm_<value>`.

    Examples
    --------
    "..._dkm_20"   -> 20.0
    "..._dkm_12.5" -> 12.5
    """
    match = re.search(
        r"(?:^|_)dkm_([0-9]+(?:\.[0-9]+)?)(?:_|$)",
        prefix.name,
    )

    if match is None:
        raise ValueError(
            "Could not determine the cell size from the input filename. "
            "Expected a component such as '_dkm_20'."
        )

    cell_size_km = float(match.group(1))

    if cell_size_km <= 0.0:
        raise ValueError("Cell size extracted from filename must be positive.")

    return cell_size_km


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




def make_model(counts: np.ndarray) -> tuple[RampDensity2D, float]:
    ny, nx = counts.shape
    cell_size_km = get_cell_size_km_from_filename(INPUT_PREFIX)
    rho_cells = RHO_KM / cell_size_km
    prior_precision = precision_matern(n=ny, m=nx, rho=rho_cells, v2=PRIOR_VARIANCE, boundary=BOUNDARY)
    mean_count = float(counts.mean())
    prior_mean_scalar = float(inv_softplus(max(mean_count, 1e-6)))
    nmax_mix = int(counts.max()) + NMAX_MIX_EXTRA
    model = RampDensity2D(prior_mean=np.full(ny * nx, prior_mean_scalar), prior_precision=prior_precision, sparse=True, n=ny, m=nx, nmax_mix=nmax_mix, softplus_k=SOFTPLUS_K, cache_dir=None)
    model.set_data(counts.ravel(order="C"))
    return model, prior_mean_scalar


def sample_softplus(
    model: RampDensity2D,
    initial_f: np.ndarray,
    histogram_flat_idx: int,
) -> dict:
    n_samples = 0

    sum_f = np.zeros(model.nbins, dtype=float)
    sumsq_f = np.zeros(model.nbins, dtype=float)

    sum_rate = np.zeros(model.nbins, dtype=float)
    sumsq_rate = np.zeros(model.nbins, dtype=float)

    histogram_samples = []

    selected_sample = None
    selected_sample_index = None
    final_f = None

    for sample_idx, f in enumerate(
        model.sample_posterior(
            n_iter=N_ITER,
            burn_in=BURN_IN,
            thin=THIN,
            initial_f=initial_f,
            random_seed=RANDOM_SEED,
        )
    ):
     
        rate = model.field_from_f(f)


        sum_f += f
        sumsq_f += f * f

      
        sum_rate += rate
        sumsq_rate += rate * rate

       
        histogram_samples.append(float(f[histogram_flat_idx]))


        if sample_idx == SAMPLE_TO_PLOT:
            selected_sample = f.copy()
            selected_sample_index = sample_idx

        final_f = f.copy()
        n_samples += 1

    if n_samples == 0:
        raise ValueError(
            "No samples retained; adjust N_ITER, BURN_IN, THIN."
        )

    posterior_mean_f = sum_f / n_samples
    posterior_var_f = np.maximum(
        sumsq_f / n_samples - posterior_mean_f**2,
        0.0,
    )

    posterior_mean_rate = sum_rate / n_samples
    posterior_var_rate = np.maximum(
        sumsq_rate / n_samples - posterior_mean_rate**2,
        0.0,
    )


    if selected_sample is None:
        selected_sample = final_f
        selected_sample_index = n_samples - 1

    return {
        "n_samples": n_samples,
        "mean_f": posterior_mean_f,
        "sd_f": np.sqrt(posterior_var_f),
        "mean_rate": posterior_mean_rate,
        "sd_rate": np.sqrt(posterior_var_rate),
        "selected_sample": selected_sample,
        "selected_sample_index": selected_sample_index,
        "histogram_samples": np.asarray(
            histogram_samples,
            dtype=float,
        ),
        "final_f": final_f,
    }


def image(model: RampDensity2D, values: np.ndarray) -> np.ndarray:
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


def plot_softplus_rate(
    posterior_mean_rate: np.ndarray,
    posterior_sd_rate: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    panels = (
        (axes[0], posterior_mean_rate, "Softplus posterior mean rate"),
        (axes[1], posterior_sd_rate, "Softplus posterior rate SD"),
    )

    for ax, values, title in panels:
        im = ax.imshow(values, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x_rot_km")
        ax.set_ylabel("y_rot_km")
        fig.colorbar(im, ax=ax)

    plt.show()

def plot_softplus_f(
    posterior_mean_f: np.ndarray,
    posterior_sd_f: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
 

    panels = (
        (
            axes[0],
            posterior_mean_f,
            "Softplus posterior mean f",
        ),
        (
            axes[1],
            posterior_sd_f,
            "Softplus posterior f SD",
        ),
    )

    for ax, values, title in panels:
        im = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("x_rot_km")
        ax.set_ylabel("y_rot_km")
        fig.colorbar(im, ax=ax)

    plt.show()

def plot_f_diagnostics(
    model: RampDensity2D,
    selected_f: np.ndarray,
    selected_sample_index: int,
    f_bin_samples: np.ndarray,
    bin_iy: int,
    bin_ix: int,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    extent = [
        x_edges[0],
        x_edges[-1],
        y_edges[0],
        y_edges[-1],
    ]

    f_image = image(model, selected_f)

    x_marker = 0.5 * (
        x_edges[bin_ix] + x_edges[bin_ix + 1]
    )
    y_marker = 0.5 * (
        y_edges[bin_iy] + y_edges[bin_iy + 1]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )

    im = axes[0].imshow(
        f_image,
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    axes[0].scatter(
        [x_marker],
        [y_marker],
        c="red",
        s=35,
    )
    axes[0].set_title(
        f"single f sample #{selected_sample_index}"
    )
    axes[0].set_xlabel("x_rot_km")
    axes[0].set_ylabel("y_rot_km")
    fig.colorbar(im, ax=axes[0])

    axes[1].hist(
        f_bin_samples,
        bins=25,
        color="0.45",
        edgecolor="white",
    )
    axes[1].axvline(
        float(f_bin_samples.mean()),
        color="red",
        linewidth=1.5,
        label="mean",
    )
    axes[1].set_title(
        f"f samples in bin iy={bin_iy}, ix={bin_ix}"
    )
    axes[1].set_xlabel("f")
    axes[1].set_ylabel("sample count")
    axes[1].legend()

    plt.show()


def main() -> None:
    counts, x_edges, y_edges, meta = load_inputs()
    ny, nx = counts.shape

    expected_shape = meta["grid"]["shape_ny_nx"]
    if [ny, nx] != expected_shape:
        raise ValueError(
            f"Counts shape {counts.shape} does not match "
            f"metadata shape {expected_shape}."
        )

    model, prior_mean_scalar = make_model(counts)

    initial_f = np.full(ny * nx, prior_mean_scalar)

    bin_iy = (
        ny // 2
        if HIST_BIN_IY is None
        else int(HIST_BIN_IY)
    )
    bin_ix = (
        nx // 2
        if HIST_BIN_IX is None
        else int(HIST_BIN_IX)
    )

    bin_iy = int(np.clip(bin_iy, 0, ny - 1))
    bin_ix = int(np.clip(bin_ix, 0, nx - 1))

    histogram_flat_idx = bin_iy * nx + bin_ix

    summary = sample_softplus(
        model=model,
        initial_f=initial_f,
        histogram_flat_idx=histogram_flat_idx,
    )

    posterior_mean_f = image(model, summary["mean_f"])
    posterior_sd_f = image(model, summary["sd_f"])

    posterior_mean_rate = image(model, summary["mean_rate"])
    posterior_sd_rate = image(model, summary["sd_rate"])

    print("Italy softplus experiment")
    print(f"input prefix: {INPUT_PREFIX}")
    print(f"grid: {nx} x {ny}")
    print(f"events: {int(counts.sum())}")
    print(f"nonzero cells: {int((counts > 0).sum())}")
    print(f"softplus_k: {SOFTPLUS_K:g}")
    print(f"prior mean scalar: {prior_mean_scalar:g}")
    print(
        f"rho: {RHO:g}; "
        f"prior variance: {PRIOR_VARIANCE:g}; "
        f"boundary: {BOUNDARY}"
    )
    print(f"retained samples: {summary['n_samples']}")
    print(f"max posterior mean rate: {summary['mean_rate'].max():.3f}")
    print(f"max posterior rate SD: {summary['sd_rate'].max():.3f}")
    print(f"max posterior mean f: {summary['mean_f'].max():.3f}")
    print(f"max posterior f SD: {summary['sd_f'].max():.3f}")

    plot_counts(
        counts,
        x_edges,
        y_edges,
    )

    plot_softplus_rate(
        posterior_mean_rate,
        posterior_sd_rate,
        x_edges,
        y_edges,
    )

    plot_softplus_f(
        posterior_mean_f,
        posterior_sd_f,
        x_edges,
        y_edges,
    )

    plot_f_diagnostics(
        model=model,
        selected_f=summary["selected_sample"],
        selected_sample_index=summary["selected_sample_index"],
        f_bin_samples=summary["histogram_samples"],
        bin_iy=bin_iy,
        bin_ix=bin_ix,
        x_edges=x_edges,
        y_edges=y_edges,
    )

if __name__ == "__main__":
    main()