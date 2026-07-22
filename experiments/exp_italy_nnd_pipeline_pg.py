from __future__ import annotations

"""PG posterior experiment for the linear Italy NND pipeline."""

import json
import os
import sys
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from covariance_kernels import precision_matern
from polyagammadensity import PolyaGammaDensity2D, inv_sigmoid


# Experiment parameters. Change values here before running the script.
INPUT_PREFIX = REPO_ROOT / "data" / "italy_nnd_rotate_cut_grid_Mc_2.5_eta_-4.60"

N_ITER = 100
BURN_IN = 10
THIN = 2
RANDOM_SEED = 0

RHO = 3.0
PRIOR_VARIANCE = 1.0
BOUNDARY = "symmetric"

# Keep as None to derive lambda from the observed count grid.
LAMBDA_SCALE = None

# Logarithmic colour scale with a linear interval from 0 to 1, so zero-count
# cells remain visible. Set to False for a linear colour scale.
LOG_COLOR_SCALE = True


def related_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def main() -> None:
    with np.load(related_path(INPUT_PREFIX, "_counts.npz")) as data:
        counts = data["counts"].astype(int)
        x_edges = data["x_edges"]
        y_edges = data["y_edges"]
    with related_path(INPUT_PREFIX, "_meta.json").open("r", encoding="utf-8") as stream:
        meta = json.load(stream)

    ny, nx = counts.shape
    if [ny, nx] != meta["grid"]["shape_ny_nx"]:
        raise ValueError("Count-grid shape does not match pipeline metadata")
    lam = LAMBDA_SCALE
    if lam is None:
        lam = float(max(int(counts.max()) + 2, np.ceil(1.35 * np.percentile(counts, 99.5)), 1))
    mean_count = float(counts.mean())
    prior_probability = float(np.clip(mean_count / lam, 1e-6, 1 - 1e-6))
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
    samples = np.asarray(
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
    if samples.size == 0:
        raise ValueError("No samples retained; adjust n-iter, burn-in, and thin")
    rate_samples = model.field_from_f(samples)
    posterior_mean = rate_samples.mean(axis=0).reshape(ny, nx, order="C")
    posterior_sd = rate_samples.std(axis=0).reshape(ny, nx, order="C")

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = ((counts, "Observed declustered counts"), (posterior_mean, "Posterior mean rate"), (posterior_sd, "Posterior rate SD"))
    for ax, (image, title) in zip(axes, panels):
        norm = None
        if LOG_COLOR_SCALE:
            norm = SymLogNorm(linthresh=1.0, vmin=0.0, vmax=max(1.0, float(np.max(image))))
        artist = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            norm=norm,
        )
        ax.set_title(title)
        ax.set_xlabel("rotated x [km]")
        ax.set_ylabel("rotated y [km]")
        fig.colorbar(artist, ax=ax)
    print(f"grid: {nx} x {ny}; events: {counts.sum()}; lambda: {lam:g}; retained samples: {len(samples)}")
    plt.show()


if __name__ == "__main__":
    main()
