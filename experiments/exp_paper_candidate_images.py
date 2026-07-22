from __future__ import annotations

"""Kandidatenbilder fuer Paper/Meeting.

Erzeugte Figures:
- Ein-Bin-Vergleich auf f-Skala: exakte Dichte, Laplace, PG-Samples.
  Code: add_single_bin_figures() -> single_bin.plot_f_space()
- Ein-Bin-Vergleich auf Rate-Skala.
  Code: add_single_bin_figures() -> single_bin.plot_rate_space()
- Ein-Bin-Histogramm mit rohen PG-Sample-Counts.
  Code: add_single_bin_figures() -> single_bin.plot_sample_counts()
- Italien-Pipeline-Diagnostik: NND-Verteilung und kept/triggered-Karte.
  Code: add_pipeline_diagnostic_figures() -> italy_diag.plot_nnd_diagnostics(...)
- Italien Count-Grids: alter Workflow, neuer Workflow, Differenz.
  Code: add_pipeline_diagnostic_figures() -> italy_diag.plot_grid_comparison(...)
- Italien Count-Histogramme und Zellvergleich.
  Code: add_pipeline_diagnostic_figures() -> italy_diag.plot_count_distributions(...)
- Italien Prior-Rate-Samples.
  Code: add_italy_prior_and_pg_figures() -> prior_rates + plt.subplots(1, 3, ...)
- Italien PG-Posterior: Counts, posterior mean rate, posterior SD, posterior CV.
  Code: add_italy_prior_and_pg_figures() -> posterior_mean/posterior_sd/posterior_cv + plt.subplots(2, 2, ...)
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import exp_italy_nnd_pipeline_diagnostics as italy_diag
import exp_single_bin_laplace_vs_pg as single_bin
from covariance_kernels import precision_matern
from polyagammadensity import PolyaGammaDensity2D, inv_sigmoid


# Main input from the decluster -> rotate -> cut -> grid/counts pipeline.
INPUT_PREFIX = REPO_ROOT / "data" / "italy_nnd_rotate_cut_grid_Mc_2.5_eta_-4.60"
OLD_CATALOG = REPO_ROOT / "data" / "italy_ingv_rotated_rect_events_declustered_Mc_2.5_eta_-4.60.csv"

# PG settings for the Italy overview figure. This is intentionally small enough
# to run as an exploratory plot script.
N_ITER = 100
BURN_IN = 10
THIN = 2
RANDOM_SEED = 0

RHO = 3.0
PRIOR_VARIANCE = 1.0
BOUNDARY = "symmetric"
LAMBDA_SCALE = None

# Output switches. Change these here when needed.
SAVE_FIGURES = False
SHOW_FIGURES = True
SKIP_SINGLE_BIN = False
SKIP_ITALY_DIAGNOSTICS = False
SKIP_ITALY_PG = False
OUTPUT_DIR = REPO_ROOT / "plots" / "paper_candidate_images"


def related_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def register_figures(figures: list[tuple[str, plt.Figure]], label: str) -> None:
    known = {num for _, fig in figures for num in [fig.number]}
    for num in plt.get_fignums():
        if num not in known:
            figures.append((f"{len(figures) + 1:02d}_{label}", plt.figure(num)))


def add_single_bin_figures(figures: list[tuple[str, plt.Figure]]) -> None:
    single_bin.plot_f_space()
    register_figures(figures, "single_bin_f_density")
    single_bin.plot_rate_space()
    register_figures(figures, "single_bin_rate_density")
    single_bin.plot_sample_counts()
    register_figures(figures, "single_bin_raw_sample_counts")


def add_pipeline_diagnostic_figures(figures: list[tuple[str, plt.Figure]]) -> None:
    events, old, new_counts, old_counts, x_edges, y_edges, meta = italy_diag.load_inputs(
        INPUT_PREFIX,
        OLD_CATALOG,
    )
    matched = italy_diag.comparison_table(events, old)
    italy_diag.print_summary(events, old, new_counts, old_counts, matched, meta)

    italy_diag.plot_nnd_diagnostics(events, float(meta["nnd"]["eta_threshold_log10"]))
    register_figures(figures, "italy_nnd_diagnostics")
    italy_diag.plot_grid_comparison(new_counts, old_counts, x_edges, y_edges)
    register_figures(figures, "italy_old_new_count_grid")
    italy_diag.plot_count_distributions(new_counts, old_counts)
    register_figures(figures, "italy_count_histograms")


def load_count_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(related_path(INPUT_PREFIX, "_counts.npz")) as data:
        counts = data["counts"].astype(int)
        x_edges = data["x_edges"]
        y_edges = data["y_edges"]
    with related_path(INPUT_PREFIX, "_meta.json").open("r", encoding="utf-8") as stream:
        meta = json.load(stream)
    return counts, x_edges, y_edges, meta


def build_italy_model(counts: np.ndarray) -> tuple[PolyaGammaDensity2D, float, float]:
    ny, nx = counts.shape
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
    return model, lam, prior_mean_scalar


def add_italy_prior_and_pg_figures(figures: list[tuple[str, plt.Figure]]) -> None:
    counts, x_edges, y_edges, meta = load_count_grid()
    ny, nx = counts.shape
    if [ny, nx] != meta["grid"]["shape_ny_nx"]:
        raise ValueError("Count-grid shape does not match pipeline metadata")

    model, lam, prior_mean_scalar = build_italy_model(counts)
    rng_state = np.random.get_state()
    np.random.seed(RANDOM_SEED)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            prior_f = np.asarray([model.random_prior_parameters() for _ in range(3)])
    finally:
        np.random.set_state(rng_state)
    prior_rates = model.field_from_f(prior_f).reshape(3, ny, nx, order="C")

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    vmax = float(np.max(prior_rates))
    for ax, image, idx in zip(axes, prior_rates, range(1, 4)):
        artist = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            norm=SymLogNorm(linthresh=1.0, vmin=0.0, vmax=max(1.0, vmax)),
        )
        ax.set_title(f"Prior rate sample {idx}")
        ax.set_xlabel("rotated x [km]")
        ax.set_ylabel("rotated y [km]")
        fig.colorbar(artist, ax=ax, label="rate")
    fig.suptitle(
        f"Italy prior draws: lambda={lam:g}, rho={RHO:g}, variance={PRIOR_VARIANCE:g}, mean f={prior_mean_scalar:.2f}",
        y=1.03,
    )
    register_figures(figures, "italy_prior_rate_samples")

    with contextlib.redirect_stdout(io.StringIO()):
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
    posterior_cv = posterior_sd / np.maximum(posterior_mean, 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = (
        (counts, "Observed declustered counts", "events per cell"),
        (posterior_mean, "Posterior mean rate", "rate"),
        (posterior_sd, "Posterior rate SD", "rate SD"),
        (posterior_cv, "Posterior coefficient of variation", "SD / mean"),
    )
    for ax, (image, title, colorbar_label) in zip(axes.ravel(), panels):
        artist = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            norm=SymLogNorm(linthresh=1.0, vmin=0.0, vmax=max(1.0, float(np.max(image)))),
        )
        ax.set_title(title)
        ax.set_xlabel("rotated x [km]")
        ax.set_ylabel("rotated y [km]")
        fig.colorbar(artist, ax=ax, label=colorbar_label)
    fig.suptitle(
        f"Italy PG posterior: {int(counts.sum())} events, lambda={lam:g}, retained samples={len(samples)}",
        y=1.03,
    )
    register_figures(figures, "italy_pg_posterior_overview")


def save_figures(figures: list[tuple[str, plt.Figure]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures:
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"saved {path}")


def main() -> None:
    figures: list[tuple[str, plt.Figure]] = []

    if not SKIP_SINGLE_BIN:
        add_single_bin_figures(figures)
    if not SKIP_ITALY_DIAGNOSTICS:
        add_pipeline_diagnostic_figures(figures)
    if not SKIP_ITALY_PG:
        add_italy_prior_and_pg_figures(figures)

    if SAVE_FIGURES:
        save_figures(figures, OUTPUT_DIR)
    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
