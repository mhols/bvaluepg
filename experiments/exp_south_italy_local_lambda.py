"""South Italy: Poisson-sigmoid model with independent local Gamma scales.


Model (Gamma uses the shape/rate parameterization):
    n_i | f_i, lambda_i ~ Poisson(lambda_i * sigmoid(f_i))
    lambda_i ~ Gamma(alpha, delta),  delta = alpha / LAMBDA_MEAN
    f ~ N(m, Q^{-1})

"""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.special import expit

import italy_data
import polyagammapoisson.catalog.catalog as catalog
import polyagammapoisson.covariance_kernels as ck
import polyagammapoisson.polyagammadensity as pgd


# -----------------------------------------------------------------------------
# Settings: match the South Italy fixed-lambda experiment
# -----------------------------------------------------------------------------
BIN_SIZE_KM = 5.0
MAG_MIN = 2.5
DECLUSTER_ETA0 = -6.2

LAMBDA_MEAN = 20.0             # Prior mean E[lambda_i], in counts per cell
LAM_PRIOR_SHAPE = 2.0       
LAM_PRIOR_RATE = LAM_PRIOR_SHAPE / LAMBDA_MEAN
LAMBDA_PER_SURFACE = LAMBDA_MEAN / BIN_SIZE_KM**2

RHO_KM = 50.0
PRIOR_VARIANCE = 8.0
PRIOR_AVALUE = -2.0

BOUNDARY = "symmetric"

MAP_NITER = 3000
N_ITER = 10000
BURN_IN = 1000
THIN = 1
RANDOM_SEED = 202
N_POSTERIOR_SAMPLES_TO_PLOT = 4

REPO_ROOT = Path(__file__).resolve().parent.parent
# Include rho and v2 so a new sensitivity run never overwrites old figures.
RUN_NAME = (
    f"south_italy_local_lambda_alpha_{LAM_PRIOR_SHAPE:g}"
    f"_rho_{RHO_KM:g}_v2_{PRIOR_VARIANCE:g}"
)
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / RUN_NAME
FIGURES_DIR = REPO_ROOT / "experiments" / "figures" / RUN_NAME
SAMPLER_LOG = RESULTS_DIR / "sampler.log"


# -----------------------------------------------------------------------------
# Catalogue and model
# -----------------------------------------------------------------------------
def prepare_catalogue_and_sampler():
    """Use the same declustering, spatial selection, and binning as the A-value run."""
    A = catalog.AValueAnalyser(
        catalogclass=italy_data.SicilyCalabria,
        catalog=italy_data.CATALOGS["INGV"],
        f_eta_0=DECLUSTER_ETA0,
        BINSIZE=BIN_SIZE_KM,
        PGCLASS=pgd.PolyaGammaDensity2D,
        PRECISIONCLASS=ck.precision_matern,
        prior_avalue=PRIOR_AVALUE,
        sparse=True,
        boundary=BOUNDARY,
        v2=PRIOR_VARIANCE,
        rhokm=RHO_KM,
        lamps=LAMBDA_PER_SURFACE,
        Mc=MAG_MIN,
        # independent local Gamma scales.
        local_lam=True,
        lam_prior_shape=LAM_PRIOR_SHAPE,
        lam_prior_rate=LAM_PRIOR_RATE,
    )

    # Fail clearly if an older polyagammadensity.py is imported by the project.
    if not getattr(A.calc, "local_lam", False):
        raise RuntimeError(
            "The imported PolyaGammaDensity2D does not support local_lam=True. "
            "Use your newer local-Gamma polyagammadensity.py implementation."
        )

    binned = A.binning_count
    counts = np.asarray(binned["counts"], dtype=int)
    ny, nx = counts.shape
    retained_events = len(A.catalog.catDataFrame)
    binned_events = int(counts.sum())

    if retained_events != binned_events:
        raise ValueError(
            f"Spatial binning lost {retained_events - binned_events} events: "
            f"{retained_events} retained, {binned_events} binned. "
            "Check SicilyCalabria filter_xy and its xmin/xmax/ymin/ymax."
        )
    if not np.array_equal(np.asarray(A.calc.nobs).ravel(), counts.ravel(order="C")):
        raise ValueError("The sampler observations do not match the binned counts.")

    # An array is required: one positive lambda value for every grid cell.
    A.calc.lam = np.full(counts.size, LAMBDA_MEAN, dtype=float)
    if np.asarray(A.calc.lam).shape != (counts.size,):
        raise RuntimeError("The sampler did not initialize one lambda per cell.")

    return A, counts, ny, nx, retained_events


def initial_field_from_fixed_lambda_map(A):
    """Conditional MAP under lambda_i = 20, used as an initial state only."""
    print("Computing fixed-baseline MAP for initialization (not a local-Gamma MAP)...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Method TNC does not use Hessian information")
        warnings.filterwarnings("ignore", message="Unknown solver options")
        f_start = A.calc.max_logposterior_estimator(niter=MAP_NITER)
    f_start = np.asarray(f_start, dtype=float).ravel()
    if f_start.shape != (A.calc.nbins,):
        raise ValueError("MAP initialization has the wrong shape.")
    return f_start


# -----------------------------------------------------------------------------
# Posterior summaries
# -----------------------------------------------------------------------------
def update_mean_m2(mean, m2, value, count):
    """Welford update for running means and sums of squared deviations."""
    delta = value - mean
    mean += delta / count
    m2 += delta * (value - mean)


def run_posterior(A, counts, initial_f):
    size = counts.size
    f_mean, f_m2 = np.zeros(size), np.zeros(size)
    lam_mean, lam_m2 = np.zeros(size), np.zeros(size)
    rate_mean, rate_m2 = np.zeros(size), np.zeros(size)
    f_lam_cross_m2 = np.zeros(size)

    total_rate_trace = []
    saved_rate_maps = []

    # Posterior-predictive summaries to detect misspecification.
    observed_stats = {
        "total": int(counts.sum()),
        "zero_cells": int(np.sum(counts == 0)),
        "cells_ge_10": int(np.sum(counts >= 10)),
        "maximum": int(counts.max()),
    }
    predictive_stats = {key: [] for key in observed_stats}
    predictive_rng = np.random.default_rng(RANDOM_SEED + 1000)


    generator = A.calc.sample_posterior(
        n_iter=N_ITER,
        burn_in=BURN_IN,
        thin=THIN,
        initial_f=initial_f,
        random_seed=RANDOM_SEED,
        sample_lam=True,
        return_lam=True,
    )

    n_expected = len(range(BURN_IN, N_ITER, THIN))
    stride = max(1, n_expected // N_POSTERIOR_SAMPLES_TO_PLOT)
    count = 0
    print(f"Sampling local Gamma scales: alpha={LAM_PRIOR_SHAPE:g}, "
          f"delta={LAM_PRIOR_RATE:g}, iterations={N_ITER}, burn-in={BURN_IN}.")


    with SAMPLER_LOG.open("w") as log, redirect_stdout(log):
        for f_sample, lam_sample in generator:
            f = np.asarray(f_sample, dtype=float).ravel()
            lam = np.asarray(lam_sample, dtype=float).ravel()
            if f.shape != (size,) or lam.shape != (size,):
                raise ValueError(
                    "Expected (f, lambda) with one entry per cell. "
                    "Check that sample_lam=True and local_lam=True are supported."
                )
            if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
                raise ValueError("Sampled local lambda contains invalid values.")

            # A draw from the joint posterior has intensity lambda_i * sigmoid(f_i).
            rate = lam * expit(f)
            count += 1

            # Running covariance between f and lambda: a confounding diagnostic.
            delta_f = f - f_mean
            update_mean_m2(f_mean, f_m2, f, count)
            update_mean_m2(lam_mean, lam_m2, lam, count)
            update_mean_m2(rate_mean, rate_m2, rate, count)
            f_lam_cross_m2 += delta_f * (lam - lam_mean)
            total_rate_trace.append(float(rate.sum()))

            if count == 1 or (count % stride == 0 and
                              len(saved_rate_maps) < N_POSTERIOR_SAMPLES_TO_PLOT):
                if len(saved_rate_maps) < N_POSTERIOR_SAMPLES_TO_PLOT:
                    saved_rate_maps.append(rate.copy())

            # Posterior-predictive replicate for one posterior draw.
            n_rep = predictive_rng.poisson(rate)
            predictive_stats["total"].append(int(n_rep.sum()))
            predictive_stats["zero_cells"].append(int(np.sum(n_rep == 0)))
            predictive_stats["cells_ge_10"].append(int(np.sum(n_rep >= 10)))
            predictive_stats["maximum"].append(int(n_rep.max()))

    if count < 2:
        raise RuntimeError("Fewer than two posterior samples were retained.")

    f_sd = np.sqrt(f_m2 / (count - 1))
    lam_sd = np.sqrt(lam_m2 / (count - 1))
    rate_sd = np.sqrt(rate_m2 / (count - 1))
    denom = f_sd * lam_sd
    f_lam_corr = np.divide(
        f_lam_cross_m2 / (count - 1), denom,
        out=np.zeros_like(denom), where=(denom > 0),
    )
    f_lam_corr = np.clip(f_lam_corr, -1.0, 1.0)

    return {
        "n_kept": count,
        "f_mean": f_mean, "f_sd": f_sd,
        "lam_mean": lam_mean, "lam_sd": lam_sd,
        "rate_mean": rate_mean, "rate_sd": rate_sd,
        "f_lam_corr": f_lam_corr,
        "total_rate_trace": np.asarray(total_rate_trace),
        "posterior_rate_samples": saved_rate_maps,
        "observed_stats": observed_stats,
        "predictive_stats": {k: np.asarray(v) for k, v in predictive_stats.items()},
    }


# -----------------------------------------------------------------------------
# Figures 
# -----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_figure(fig, name):
    pdf = FIGURES_DIR / f"{name}.pdf"
    png = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {pdf}")


def plot_map(fig, ax, values, A, title, colorbar_label,
             cmap="viridis", norm=None, vmin=None, vmax=None):

    options = {"axis": ax, "interpolation": "nearest",
               "cmap": cmap, "zorder": 1}
    if norm is not None:
        options["norm"] = norm
    else:
        if vmin is not None:
            options["vmin"] = vmin
        if vmax is not None:
            options["vmax"] = vmax
    im = A.imshow(np.asarray(values).ravel(order="C"), **options)
    A.plot_coastlines(axis=ax, color="white", linewidth=0.65, zorder=3)
    ax.set_title(title)
    ax.set_xlabel(r"$x$ [km]")
    ax.set_ylabel(r"$y$ [km]")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    return im


def make_figures(A, counts, posterior):
    observed = counts.ravel(order="C")
    rate_mean = posterior["rate_mean"]
    rate_sd = posterior["rate_sd"]

    # Shared norm across observed and predicted mean counts.
    intensity_norm = PowerNorm(
        gamma=0.5, vmin=0.0,
        vmax=max(float(observed.max()), float(rate_mean.max())),
    )
    sd_norm = PowerNorm(gamma=0.7, vmin=0.0, vmax=float(rate_sd.max()))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8),
                             constrained_layout=True)
    plot_map(fig, axes[0], observed, A,
             "(a) Observed counts", "Count per cell", norm=intensity_norm)
    plot_map(fig, axes[1], rate_mean, A,
             "(b) Posterior mean intensity", r"$E[\nu_i\mid n]$",
             norm=intensity_norm)
    plot_map(fig, axes[2], rate_sd, A,
             "(c) Posterior SD", r"$\mathrm{SD}(\nu_i\mid n)$",
             cmap="magma", norm=sd_norm)
    save_figure(fig, "main_local_lambda")

    # Visualize the spatial latent field separately from local Gamma effects.
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 7.7),
                             constrained_layout=True)
    plot_map(fig, axes[0, 0], posterior["f_mean"], A,
             "(a) Posterior mean of $f$", "Latent field")
    plot_map(fig, axes[0, 1], posterior["lam_mean"], A,
             r"(b) Posterior mean local $\lambda_i$", "Scale per cell")
    plot_map(fig, axes[1, 0], posterior["lam_sd"], A,
             r"(c) Posterior SD of $\lambda_i$", "SD", cmap="magma")
    plot_map(fig, axes[1, 1], posterior["f_lam_corr"], A,
             r"(d) Posterior corr$(f_i,\lambda_i)$", "Correlation",
             cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    save_figure(fig, "components_and_confounding")

    rate_maps = posterior["posterior_rate_samples"]
    if rate_maps:
        norm = PowerNorm(gamma=0.5, vmin=0.0,
                         vmax=max(float(x.max()) for x in rate_maps))
        fig, axes = plt.subplots(2, 2, figsize=(8.7, 7.7),
                                 constrained_layout=True)
        for j, (ax, rate) in enumerate(zip(axes.flat, rate_maps), start=1):
            plot_map(fig, ax, rate, A, f"Posterior draw {j}",
                     r"$\nu_i$", norm=norm)
        for ax in list(axes.flat)[len(rate_maps):]:
            ax.axis("off")
        save_figure(fig, "posterior_intensity_draws")

    total_trace = np.asarray(posterior["total_rate_trace"], dtype=float)
    mean_total = float(total_trace.mean())
    q025, q975 = np.quantile(total_trace, [0.025, 0.975])
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.hist(total_trace, bins=35, density=True, alpha=0.75,
            edgecolor="white", linewidth=0.5, label="Posterior")
    ax.axvline(mean_total, linewidth=1.6,
               label=f"Posterior mean = {mean_total:.1f}")
    ax.axvline(int(counts.sum()), color="0.15", linestyle="--", linewidth=1.6,
               label=f"Observed total = {int(counts.sum())}")
    ax.axvspan(q025, q975, alpha=0.12,
               label=f"95% interval [{q025:.0f}, {q975:.0f}]")
    ax.set_xlabel(r"Total intensity $T=\sum_i \nu_i$")
    ax.set_ylabel("Posterior density")
    ax.set_title("Posterior distribution of total intensity")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "total_intensity")

    # Posterior predictive diagnostics
    fields = ["total", "zero_cells", "cells_ge_10", "maximum"]
    names = ["Total events", "Empty cells", "Cells with >= 10 events",
             "Maximum cell count"]
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 7.7),
                             constrained_layout=True)
    for ax, key, name in zip(axes.flat, fields, names):
        predicted = posterior["predictive_stats"][key]
        observed_val = posterior["observed_stats"][key]
        ax.hist(predicted, bins=35, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(observed_val, color="0.15", linestyle="--", linewidth=1.6,
                   label=f"Observed = {observed_val}")
        ax.set_title(name)
        ax.set_xlabel("Replicated statistic")
        ax.set_ylabel("Frequency")
        ax.legend(frameon=False, loc="upper left")
    save_figure(fig, "posterior_predictive_checks")


# -----------------------------------------------------------------------------
# Results and entry point
# -----------------------------------------------------------------------------
def save_results(A, counts, initial_f, posterior, retained_events):

    np.savez_compressed(
        RESULTS_DIR / "posterior_summary.npz",
        counts=counts,
        initial_f_fixed_lambda_map=initial_f,
        f_mean=posterior["f_mean"], f_sd=posterior["f_sd"],
        lam_mean=posterior["lam_mean"], lam_sd=posterior["lam_sd"],
        rate_mean=posterior["rate_mean"], rate_sd=posterior["rate_sd"],
        f_lam_corr=posterior["f_lam_corr"],
        total_rate_trace=posterior["total_rate_trace"],
        **{f"ppc_{k}": v for k, v in posterior["predictive_stats"].items()},
    )

    trace = posterior["total_rate_trace"]
    q_low, q_med, q_high = np.quantile(trace, [0.025, 0.5, 0.975])
    with (RESULTS_DIR / "summary.txt").open("w") as out:
        print("SOUTH ITALY: INDEPENDENT LOCAL GAMMA SCALES", file=out)
        print(f"Events before declustering (raw file): {len(A.catalog.original_catalog)}", file=out)
        print(f"Retained and binned events: {retained_events}", file=out)
        print(f"Grid shape: {counts.shape}, cells: {counts.size}", file=out)
        print(f"Observed max cell count: {counts.max()}", file=out)
        print(f"Cell size: {BIN_SIZE_KM} km", file=out)
        print(f"Local lambda prior: Gamma(shape={LAM_PRIOR_SHAPE}, rate={LAM_PRIOR_RATE})", file=out)
        print(f"Prior mean lambda: {LAMBDA_MEAN}", file=out)
        print(f"Prior SD lambda: {LAMBDA_MEAN / np.sqrt(LAM_PRIOR_SHAPE):.4f}", file=out)
        print(f"Gaussian mean f: {float(np.mean(A.calc.prior_mean)):.6f}", file=out)
        print(f"Gaussian precision: rho={RHO_KM} km, v2={PRIOR_VARIANCE}, boundary={BOUNDARY}", file=out)
        print(f"Iterations: {N_ITER}, burn-in: {BURN_IN}, thin: {THIN}", file=out)
        print(f"Retained posterior draws: {posterior['n_kept']}", file=out)
        print(f"Observed total count: {counts.sum()}", file=out)
        print(f"Posterior total intensity mean: {trace.mean():.3f}", file=out)
        print(f"Posterior total intensity median: {q_med:.3f}", file=out)
        print(f"95% total-intensity interval: [{q_low:.3f}, {q_high:.3f}]", file=out)
        print("\nPOSTERIOR-PREDICTIVE CHECKS", file=out)
        for key, observed in posterior["observed_stats"].items():
            rep = posterior["predictive_stats"][key]
            lo, hi = np.quantile(rep, [0.025, 0.975])
            print(f"{key}: observed={observed}, predictive mean={rep.mean():.2f}, "
                  f"95% interval=[{lo:.1f}, {hi:.1f}]", file=out)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    A, counts, ny, nx, retained = prepare_catalogue_and_sampler()

    print("\n" + "=" * 65)
    print("SOUTH ITALY — LOCAL GAMMA INTENSITIES")
    print("=" * 65)
    print(f"Events before declustering: {len(A.catalog.original_catalog)}")
    print(f"Events retained and binned: {retained} / {counts.sum()}")
    print(f"Grid: {ny} x {nx} = {counts.size} cells, max count: {counts.max()}")
    print(f"Local lambda prior mean = {LAMBDA_MEAN}, alpha = {LAM_PRIOR_SHAPE}, "
          f"delta = {LAM_PRIOR_RATE}")
    print(f"Latent Gaussian prior mean f = {float(np.mean(A.calc.prior_mean)):.4f}")
    print(f"rho = {RHO_KM} km, v2 = {PRIOR_VARIANCE}, boundary = {BOUNDARY}")

    initial_f = initial_field_from_fixed_lambda_map(A)
    posterior = run_posterior(A, counts, initial_f)

    print(f"Retained posterior samples: {posterior['n_kept']}")
    trace = posterior["total_rate_trace"]
    low, med, high = np.quantile(trace, [0.025, 0.5, 0.975])
    print(f"Observed N: {counts.sum()}")
    print(f"Posterior total intensity mean: {trace.mean():.3f}")
    print(f"Posterior total intensity median: {med:.3f}")
    print(f"95% total intensity interval: [{low:.3f}, {high:.3f}]")

    make_figures(A, counts, posterior)
    save_results(A, counts, initial_f, posterior, retained)
    print("Results:", RESULTS_DIR)
    print("Figures:", FIGURES_DIR)
    print("Sampler log:", SAMPLER_LOG)


if __name__ == "__main__":
    main()
