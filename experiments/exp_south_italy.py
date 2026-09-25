"""
Simple South Italy earthquake experiment with fixed lambda.

Model
-----
    n_i | f_i ~ Poisson(lambda * sigmoid(f_i))
    f ~ N(m, Q^{-1})

This is intentionally a small real-data example:
- lambda is fixed;
- one MCMC chain;
- no global-shift move;
- no multiple-chain diagnostics;
- no sensitivity or posterior-predictive suite.

The catalogue selection, declustering, binning, coordinates, and coastlines are
taken from source/catalog/catalog.py.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.special import expit
import italy_data


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import polyagammapoisson.catalog.catalog as catalog
import polyagammapoisson.covariance_kernels as ck
import polyagammapoisson.polyagammadensity as pgd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BIN_SIZE_KM = 5.0
MAG_MIN = 2.5
DECLUSTER_ETA0 = -6.2

LAMBDA = 100.0

RHO_KM = 40.0
PRIOR_VARIANCE = 6.0
PRIOR_MEAN = -5.3
BOUNDARY = "symmetric"

MAP_NITER = 3000
N_ITER = 10
BURN_IN = 3
THIN = 1
RANDOM_SEED = 101
N_POSTERIOR_SAMPLES_TO_PLOT = 4

RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "south_italy_simple_fixed_lambda"
FIGURES_DIR = REPO_ROOT / "experiments" / "figures" / "south_italy_simple_fixed_lambda"
SAMPLER_LOG = RESULTS_DIR / "sampler.log"


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


# ---------------------------------------------------------------------
# Small plotting utilities
# ---------------------------------------------------------------------

def save_figure(fig, name):
    pdf = FIGURES_DIR / f"{name}.pdf"
    png = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {pdf}")


def add_coastlines(ax, cat):
    for poly in cat.coastlines:
        lon, lat = poly[:, 0], poly[:, 1]
        x, y = cat.coordinates.lonlat_to_rotated_xy(lon, lat)
        ax.plot(x, y, color="white", linewidth=0.65, zorder=3)

    xmin, xmax, ymin, ymax = cat.extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")


def vector_to_image(x, nbinx, nbiny):
    return np.asarray(x).reshape(nbinx, nbiny, order="C").T


def plot_map(ax, x, cat, nbinx, nbiny, title, cbar_label, cmap="viridis", norm=None):
    im = ax.imshow(
        vector_to_image(x, nbinx, nbiny),
        extent=cat.extent,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        zorder=1,
    )
    add_coastlines(ax, cat)
    ax.set_title(title)
    ax.set_xlabel(r"$x$ [km]")
    ax.set_ylabel(r"$y$ [km]")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)




def prepare_catalogue():
    # SicilyCalabria already applies the fixed rotated-coordinate spatial cut.
    cat = italy_data.SicilyCalabria(catalog=italy_data.CATALOGS["INGV"], BINSIZE=BIN_SIZE_KM)
    n_before = len(cat.catDataFrame)

    # Keep non-child events according to the NND declustering rule.
    cat.filter_decluster(Mc=MAG_MIN, f_eta_0=DECLUSTER_ETA0)
    n_after = len(cat.catDataFrame)

    binned = cat.binning_count
    counts = np.asarray(binned["counts"], dtype=int)
    nbinx = int(binned["nbinx"])
    nbiny = int(binned["nbiny"])

    if counts.shape != (nbinx, nbiny):
        raise ValueError(
            f"Count-grid shape {counts.shape} does not match ({nbinx}, {nbiny})."
        )

    return cat, counts, nbinx, nbiny, n_before, n_after


def prepare_sampler(counts, nbinx, nbiny):
    prior_precision = ck.precision_matern(
        n=nbinx,
        m=nbiny,
        rho=RHO_KM / BIN_SIZE_KM,
        v2=PRIOR_VARIANCE,
        boundary=BOUNDARY,
    )
    prior_mean = np.full(nbinx * nbiny, PRIOR_MEAN, dtype=float)

    sampler = pgd.PolyaGammaDensity2D(
        prior_precision=prior_precision,
        prior_mean=prior_mean,
        sparse=True,
        lam=LAMBDA,
        n=nbinx,
        m=nbiny,
        seed=RANDOM_SEED,
    )
    sampler.set_nobs(counts.ravel(order="C"))
    return sampler




def compute_map(sampler):
    print(f"\nComputing MAP with fixed lambda={LAMBDA:g}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Method TNC does not use Hessian information")
        warnings.filterwarnings("ignore", message="Unknown solver options")
        f_map = sampler.max_logposterior_estimator(niter=MAP_NITER)

    f_map = np.asarray(f_map, dtype=float)
    rate_map = LAMBDA * expit(f_map)
    return f_map, rate_map


def run_posterior(sampler, initial_f):
    nbins = initial_f.size

    f_mean = np.zeros(nbins)
    f_M2 = np.zeros(nbins)
    rate_mean = np.zeros(nbins)
    rate_M2 = np.zeros(nbins)

    total_rate_trace = []
    posterior_rate_samples = []

    n_expected = len(range(BURN_IN, N_ITER, THIN))
    sample_stride = max(1, n_expected // N_POSTERIOR_SAMPLES_TO_PLOT)

    generator = sampler.sample_posterior(
        n_iter=N_ITER,
        burn_in=BURN_IN,
        thin=THIN,
        initial_f=initial_f,
        random_seed=RANDOM_SEED,
        sample_lam=False,       # lambda stays fixed
        return_lam=False,
    )

    print(
        f"Running one chain: {N_ITER} iterations, {BURN_IN} burn-in, "
        f"lambda fixed at {LAMBDA:g}."
    )

    count = 0
    with open(SAMPLER_LOG, "w") as log_handle, redirect_stdout(log_handle):
        for f in generator:
            f = np.asarray(f, dtype=float)
            rate = LAMBDA * expit(f)
            count += 1

            delta_f = f - f_mean
            f_mean += delta_f / count
            f_M2 += delta_f * (f - f_mean)

            delta_rate = rate - rate_mean
            rate_mean += delta_rate / count
            rate_M2 += delta_rate * (rate - rate_mean)

            total_rate_trace.append(float(np.sum(rate)))

            if (
                count % sample_stride == 0
                and len(posterior_rate_samples) < N_POSTERIOR_SAMPLES_TO_PLOT
            ):
                posterior_rate_samples.append(rate.copy())

    if count < 2:
        raise RuntimeError("Too few retained posterior samples.")

    return {
        "n_kept": count,
        "f_mean": f_mean,
        "f_sd": np.sqrt(f_M2 / (count - 1)),
        "rate_mean": rate_mean,
        "rate_sd": np.sqrt(rate_M2 / (count - 1)),
        "total_rate_trace": np.asarray(total_rate_trace),
        "posterior_rate_samples": posterior_rate_samples,
    }



def make_main_figure(cat, counts, nbinx, nbiny, posterior):
    observed = counts.ravel(order="C")
    rate_mean = posterior["rate_mean"]
    rate_sd = posterior["rate_sd"]

    # Same scale for observed counts and posterior mean.
    intensity_norm = PowerNorm(
        gamma=0.5,
        vmin=0.0,
        vmax=max(float(observed.max()), float(rate_mean.max())),
    )
    sd_norm = PowerNorm(
        gamma=0.7,
        vmin=0.0,
        vmax=float(rate_sd.max()),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)

    plot_map(
        axes[0], observed, cat, nbinx, nbiny,
        "(a) Observed counts", "Count per cell",
        cmap="viridis", norm=intensity_norm,
    )
    plot_map(
        axes[1], rate_mean, cat, nbinx, nbiny,
        "(b) Posterior mean intensity", r"$E[\nu_i\mid n]$",
        cmap="viridis", norm=intensity_norm,
    )
    plot_map(
        axes[2], rate_sd, cat, nbinx, nbiny,
        "(c) Posterior SD", r"$\mathrm{SD}(\nu_i\mid n)$",
        cmap="magma", norm=sd_norm,
    )

    save_figure(fig, "south_italy_simple_main")
    plt.close(fig)


def make_map_comparison(cat, rate_map, posterior, nbinx, nbiny):
    rate_mean = posterior["rate_mean"]
    norm = PowerNorm(
        gamma=0.5,
        vmin=0.0,
        vmax=max(float(rate_map.max()), float(rate_mean.max())),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.7), constrained_layout=True)

    plot_map(
        axes[0], rate_map, cat, nbinx, nbiny,
        "(a) MAP intensity", r"$\nu_i^{\mathrm{MAP}}$",
        norm=norm,
    )
    plot_map(
        axes[1], rate_mean, cat, nbinx, nbiny,
        "(b) Posterior mean intensity", r"$E[\nu_i\mid n]$",
        norm=norm,
    )

    save_figure(fig, "south_italy_simple_map_vs_posterior")
    plt.close(fig)


def make_total_intensity_figure(total_count, posterior):
    total_intensity = np.asarray(
        posterior["total_rate_trace"],
        dtype=float,
    )

    posterior_mean = float(np.mean(total_intensity))
    q025, q975 = np.quantile(
        total_intensity,
        [0.025, 0.975],
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    ax.hist(
        total_intensity,
        bins=35,
        density=True,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
        label="Posterior",
    )

    ax.axvline(
        posterior_mean,
        linewidth=1.6,
        label=rf"Posterior mean = {posterior_mean:.1f}",
    )

    ax.axvline(
        total_count,
        color="0.15",
        linestyle="--",
        linewidth=1.6,
        label=rf"Observed total = {total_count}",
    )

    ax.axvspan(
        q025,
        q975,
        alpha=0.12,
        label=rf"95\% interval [{q025:.0f}, {q975:.0f}]",
    )

    ax.set_xlabel(r"Total intensity $T=\sum_i \nu_i$")
    ax.set_ylabel("Posterior density")
    ax.set_title("Posterior distribution of total intensity")

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "south_italy_simple_total_intensity",
    )
    plt.close(fig)


def make_sample_figure(cat, posterior, nbinx, nbiny):
    samples = posterior["posterior_rate_samples"]
    if not samples:
        return

    norm = PowerNorm(
        gamma=0.5,
        vmin=0.0,
        vmax=max(float(x.max()) for x in samples),
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.7, 7.7), constrained_layout=True)

    for j, (ax, sample) in enumerate(zip(axes.ravel(), samples), start=1):
        plot_map(
            ax, sample, cat, nbinx, nbiny,
            f"Posterior draw {j}", r"$\nu_i$",
            norm=norm,
        )

    save_figure(fig, "south_italy_simple_samples")
    plt.close(fig)


# ---------------------------------------------------------------------
# Numerical output
# ---------------------------------------------------------------------

def save_results(counts, f_map, rate_map, posterior, n_before, n_after):
    np.savez_compressed(
        RESULTS_DIR / "posterior_summary.npz",
        counts=counts,
        f_map=f_map,
        rate_map=rate_map,
        f_mean=posterior["f_mean"],
        f_sd=posterior["f_sd"],
        rate_mean=posterior["rate_mean"],
        rate_sd=posterior["rate_sd"],
        total_rate_trace=posterior["total_rate_trace"],
    )

    trace = posterior["total_rate_trace"]
    with open(RESULTS_DIR / "summary.txt", "w") as handle:
        print("SOUTH ITALY FIXED-LAMBDA EXPERIMENT", file=handle)
        print("=" * 60, file=handle)
        print(f"Events before declustering : {n_before}", file=handle)
        print(f"Events after declustering  : {n_after}", file=handle)
        print(f"Grid shape                 : {counts.shape}", file=handle)
        print(f"Observed total N           : {int(counts.sum())}", file=handle)
        print(f"Fixed lambda               : {LAMBDA:g}", file=handle)
        print(f"rho [km]                   : {RHO_KM:g}", file=handle)
        print(f"Prior variance parameter   : {PRIOR_VARIANCE:g}", file=handle)
        print(f"Prior mean                 : {PRIOR_MEAN:g}", file=handle)
        print(f"MCMC iterations            : {N_ITER}", file=handle)
        print(f"Burn-in                    : {BURN_IN}", file=handle)
        print(f"Retained samples           : {posterior['n_kept']}", file=handle)
        print(f"MAP total intensity        : {rate_map.sum():.6f}", file=handle)
        print(f"Posterior mean total rate  : {trace.mean():.6f}", file=handle)
        print(f"Posterior SD total rate    : {trace.std(ddof=1):.6f}", file=handle)




def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cat, counts, nbinx, nbiny, n_before, n_after = prepare_catalogue()
    total_count = int(counts.sum())

    print("\n" + "=" * 60)
    print("SOUTH ITALY EXPERIMENT")
    print("=" * 60)
    print(f"Events before declustering : {n_before}")
    print(f"Events after declustering  : {n_after}")
    print(f"Grid shape                 : {counts.shape}")
    print(f"Grid cells                 : {counts.size}")
    print(f"Observed total N           : {total_count}")
    print(f"Maximum cell count         : {int(counts.max())}")
    print("-" * 60)
    print(f"Fixed lambda               : {LAMBDA:g}")
    print(f"rho                         : {RHO_KM:g} km")
    print(f"Prior variance parameter   : {PRIOR_VARIANCE:g}")
    print(f"Prior mean                 : {PRIOR_MEAN:g}")
    print("=" * 60)

    sampler = prepare_sampler(counts, nbinx, nbiny)
    f_map, rate_map = compute_map(sampler)
    posterior = run_posterior(sampler, f_map)

    print(f"Retained posterior draws   : {posterior['n_kept']}")
    total_intensity = posterior["total_rate_trace"]
    q025, q50, q975 = np.quantile(
        total_intensity,
        [0.025, 0.5, 0.975],
    )

    print(f"Posterior mean total rate  : {total_intensity.mean():.3f}")
    print(f"Posterior median total rate: {q50:.3f}")
    print(f"95% posterior interval     : [{q025:.3f}, {q975:.3f}]")
    print(f"Observed total count       : {total_count}")
    print(f"Low-level sampler log      : {SAMPLER_LOG}")

    make_main_figure(cat, counts, nbinx, nbiny, posterior)
    make_map_comparison(cat, rate_map, posterior, nbinx, nbiny)
    make_sample_figure(cat, posterior, nbinx, nbiny)
    make_total_intensity_figure(total_count, posterior)
    save_results(counts, f_map, rate_map, posterior, n_before, n_after)

    print("\nSOUTH ITALY experiment completed.")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
