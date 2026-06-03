"""
USGS earthquake grid (30x30) with model

    n_i ~ Poisson( softplus(f_i) )

using Gibbs with data augmentation via Gaussian-mixture approximation
of the pixelwise likelihood.

"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import syntheticdata as sd

from gibbs_softplus_mixture import (
    SoftplusMixtureDensity2D,
    gibbs_sampler, load_or_build_mix
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
COUNTS_PATH = DATA_DIR / "eq_counts_30x30.npy"




def main():
    # -------------------------
    # Load data
    # -------------------------
    counts = np.load(COUNTS_PATH)   # shape (30, 30)
    n, m = counts.shape
    N = n * m

    nobs = counts.ravel(order="C").astype(int)
    nmax_data = int(np.max(nobs))
    print("[data] nmax_data =", nmax_data)

    # -------------------------
    # Bucketing for mixture table
    # -------------------------
    nmax_mix = 180
    nobs_clip = np.minimum(nobs, nmax_mix).astype(int)

    frac_clipped = float(np.mean(nobs > nmax_mix))
    print(f"[data] nmax_mix={nmax_mix} | fraction clipped={frac_clipped:.3f}")

    # -------------------------
    # Prior choice
    # -------------------------
 
    lam0 = 10.0
    prior_mean = np.zeros(N) 
    prior_cov = sd.spatial_covariance_gaussian(n, m, rho=7, v2=lam0)

    # Density object 
    dens = SoftplusMixtureDensity2D(
        prior_mean=prior_mean,
        prior_covariance=prior_cov,
        mix=None,        
        n=n,
        m=m,
    )

    # set prior mean 
    dens.prior_mean = dens.f_from_field(lam0 * np.ones(N))
    dens.prior_covariance = prior_cov

    # attach data
    dens.set_data(nobs_clip)

    # -------------------------
    # Mixture approximation
    # -------------------------
    mix = load_or_build_mix(nmax_mix=nmax_mix, cache_dir=DATA_DIR)
    dens.mix = mix

    # -------------------------
    # Initialization
    # -------------------------
    try:
        initial_f = dens.first_guess_estimator()
        print("[init] using dens.first_guess_estimator()")
    except Exception as e:
        print(f"[init] first_guess_estimator failed; using prior mean. Reason: {e!r}")
        initial_f = dens.prior_mean.copy()

    # -------------------------
    # Gibbs sampling
    # -------------------------
    n_iter = 200
    burn_in = 100
    thin = 10

    samples = gibbs_sampler(
        dens,
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        initial_f=initial_f,
        random_seed=0,
    )

    # Posterior summaries
    f_est = samples.mean(axis=0)
    rate_est = np.mean([dens.field_from_f(s) for s in samples], axis=0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Plot 1: estimated latent field
    # -------------------------
    plt.figure(figsize=(5, 4))
    plt.title("Estimated latent field E[f]")
    dens.imshow(f_est)
    plt.colorbar()
    plt.tight_layout()

    out_f = RESULTS_DIR / "usgs_softplus_estimated_field_f.png"
    plt.savefig(out_f, dpi=180, bbox_inches="tight")
    plt.show()

    # -------------------------
    # Plot 2: estimated rate field
    # -------------------------
    plt.figure(figsize=(5, 4))
    plt.title("Estimated rate E[softplus(f)]")
    dens.imshow(rate_est)
    plt.colorbar()
    plt.tight_layout()

    out_rate = RESULTS_DIR / "usgs_softplus_estimated_rate.png"
    plt.savefig(out_rate, dpi=180, bbox_inches="tight")
    plt.show()

    # -------------------------
    # Plot 3: counts vs estimated field
    # -------------------------
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Observed counts (raw)")
    plt.imshow(counts, origin="lower")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Estimated latent field")
    dens.imshow(f_est)
    plt.colorbar()

    plt.tight_layout()
    out_field = RESULTS_DIR / "usgs_softplus_bucket_counts_vs_field.png"
    plt.savefig(out_field, dpi=180, bbox_inches="tight")
    plt.show()

    # -------------------------
    # Plot 4: counts vs estimated rate
    # -------------------------
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Observed counts (raw)")
    plt.imshow(counts, origin="lower")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Estimated rate E[softplus(f)]")
    dens.imshow(rate_est)
    plt.colorbar()

    plt.tight_layout()
    out_rate_compare = RESULTS_DIR / "usgs_softplus_bucket_counts_vs_rate.png"
    plt.savefig(out_rate_compare, dpi=180, bbox_inches="tight")
    plt.show()

    # -------------------------
    # Plot 5: PowerNorm comparison
    # -------------------------
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Observed counts (PowerNorm)")
    plt.imshow(counts, origin="lower", norm=PowerNorm(gamma=0.5))
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Estimated rate (PowerNorm)")
    plt.imshow(
        sd.scanorder_to_image(rate_est, n, m),
        origin="lower",
        norm=PowerNorm(gamma=0.5),
    )
    plt.colorbar()

    plt.tight_layout()
    out_power = RESULTS_DIR / "usgs_softplus_bucket_counts_vs_rate_powernorm.png"
    plt.savefig(out_power, dpi=180, bbox_inches="tight")
    plt.show()

    print("Done.")
    print("Saved:", out_f)
    print("Saved:", out_rate)
    print("Saved:", out_field)
    print("Saved:", out_rate_compare)
    print("Saved:", out_power)

    return samples, f_est, rate_est, counts, (n, m), (nmax_data, nmax_mix), frac_clipped


if __name__ == "__main__":
    main()