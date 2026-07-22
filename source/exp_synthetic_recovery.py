from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import polyagammadensity as pgd
import covariance_kernels as ck
import syntheticdata as sd

# ---------------------------------------------------------------------
# Compatibility patch for scikit-sparse / CHOLMOD
# ---------------------------------------------------------------------
# Some versions of sksparse.cholmod.cholesky accept lower=True,
# while others do not. polyagammadensity.py calls cholesky(A, lower=True).
# We patch only the imported module inside this experiment script.
_original_cholmod_cholesky = pgd.cholesky

def cholmod_cholesky_compat(A, *args, **kwargs):
    kwargs.pop("lower", None)
    return _original_cholmod_cholesky(A, *args, **kwargs)

pgd.cholesky = cholmod_cholesky_compat


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def save_image(ax, estim, values, title, vmin=None, vmax=None):
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    img = estim.scanorder_to_image(values).T
    im = ax.imshow(img, vmin=vmin, vmax=vmax)
    return im


def main():
    outdir = Path("results/exp01_synthetic_recovery")
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1. Experiment parameters
    # -------------------------
    seed_data = 1
    seed_sampler = 2

    n = 64
    m = 64

    lambda_low = 1.0
    lambda_high = 4.0
    square_size = 32

    lam = 5.0
    rho = 4.0
    v2 = 0.5

    n_iter = 500
    burn_in = 100
    thin = 5

    # -------------------------
    # 2. Initialize estimator
    # -------------------------
    estim = pgd.PolyaGammaDensity2D(n=n, m=m, lam=lam)

    # -------------------------
    # 3. Build synthetic truth
    # -------------------------
    lambda_true_img = sd.single_square(
        n=n,
        m=m,
        nn=square_size,
        a=lambda_low,
        b=lambda_high,
    )

    lambda_true = estim.image_to_scanorder(lambda_true_img)
    f_true = estim.f_from_field(lambda_true)

    # -------------------------
    # 4. Generate observations
    # -------------------------
    np.random.seed(seed_data)
    nobs = estim.random_events_from_field(lambda_true)
    estim.set_data(nobs)

    # -------------------------
    # 5. Set prior
    # -------------------------
    mean_intensity = np.mean(lambda_true)
    prior_mean = estim.f_from_field(mean_intensity * np.ones(n * m))

    prior_precision = ck.precision_matern(
        n=n,
        m=m,
        rho=rho,
        v2=v2,
    )

    estim.set_prior_Gaussian(
        prior_mean=prior_mean,
        prior_precision=prior_precision,
        sparse=True,
    )

    # -------------------------
    # 6. MAP estimate
    # -------------------------
    f0 = estim.first_guess_estimator()

    f_map = estim.max_logposterior_estimator(
        f0=f0,
        method="TNC",
        niter=1000,
    )

    lambda_map = estim.field_from_f(f_map)

    # -------------------------
    # 7. Posterior sampling
    # -------------------------
    samples_f = []

    for f_sample in estim.sample_posterior(
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        initial_f=f_map,
        random_seed=seed_sampler,
    ):
        samples_f.append(f_sample.copy())

    samples_f = np.asarray(samples_f)
    samples_lambda = estim.field_from_f(samples_f)

    # -------------------------
    # 8. Posterior summaries
    # -------------------------
    lambda_post_mean = samples_lambda.mean(axis=0)
    lambda_post_sd = samples_lambda.std(axis=0)
    lambda_post_q025 = np.quantile(samples_lambda, 0.025, axis=0)
    lambda_post_q975 = np.quantile(samples_lambda, 0.975, axis=0)

    covered = (
        (lambda_true >= lambda_post_q025)
        & (lambda_true <= lambda_post_q975)
    )
    coverage_95 = float(covered.mean())

    # -------------------------
    # 9. Metrics
    # -------------------------
    metrics = {
        "grid_n": n,
        "grid_m": m,
        "lambda_low": lambda_low,
        "lambda_high": lambda_high,
        "lambda_max_sigmoid": lam,
        "rho": rho,
        "v2": v2,
        "n_iter": n_iter,
        "burn_in": burn_in,
        "thin": thin,
        "n_samples": samples_lambda.shape[0],
        "rmse_raw_counts": rmse(nobs, lambda_true),
        "rmse_map": rmse(lambda_map, lambda_true),
        "rmse_posterior_mean": rmse(lambda_post_mean, lambda_true),
        "coverage_95": coverage_95,
        "mean_posterior_sd": float(np.mean(lambda_post_sd)),
        "mean_posterior_sd_low_region": float(
            np.mean(lambda_post_sd[lambda_true == lambda_low])
        ),
        "mean_posterior_sd_high_region": float(
            np.mean(lambda_post_sd[lambda_true == lambda_high])
        ),
    }

    pd.DataFrame([metrics]).to_csv(outdir / "metrics.csv", index=False)

    print(pd.DataFrame([metrics]).T)

    # -------------------------
    # 10. Main figure
    # -------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    vmin = 0.0
    vmax = lam

    im0 = save_image(
        axes[0, 0],
        estim,
        lambda_true,
        "True intensity",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = save_image(
        axes[0, 1],
        estim,
        nobs,
        "Observed counts",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = save_image(
        axes[0, 2],
        estim,
        lambda_map,
        "MAP estimate",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = save_image(
        axes[1, 0],
        estim,
        lambda_post_mean,
        "Posterior mean",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = save_image(
        axes[1, 1],
        estim,
        lambda_post_sd,
        "Posterior SD",
    )
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    abs_error = np.abs(lambda_post_mean - lambda_true)

    im5 = save_image(
        axes[1, 2],
        estim,
        abs_error,
        "Abs. posterior mean error",
    )
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    fig.savefig(outdir / "fig_synthetic_recovery.png", dpi=300)
    fig.savefig(outdir / "fig_synthetic_recovery.pdf")

    # -------------------------
    # 11. Save arrays
    # -------------------------
    np.savez_compressed(
        outdir / "arrays.npz",
        lambda_true=lambda_true,
        f_true=f_true,
        nobs=nobs,
        f_map=f_map,
        lambda_map=lambda_map,
        lambda_post_mean=lambda_post_mean,
        lambda_post_sd=lambda_post_sd,
        lambda_post_q025=lambda_post_q025,
        lambda_post_q975=lambda_post_q975,
        samples_f=samples_f,
        samples_lambda=samples_lambda,
    )

    plt.show()


if __name__ == "__main__":
    main()