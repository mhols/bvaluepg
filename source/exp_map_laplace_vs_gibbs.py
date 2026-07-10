from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sps

import polyagammadensity as pgd
import covariance_kernels as ck
import syntheticdata as sd


# ---------------------------------------------------------------------
# Compatibility patch for scikit-sparse / CHOLMOD
# ---------------------------------------------------------------------
# Some scikit-sparse versions do not accept lower=True.
# polyagammadensity.py calls cholesky(A, lower=True), so we patch only
# inside this experiment script and do not modify polyagammadensity.py.
_original_cholmod_cholesky = pgd.cholesky

def cholmod_cholesky_compat(A, *args, **kwargs):
    kwargs.pop("lower", None)
    return _original_cholmod_cholesky(A, *args, **kwargs)

pgd.cholesky = cholmod_cholesky_compat


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_truth(estim, n, m, low, high, square_size):
    """
    Synthetic truth: one high-intensity square in a low background.
    """
    lambda_true_img = sd.single_square(
        n=n,
        m=m,
        nn=square_size,
        a=low,
        b=high,
    )
    lambda_true = estim.image_to_scanorder(lambda_true_img)
    f_true = estim.f_from_field(lambda_true)
    return lambda_true, f_true


def build_prior(estim, lambda_true, n, m, rho, v2):
    """
    Neutral prior mean: inverse-link of the global average intensity.
    Sparse precision prior: ck.precision_matern.
    """
    mean_intensity = float(np.mean(lambda_true))
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

    return prior_mean, prior_precision


def fit_map(estim, niter=1000):
    """
    MAP estimate using existing methods from polyagammadensity.py.
    """
    f0 = estim.first_guess_estimator()
    f_map = estim.max_logposterior_estimator(
        f0=f0,
        method="TNC",
        niter=niter,
    )
    lambda_map = estim.field_from_f(f_map)
    return f_map, lambda_map


def laplace_precision_at_map(estim, f_map, jitter=1e-8):
    """
    Build the Laplace precision matrix around the MAP estimate.

    Negative log posterior:
        U(f) = - log p(n | f) + 0.5 (f - mu)^T Q (f - mu)

    For independent Poisson observations:
        log p(n_i | f_i) = n_i log(lambda_i(f_i)) - lambda_i(f_i)

    Therefore the diagonal likelihood contribution to the Hessian of U is:
        lambda_i''(f_i) - n_i * (log lambda_i)''(f_i)

    We use the derivative methods already provided by the density class.
    """
    nobs = estim.nobs

    d2_lambda = estim.second_derivate_field_from_f(f_map)
    d2_log_lambda = estim.second_derivative_log_field_from_f(f_map)

    diag_lik = d2_lambda - nobs * d2_log_lambda

    # Numerical safety. The exact Hessian at the MAP should be positive,
    # but small negative values can occur from numerical precision.
    diag_lik = np.maximum(diag_lik, jitter)

    H = (estim.prior_precision + sps.diags(diag_lik, format="csc")).tocsc()

    return H, diag_lik


def sample_laplace_intensity(estim, f_map, n_samples, seed):
    """
    Draw samples from the Laplace approximation:
        f | n approx N(f_map, H^{-1})
    where H is the Hessian / posterior precision at the MAP.

    Uses CHOLMOD and the sparse Cholesky helper methods already present in
    polyagammadensity.py.
    """
    rng = np.random.default_rng(seed)

    H, diag_lik = laplace_precision_at_map(estim, f_map)
    factor = pgd.cholesky(H)

    samples_f = np.empty((n_samples, estim.nbins), dtype=float)

    for s in range(n_samples):
        z = rng.normal(size=estim.nbins)
        eps = pgd.Density.apply_cholesky_sparse_inverse_T(factor, z)
        samples_f[s] = f_map + eps

    samples_lambda = estim.field_from_f(samples_f)

    return samples_f, samples_lambda, diag_lik


def sample_gibbs_intensity(
    estim,
    f_map,
    n_iter,
    burn_in,
    thin,
    seed,
):
    """
    Draw exact posterior samples using the existing Pólya-Gamma sampler.
    """
    samples_f = []

    for f_sample in estim.sample_posterior(
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        initial_f=f_map,
        random_seed=seed,
    ):
        samples_f.append(f_sample.copy())

    samples_f = np.asarray(samples_f)
    samples_lambda = estim.field_from_f(samples_f)

    return samples_f, samples_lambda


def summarize_samples(lambda_true, samples_lambda):
    """
    Posterior mean, posterior sd, credible intervals, coverage.
    """
    mean = samples_lambda.mean(axis=0)
    sd = samples_lambda.std(axis=0)

    q025 = np.quantile(samples_lambda, 0.025, axis=0)
    q975 = np.quantile(samples_lambda, 0.975, axis=0)

    covered = (lambda_true >= q025) & (lambda_true <= q975)
    coverage = float(np.mean(covered))

    interval_width = float(np.mean(q975 - q025))

    return {
        "mean": mean,
        "sd": sd,
        "q025": q025,
        "q975": q975,
        "coverage_95": coverage,
        "mean_interval_width_95": interval_width,
    }


def save_image(ax, estim, values, title, vmin=None, vmax=None):
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    img = estim.scanorder_to_image(values).T
    im = ax.imshow(img, vmin=vmin, vmax=vmax)
    return im


def make_seed_figure(
    outdir,
    estim,
    lambda_true,
    nobs,
    lambda_map,
    laplace_summary,
    gibbs_summary,
    lam,
    seed,
):
    """
    Figure comparing truth, observations, MAP, Laplace, Gibbs, uncertainty.
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    vmin = 0.0
    vmax = lam

    panels = [
        (axes[0, 0], lambda_true, "True intensity", vmin, vmax),
        (axes[0, 1], nobs, "Observed counts", vmin, vmax),
        (axes[0, 2], lambda_map, "MAP estimate", vmin, vmax),
        (axes[0, 3], laplace_summary["mean"], "Laplace mean", vmin, vmax),
        (axes[1, 0], gibbs_summary["mean"], "Gibbs posterior mean", vmin, vmax),
        (axes[1, 1], laplace_summary["sd"], "Laplace SD", None, None),
        (axes[1, 2], gibbs_summary["sd"], "Gibbs posterior SD", None, None),
        (
            axes[1, 3],
            np.abs(gibbs_summary["mean"] - lambda_true),
            "Abs. Gibbs mean error",
            None,
            None,
        ),
    ]

    for ax, values, title, lo, hi in panels:
        im = save_image(ax, estim, values, title, vmin=lo, vmax=hi)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fig.savefig(outdir / f"fig_exp02_seed_{seed}.png", dpi=300)
    fig.savefig(outdir / f"fig_exp02_seed_{seed}.pdf")
    plt.close(fig)


def run_one_seed(args, seed, make_figure=False):
    """
    Run one synthetic dataset and compare:
        raw counts,
        MAP,
        Laplace approximation,
        exact Gibbs posterior.
    """
    np.random.seed(seed)

    estim = pgd.PolyaGammaDensity2D(
        n=args.n,
        m=args.m,
        lam=args.lam,
    )

    lambda_true, f_true = make_truth(
        estim=estim,
        n=args.n,
        m=args.m,
        low=args.low,
        high=args.high,
        square_size=args.square_size,
    )

    # Generate observations
    nobs = estim.random_events_from_field(lambda_true)
    estim.set_data(nobs)

    # Prior
    prior_mean, prior_precision = build_prior(
        estim=estim,
        lambda_true=lambda_true,
        n=args.n,
        m=args.m,
        rho=args.rho,
        v2=args.v2,
    )

    # MAP
    t0 = time.time()
    f_map, lambda_map = fit_map(estim, niter=args.map_niter)
    map_time = time.time() - t0

    # Laplace approximation
    t0 = time.time()
    laplace_f, laplace_lambda, diag_lik = sample_laplace_intensity(
        estim=estim,
        f_map=f_map,
        n_samples=args.n_laplace_samples,
        seed=seed + 10_000,
    )
    laplace_time = time.time() - t0

    laplace_summary = summarize_samples(lambda_true, laplace_lambda)

    # Exact Gibbs posterior
    t0 = time.time()
    gibbs_f, gibbs_lambda = sample_gibbs_intensity(
        estim=estim,
        f_map=f_map,
        n_iter=args.gibbs_n_iter,
        burn_in=args.gibbs_burn_in,
        thin=args.gibbs_thin,
        seed=seed + 20_000,
    )
    gibbs_time = time.time() - t0

    gibbs_summary = summarize_samples(lambda_true, gibbs_lambda)

    # Metrics
    metrics = {
        "seed": seed,
        "grid_n": args.n,
        "grid_m": args.m,
        "lambda_low": args.low,
        "lambda_high": args.high,
        "lambda_max_sigmoid": args.lam,
        "rho": args.rho,
        "v2": args.v2,
        "rmse_raw_counts": rmse(nobs, lambda_true),
        "rmse_map": rmse(lambda_map, lambda_true),
        "rmse_laplace_mean": rmse(laplace_summary["mean"], lambda_true),
        "rmse_gibbs_mean": rmse(gibbs_summary["mean"], lambda_true),
        "coverage_laplace_95": laplace_summary["coverage_95"],
        "coverage_gibbs_95": gibbs_summary["coverage_95"],
        "mean_sd_laplace": float(np.mean(laplace_summary["sd"])),
        "mean_sd_gibbs": float(np.mean(gibbs_summary["sd"])),
        "mean_width_laplace_95": laplace_summary["mean_interval_width_95"],
        "mean_width_gibbs_95": gibbs_summary["mean_interval_width_95"],
        "n_laplace_samples": args.n_laplace_samples,
        "n_gibbs_samples": gibbs_lambda.shape[0],
        "map_time_seconds": map_time,
        "laplace_time_seconds": laplace_time,
        "gibbs_time_seconds": gibbs_time,
    }

    if make_figure:
        make_seed_figure(
            outdir=Path(args.outdir),
            estim=estim,
            lambda_true=lambda_true,
            nobs=nobs,
            lambda_map=lambda_map,
            laplace_summary=laplace_summary,
            gibbs_summary=gibbs_summary,
            lam=args.lam,
            seed=seed,
        )

        np.savez_compressed(
            Path(args.outdir) / f"arrays_exp02_seed_{seed}.npz",
            lambda_true=lambda_true,
            f_true=f_true,
            nobs=nobs,
            f_map=f_map,
            lambda_map=lambda_map,
            laplace_mean=laplace_summary["mean"],
            laplace_sd=laplace_summary["sd"],
            laplace_q025=laplace_summary["q025"],
            laplace_q975=laplace_summary["q975"],
            gibbs_mean=gibbs_summary["mean"],
            gibbs_sd=gibbs_summary["sd"],
            gibbs_q025=gibbs_summary["q025"],
            gibbs_q975=gibbs_summary["q975"],
        )

    return metrics


def aggregate_metrics(df):
    """
    Mean and standard deviation over random seeds.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "seed"]

    rows = []

    for col in numeric_cols:
        rows.append(
            {
                "quantity": col,
                "mean": float(df[col].mean()),
                "sd": float(df[col].std(ddof=1)) if len(df) > 1 else 0.0,
            }
        )

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 2: MAP vs Laplace vs Pólya-Gamma Gibbs posterior."
    )

    parser.add_argument("--outdir", type=str, default="results/exp02_map_laplace_vs_gibbs")

    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--low", type=float, default=1.0)
    parser.add_argument("--high", type=float, default=4.0)
    parser.add_argument("--square-size", type=int, default=32)

    parser.add_argument("--lam", type=float, default=5.0)
    parser.add_argument("--rho", type=float, default=4.0)
    parser.add_argument("--v2", type=float, default=0.5)

    parser.add_argument("--map-niter", type=int, default=1000)

    parser.add_argument("--n-laplace-samples", type=int, default=800)

    parser.add_argument("--gibbs-n-iter", type=int, default=5000)
    parser.add_argument("--gibbs-burn-in", type=int, default=1000)
    parser.add_argument("--gibbs-thin", type=int, default=5)

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of repeated synthetic datasets. Use 20 for the paper table.",
    )
    parser.add_argument("--first-seed", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))

    all_metrics = []

    for i, seed in enumerate(seeds):
        print("=" * 80)
        print(f"Running seed {seed} ({i + 1}/{len(seeds)})")
        print("=" * 80)

        metrics = run_one_seed(
            args=args,
            seed=seed,
            make_figure=(i == 0),
        )

        all_metrics.append(metrics)

        # Save progress after every seed
        df_partial = pd.DataFrame(all_metrics)
        df_partial.to_csv(outdir / "metrics_by_seed.csv", index=False)

        print(pd.DataFrame([metrics]).T)

    df = pd.DataFrame(all_metrics)
    df.to_csv(outdir / "metrics_by_seed.csv", index=False)

    summary = aggregate_metrics(df)
    summary.to_csv(outdir / "metrics_summary.csv", index=False)

    print("\nSummary over seeds:")
    print(summary)

    print(f"\nSaved results in: {outdir.resolve()}")


if __name__ == "__main__":
    main()