from __future__ import annotations

"""
Parameter influence experiment.

This script creates diagnostic plots for the main distributional building
blocks used in the project:
- link functions f -> Poisson rate
- induced rate distributions from Gaussian latent priors
- Poisson likelihood shapes as functions of f
- Poisson count distributions as functions of the rate
- spatial Gaussian prior covariance parameters

The script is intentionally lightweight and does not run Gibbs sampling or
mixture precomputation. It only needs numpy and matplotlib.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

try:
    from polyagammadensity import inv_sigmoid as logit
    from polyagammadensity import sigmoid, softplus
except Exception:
    def sigmoid(x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        pos = x >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out

    def logit(p: np.ndarray | float) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.log(p / (1.0 - p))

    def softplus(x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

try:
    from exp_mix_explink import gaussian_pdf as normal_pdf
    from exp_mix_explink import safe_exp as exp_link
except Exception:
    def exp_link(f: np.ndarray | float) -> np.ndarray:
        return np.exp(np.clip(f, -745.0, 700.0))

    def normal_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
        z = (x - mean) / sigma
        return np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma)

try:
    from covariance_kernels import spatial_covariance_gaussian
    from covariance_kernels import spatial_covariance_matern_1_2
    from covariance_kernels import spatial_covariance_matern_2_3
    from covariance_kernels import spatial_covariance_matern_3_5
except Exception:
    spatial_covariance_gaussian = None
    spatial_covariance_matern_1_2 = None
    spatial_covariance_matern_2_3 = None
    spatial_covariance_matern_3_5 = None


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results" / "parameter_influence"


def sigmoid_rate(f: np.ndarray, lam: float = 10.0, beta0: float = 0.0, beta1: float = 1.0) -> np.ndarray:
    return lam * sigmoid(beta0 + beta1 * f)


def softplus_rate(f: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    kappa = float(kappa)
    return softplus(kappa * f) / kappa


def poisson_pmf(n: np.ndarray, rate: float) -> np.ndarray:
    n = np.asarray(n, dtype=int)
    rate = max(float(rate), 1e-300)
    logp = n * math.log(rate) - rate - np.vectorize(math.lgamma)(n + 1)
    return np.exp(logp)


def normalize_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    area = float(np.trapz(y, x))
    if not np.isfinite(area) or area <= 0.0:
        return y
    return y / area


def poisson_likelihood_over_f(f: np.ndarray, n: int, rate: np.ndarray) -> np.ndarray:
    rate = np.maximum(rate, 1e-300)
    logy = int(n) * np.log(rate) - rate
    logy -= np.nanmax(logy)
    return normalize_area(f, np.exp(logy))


def density_rate_sigmoid(rate_grid: np.ndarray, mu: float, v2: float, lam: float) -> np.ndarray:
    rate_grid = np.asarray(rate_grid, dtype=float)
    sigma = math.sqrt(float(v2))
    eps = 1e-12
    s = np.clip(rate_grid / lam, eps, 1.0 - eps)
    f = logit(s)
    deriv = lam * s * (1.0 - s)
    y = normal_pdf(f, mu, sigma) / np.maximum(np.abs(deriv), 1e-300)
    y[(rate_grid <= 0.0) | (rate_grid >= lam)] = 0.0
    return normalize_area(rate_grid, y)


def density_rate_exp(rate_grid: np.ndarray, mu: float, v2: float) -> np.ndarray:
    rate_grid = np.asarray(rate_grid, dtype=float)
    sigma = math.sqrt(float(v2))
    r = np.maximum(rate_grid, 1e-300)
    y = normal_pdf(np.log(r), mu, sigma) / r
    y[rate_grid <= 0.0] = 0.0
    return normalize_area(rate_grid, y)


def density_rate_softplus(rate_grid: np.ndarray, mu: float, v2: float, kappa: float) -> np.ndarray:
    rate_grid = np.asarray(rate_grid, dtype=float)
    sigma = math.sqrt(float(v2))
    kappa = float(kappa)
    r = np.maximum(rate_grid, 1e-12)
    f = np.log(np.expm1(np.clip(kappa * r, 1e-12, 700.0))) / kappa
    deriv = sigmoid(kappa * f)
    y = normal_pdf(f, mu, sigma) / np.maximum(np.abs(deriv), 1e-300)
    y[rate_grid <= 0.0] = 0.0
    return normalize_area(rate_grid, y)


def covariance_matrix_from_formula(n: int, m: int, rho: float, v2: float, kernel: str) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    coords = np.column_stack([yy.ravel(), xx.ravel()])
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    d = np.sqrt(d2)

    if kernel == "gaussian":
        corr = np.exp(-d2 / (2.0 * rho * rho))
    elif kernel == "matern12":
        corr = np.exp(-d / rho)
    elif kernel == "matern32":
        a = math.sqrt(3.0) * d / rho
        corr = (1.0 + a) * np.exp(-a)
    elif kernel == "matern52":
        a = math.sqrt(5.0) * d / rho
        corr = (1.0 + a + 5.0 * d2 / (3.0 * rho * rho)) * np.exp(-a)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    cov = v2 * corr
    cov += 1e-8 * v2 * np.eye(n * m)
    return cov


def covariance_matrix(n: int, m: int, rho: float, v2: float, kernel: str) -> np.ndarray:
    imported_kernels: dict[str, Callable[[int, int, float, float], np.ndarray] | None] = {
        "gaussian": spatial_covariance_gaussian,
        "matern12": spatial_covariance_matern_1_2,
        "matern32": spatial_covariance_matern_2_3,
        "matern52": spatial_covariance_matern_3_5,
    }
    kernel_func = imported_kernels.get(kernel)
    if kernel_func is not None:
        return np.asarray(kernel_func(n, m, rho, v2), dtype=float)
    return covariance_matrix_from_formula(n, m, rho, v2, kernel)


def savefig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def plot_link_functions(outdir: Path) -> None:
    f = np.linspace(-8.0, 8.0, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for lam in [2, 5, 10, 20]:
        ax.plot(f, sigmoid_rate(f, lam=lam), label=f"lam={lam}")
    ax.set_title("Sigmoid link: lam controls upper bound")
    ax.set_xlabel("f")
    ax.set_ylabel("rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for beta1 in [-1.0, 0.5, 1.0, 2.0]:
        ax.plot(f, sigmoid_rate(f, lam=10, beta1=beta1), label=f"beta1={beta1}")
    ax.set_title("Sigmoid link: beta1 controls slope and direction")
    ax.set_xlabel("f")
    ax.set_ylabel("rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for beta0 in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        ax.plot(f, sigmoid_rate(f, lam=10, beta0=beta0), label=f"beta0={beta0}")
    ax.set_title("Sigmoid link: beta0 shifts the curve horizontally")
    ax.set_xlabel("f")
    ax.set_ylabel("rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(f, exp_link(f), label="exp(f)")
    for kappa in [0.5, 1.0, 2.0, 5.0]:
        ax.plot(f, softplus_rate(f, kappa=kappa), label=f"softplus k={kappa}")
    ax.set_ylim(0, 20)
    ax.set_title("Exp vs softplus links")
    ax.set_xlabel("f")
    ax.set_ylabel("rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    savefig(fig, outdir / "01_link_functions.png")


def plot_link_derivatives(outdir: Path) -> None:
    f = np.linspace(-8.0, 8.0, 1000)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    for lam in [2, 5, 10, 20]:
        y = lam * sigmoid(f) * sigmoid(-f)
        ax.plot(f, y, label=f"lam={lam}")
    ax.set_title("d/df lam*sigmoid(f)")
    ax.set_xlabel("f")
    ax.set_ylabel("derivative")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for kappa in [0.5, 1.0, 2.0, 5.0]:
        ax.plot(f, sigmoid(kappa * f), label=f"k={kappa}")
    ax.set_title("d/df softplus(kf)/k")
    ax.set_xlabel("f")
    ax.set_ylabel("derivative")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(f, exp_link(f), label="exp(f)")
    ax.set_ylim(0, 20)
    ax.set_title("d/df exp(f)")
    ax.set_xlabel("f")
    ax.set_ylabel("derivative")
    ax.legend()
    ax.grid(True, alpha=0.3)

    savefig(fig, outdir / "02_link_derivatives.png")


def plot_induced_rate_distributions(outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    r = np.linspace(1e-4, 9.999, 1500)
    for mu in [-2.0, 0.0, 2.0]:
        ax.plot(r, density_rate_sigmoid(r, mu=mu, v2=1.0, lam=10), label=f"mu={mu}")
    ax.set_title("Sigmoid-induced rate density: mu")
    ax.set_xlabel("rate")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for v2 in [0.1, 0.5, 1.0, 2.0]:
        ax.plot(r, density_rate_sigmoid(r, mu=0.0, v2=v2, lam=10), label=f"v2={v2}")
    ax.set_title("Sigmoid-induced rate density: v2")
    ax.set_xlabel("rate")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    r_exp = np.linspace(1e-4, 35.0, 1800)
    for v2 in [0.1, 0.5, 1.0, 2.0]:
        ax.plot(r_exp, density_rate_exp(r_exp, mu=1.0, v2=v2), label=f"v2={v2}")
    ax.set_title("Exp-induced rate density: v2")
    ax.set_xlabel("rate")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    r_soft = np.linspace(1e-4, 15.0, 1500)
    for kappa in [0.5, 1.0, 2.0, 5.0]:
        ax.plot(r_soft, density_rate_softplus(r_soft, mu=0.0, v2=1.0, kappa=kappa), label=f"k={kappa}")
    ax.set_title("Softplus-induced rate density: k")
    ax.set_xlabel("rate")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    savefig(fig, outdir / "03_induced_rate_distributions.png")


def plot_poisson_likelihoods(outdir: Path) -> None:
    f = np.linspace(-8.0, 8.0, 1600)
    n_values = [0, 1, 3, 10, 25]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    links = [
        ("sigmoid, lam=30", sigmoid_rate(f, lam=30.0)),
        ("softplus, k=1", softplus_rate(f, kappa=1.0)),
        ("exp", exp_link(f)),
    ]

    for ax, (title, rate) in zip(axes, links):
        for n in n_values:
            ax.plot(f, poisson_likelihood_over_f(f, n=n, rate=rate), label=f"n={n}")
        ax.set_title(title)
        ax.set_xlabel("f")
        ax.set_ylabel("normalized likelihood")
        ax.legend()
        ax.grid(True, alpha=0.3)

    savefig(fig, outdir / "04_poisson_likelihood_over_f.png")


def plot_softplus_likelihood_kappa(outdir: Path) -> None:
    f = np.linspace(-8.0, 15.0, 1800)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, n in zip(axes, [0, 3, 10]):
        for kappa in [0.5, 1.0, 2.0, 5.0]:
            rate = softplus_rate(f, kappa=kappa)
            ax.plot(f, poisson_likelihood_over_f(f, n=n, rate=rate), label=f"k={kappa}")
        ax.set_title(f"Softplus likelihood, n={n}")
        ax.set_xlabel("f")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("normalized likelihood")
    axes[-1].legend()

    savefig(fig, outdir / "05_softplus_kappa_likelihood.png")


def plot_poisson_count_distributions(outdir: Path) -> None:
    n = np.arange(0, 50)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for rate in [0.5, 1.0, 3.0, 10.0, 25.0]:
        ax.plot(n, poisson_pmf(n, rate), marker="o", markersize=3, label=f"rate={rate}")
    ax.set_title("Poisson count distribution")
    ax.set_xlabel("observed count n")
    ax.set_ylabel("probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    rate = np.linspace(0.01, 40.0, 1200)
    for obs in [0, 1, 3, 10, 25]:
        y = poisson_likelihood_over_f(rate, obs, rate)
        ax.plot(rate, y, label=f"n={obs}")
    ax.set_title("Poisson likelihood as function of rate")
    ax.set_xlabel("rate")
    ax.set_ylabel("normalized likelihood")
    ax.legend()
    ax.grid(True, alpha=0.3)

    savefig(fig, outdir / "06_poisson_count_and_rate_likelihood.png")


def plot_covariance_parameters(outdir: Path) -> None:
    d = np.linspace(0.0, 20.0, 500)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for rho in [1.0, 2.0, 4.0, 8.0]:
        ax.plot(d, np.exp(-d * d / (2.0 * rho * rho)), label=f"rho={rho}")
    ax.set_title("Gaussian correlation: rho")
    ax.set_xlabel("distance")
    ax.set_ylabel("correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    rho = 4.0
    kernels = {
        "matern12": np.exp(-d / rho),
        "matern32": (1.0 + math.sqrt(3.0) * d / rho) * np.exp(-math.sqrt(3.0) * d / rho),
        "matern52": (
            1.0 + math.sqrt(5.0) * d / rho + 5.0 * d * d / (3.0 * rho * rho)
        ) * np.exp(-math.sqrt(5.0) * d / rho),
        "gaussian": np.exp(-d * d / (2.0 * rho * rho)),
    }
    for name, corr in kernels.items():
        ax.plot(d, corr, label=name)
    ax.set_title("Kernel smoothness at fixed rho=4")
    ax.set_xlabel("distance")
    ax.set_ylabel("correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    savefig(fig, outdir / "07_covariance_correlation_curves.png")


def plot_prior_samples(outdir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = m = 24

    settings = [
        ("rho=1, v2=0.5", "gaussian", 1.0, 0.5),
        ("rho=4, v2=0.5", "gaussian", 4.0, 0.5),
        ("rho=4, v2=2.0", "gaussian", 4.0, 2.0),
        ("matern12, rho=4", "matern12", 4.0, 0.5),
        ("matern32, rho=4", "matern32", 4.0, 0.5),
        ("matern52, rho=4", "matern52", 4.0, 0.5),
    ]

    fields = []
    for _, kernel, rho, v2 in settings:
        cov = covariance_matrix(n, m, rho=rho, v2=v2, kernel=kernel)
        sample = rng.multivariate_normal(np.zeros(n * m), cov)
        fields.append(sample.reshape(n, m))

    vmin = min(float(np.min(x)) for x in fields)
    vmax = max(float(np.max(x)) for x in fields)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (title, _, _, _), field in zip(axes.ravel(), settings, fields):
        im = ax.imshow(field, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)

    savefig(fig, outdir / "08_spatial_prior_samples.png")


@dataclass(frozen=True)
class ExperimentConfig:
    outdir: Path = RESULTS_DIR
    seed: int = 0


def run(config: ExperimentConfig) -> None:
    config.outdir.mkdir(parents=True, exist_ok=True)
    plot_link_functions(config.outdir)
    plot_link_derivatives(config.outdir)
    plot_induced_rate_distributions(config.outdir)
    plot_poisson_likelihoods(config.outdir)
    plot_softplus_likelihood_kappa(config.outdir)
    plot_poisson_count_distributions(config.outdir)
    plot_covariance_parameters(config.outdir)
    plot_prior_samples(config.outdir, seed=config.seed)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for generated plots.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for prior samples.")
    args = parser.parse_args()
    return ExperimentConfig(outdir=args.outdir, seed=args.seed)


if __name__ == "__main__":
    run(parse_args())
