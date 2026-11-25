"""Synthetic initialization-sensitivity experiment with INDEPENDENT LOCAL Gamma scales.

Model (Gamma shape/rate):
    f ~ N(m, Q^{-1})
    lambda_i iid ~ Gamma(alpha, delta)
    n_i | f_i, lambda_i ~ Poisson(lambda_i * sigmoid(f_i))

Three Gibbs chains share the SAME synthetic dataset but start from different
constant latent fields.  In every chain, ALL local scales are sampled anew.


"""

from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from scipy.special import expit
import arviz as az

import polyagammapoisson.covariance_kernels as ck
import polyagammapoisson.polyagammadensity as pgd


# ---------------------------------------------------------------------
# Configuration 
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "Plots" / "uniform_ergodicity_local_lambda" / "figures"

n, m = 20, 20
M = n * m
rho = 3.0
prior_mean = -1.0
v2 = 1.0

LAMBDA_MEAN = 12.0
LAMBDA_SHAPE = 1.0
LAMBDA_RATE = LAMBDA_SHAPE / LAMBDA_MEAN

N_ITER = 10000
SAMPLER_BURN_IN = 0      # Keep initial transient visible in the trace plots
SUMMARY_BURN_IN = 2000  # Only iterations >= 2000 enter posterior summaries
THIN = 1
DATA_SEED = 20261124
CHAIN_SEEDS = {"low": 1, "medium": 2, "high": 3}
INITIAL_MEANS = {"low": -3.0, "medium": prior_mean, "high": 3.0}
SAVE_PLOTS = True

CHAIN_COLORS = {"low": "tab:blue", "medium": "tab:orange", "high": "tab:green"}
CHAIN_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_figure(fig, name):
    if SAVE_PLOTS:
        for ext in ("pdf", "png"):
            path = PLOTS_DIR / f"uniform_ergodicity_{name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved figure: {path}")
        plt.close(fig)


def plot_grid(fig, ax, values, title, label, cmap, norm):
    im = ax.imshow(
        np.asarray(values).reshape(n, m),
        origin="lower", interpolation="nearest", aspect="equal",
        cmap=cmap, norm=norm,
    )
    ax.set_title(title)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    return im


def make_boxplot(ax, values, keys):
    bp = ax.boxplot(
        values, positions=np.arange(1, len(keys) + 1), widths=0.50,
        patch_artist=True, showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.4},
        whiskerprops={"color": "0.35", "linewidth": 1.0},
        capprops={"color": "0.35", "linewidth": 1.0},
    )
    for patch, key in zip(bp["boxes"], keys):
        patch.set_facecolor(CHAIN_COLORS[key])
        patch.set_alpha(0.50)
        patch.set_edgecolor("0.3")
    ax.set_xticks(np.arange(1, len(keys) + 1))
    ax.set_xticklabels([CHAIN_LABELS[k] for k in keys])


def make_model(precision):
    """Make a local-scale sampler, initially with all lambda_i at their prior mean."""
    model = pgd.PolyaGammaDensity2D(
        lam=np.full(M, LAMBDA_MEAN, dtype=float),
        local_lam=True,
        lam_prior_shape=LAMBDA_SHAPE,
        lam_prior_rate=LAMBDA_RATE,
        n=n, m=m,
    )
    model.set_prior_Gaussian(
        prior_mean=np.full(M, prior_mean),
        prior_precision=precision,
        sparse=True,
    )
    if not getattr(model, "local_lam", False):
        raise RuntimeError("This polyagammadensity.py does not implement local_lam=True")
    if np.asarray(model.lam).shape != (M,):
        raise RuntimeError("Expected one initial lambda per grid cell")
    return model


def generate_data(precision):
    # Separate the field-drawing RNG from the Gamma/Poisson RNG.
    np.random.seed(DATA_SEED)
    generative_model = make_model(precision)
    true_f = np.asarray(generative_model.random_prior_parameters(), dtype=float).ravel()
    rng = np.random.default_rng(DATA_SEED + 1)
    true_lambda = rng.gamma(
        shape=LAMBDA_SHAPE, scale=1.0 / LAMBDA_RATE, size=M,
    )
    true_rate = true_lambda * expit(true_f)
    counts = rng.poisson(true_rate)
    return true_f, true_lambda, true_rate, counts


def sample_chains(precision, counts):
    chains = {}
    for key, start in INITIAL_MEANS.items():
        print(f"Chain {key}: initial f={start:g}, seed={CHAIN_SEEDS[key]}")
        model = make_model(precision)
        model.set_nobs(counts)
        generator = model.sample_posterior(
            initial_f=np.full(M, start),
            n_iter=N_ITER, burn_in=SAMPLER_BURN_IN, thin=THIN,
            random_seed=CHAIN_SEEDS[key],
            sample_lam=True, return_lam=True,
        )
        mean_f_trace = []
        mean_lambda_trace = []
        total_rate_trace = []
        posterior_f_sum = np.zeros(M)
        posterior_lambda_sum = np.zeros(M)
        posterior_rate_sum = np.zeros(M)
        n_summary = 0
        count = 0
        log_path = PLOTS_DIR / f"sampler_{key}.log"

        with log_path.open("w") as log, redirect_stdout(log):
            for f_draw, lam_draw in generator:
                f = np.asarray(f_draw, dtype=float).ravel()
                lam = np.asarray(lam_draw, dtype=float).ravel()
                if f.shape != (M,) or lam.shape != (M,):
                    raise ValueError("The sampler must return f and a VECTOR of M local lambdas")
                if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
                    raise ValueError("Invalid local Gamma draw")

                rate = lam * expit(f)
                mean_f_trace.append(float(f.mean()))
                mean_lambda_trace.append(float(lam.mean()))
                total_rate_trace.append(float(rate.sum()))

                if count >= SUMMARY_BURN_IN:
                    posterior_f_sum += f
                    posterior_lambda_sum += lam
                    posterior_rate_sum += rate
                    n_summary += 1
                count += 1

        if n_summary < 2:
            raise RuntimeError("Too few retained draws; reduce SUMMARY_BURN_IN")
        chains[key] = {
            "mean_f_trace": np.asarray(mean_f_trace),
            "mean_lambda_trace": np.asarray(mean_lambda_trace),
            "total_rate_trace": np.asarray(total_rate_trace),
            "posterior_f_mean": posterior_f_sum / n_summary,
            "posterior_lambda_mean": posterior_lambda_sum / n_summary,
            "posterior_rate_mean": posterior_rate_sum / n_summary,
            "n_kept": n_summary,
        }
        print(f"  {count} iterations, {n_summary} posterior draws; log: {log_path.name}")
    return chains


def draw_fields(chains, true_f, true_rate, counts):
    field_arrays = [true_f] + [chains[k]["posterior_f_mean"] for k in INITIAL_MEANS]
    f_norm = Normalize(
        vmin=min(float(v.min()) for v in field_arrays),
        vmax=max(float(v.max()) for v in field_arrays),
    )
    rate_norm = PowerNorm(
        gamma=0.5, vmin=0.0,
        vmax=max(float(true_rate.max()), float(counts.max())),
    )
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.8), constrained_layout=True)
    plot_grid(fig, axs[0, 0], true_f, "(a) True latent field", r"$f_i$", "viridis", f_norm)
    plot_grid(fig, axs[0, 1], true_rate, "(b) True Poisson intensity", r"$\nu_i$", "viridis", rate_norm)
    plot_grid(fig, axs[0, 2], counts, "(c) Observed counts", "Count per cell", "viridis", rate_norm)
    for j, key in enumerate(INITIAL_MEANS):
        plot_grid(
            fig, axs[1, j], chains[key]["posterior_f_mean"],
            f"({chr(100 + j)}) Posterior mean of $f$: {key} init",
            r"$E[f_i\mid n]$", "viridis", f_norm,
        )
    save_figure(fig, "true_and_posterior_mean_fields")


def draw_local_lambda_maps(chains, true_lambda):
    maps = [true_lambda] + [chains[k]["posterior_lambda_mean"] for k in INITIAL_MEANS]
    norm = PowerNorm(
        gamma=0.65, vmin=0.0,
        vmax=max(float(arr.max()) for arr in maps),
    )
    fig, axs = plt.subplots(2, 2, figsize=(8.7, 7.7), constrained_layout=True)
    titles = [
        "(a) True local scales", "(b) Posterior mean: low init",
        "(c) Posterior mean: medium init", "(d) Posterior mean: high init",
    ]
    for ax, arr, title in zip(axs.flat, maps, titles):
        plot_grid(fig, ax, arr, title, r"$\lambda_i$", "viridis", norm)
    save_figure(fig, "local_lambda_maps")


def draw_intensity_maps(chains, true_rate):
    maps = [true_rate] + [chains[k]["posterior_rate_mean"] for k in INITIAL_MEANS]
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=max(float(x.max()) for x in maps))
    fig, axs = plt.subplots(2, 2, figsize=(8.7, 7.7), constrained_layout=True)
    titles = [
        "(a) True intensity", "(b) Posterior mean: low init",
        "(c) Posterior mean: medium init", "(d) Posterior mean: high init",
    ]
    for ax, arr, title in zip(axs.flat, maps, titles):
        plot_grid(fig, ax, arr, title, r"$\nu_i$", "viridis", norm)
    save_figure(fig, "local_intensity_maps")


def draw_trace(chains, key, ylabel, title, filename, reference_lines):
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    for label, chain in chains.items():
        vals = chain[key]
        ax.plot(
            np.arange(1, len(vals) + 1), vals,
            color=CHAIN_COLORS[label], lw=0.55, alpha=0.52,
            label=f"{CHAIN_LABELS[label]} initialization", rasterized=True,
        )
    for value, color, ls, caption in reference_lines:
        ax.axhline(value, color=color, ls=ls, lw=1.6, label=caption)
    ax.axvline(SUMMARY_BURN_IN, color="0.45", ls=":", lw=1.2, label="Summary burn-in")
    ax.set(xlabel="Iteration", ylabel=ylabel, title=title)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, filename)


def draw_box(chains, key, ylabel, title, filename, reference_lines):
    keys = list(INITIAL_MEANS)
    values = [chains[k][key][SUMMARY_BURN_IN:] for k in keys]
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    make_boxplot(ax, values, keys)
    for value, color, ls, caption in reference_lines:
        ax.axhline(value, color=color, ls=ls, lw=1.6, label=caption)
    ax.set(xlabel="Chain initialization", ylabel=ylabel, title=title)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, filename)


def save_results(chains, true_f, true_lambda, true_rate, counts):
    output = {
        "true_f": true_f,
        "true_lambda": true_lambda,
        "true_rate": true_rate,
        "counts": counts,
        "summary_burn_in": SUMMARY_BURN_IN,
    }
    for k in INITIAL_MEANS:
        for key, val in chains[k].items():
            output[f"{k}_{key}"] = val
    np.savez_compressed(PLOTS_DIR / "local_lambda_synthetic_results.npz", **output)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    precision = ck.precision_matern(n, m, rho=rho, v2=v2, boundary="symmetric")
    true_f, true_lambda, true_rate, counts = generate_data(precision)
    print("Independent local-Gamma synthetic experiment")
    print(f"Grid: {n} x {m} ({M} cells)")
    print(f"Gamma(shape={LAMBDA_SHAPE:g}, rate={LAMBDA_RATE:g})")
    print(f"True mean local lambda: {true_lambda.mean():.3f}; prior mean: {LAMBDA_MEAN:g}")
    print(f"Observed count total: {counts.sum()}; true intensity total: {true_rate.sum():.2f}")
    chains = sample_chains(precision, counts)

    draw_fields(chains, true_f, true_rate, counts)
    draw_local_lambda_maps(chains, true_lambda)
    draw_intensity_maps(chains, true_rate)

    draw_trace(
        chains, "mean_f_trace", r"$\bar f=M^{-1}\sum_i f_i$",
        "Trace of the mean latent field", "trace_mean_f",
        [(prior_mean, "0.45", "--", "Prior mean"),
         (float(true_f.mean()), "0.15", "-.", "True spatial mean")],
    )
    draw_trace(
        chains, "mean_lambda_trace", r"$\bar\lambda=M^{-1}\sum_i \lambda_i$",
        "Trace of the mean local scale", "trace_mean_lambda",
        [(float(true_lambda.mean()), "0.15", "--", "True mean local scale")],
    )
    draw_trace(
        chains, "total_rate_trace", r"$T=\sum_i \lambda_i\sigma(f_i)$",
        "Trace of the total intensity", "trace_total_rate",
        [(float(true_rate.sum()), "0.15", "--", "True total intensity"),
         (float(counts.sum()), "tab:red", ":", "Observed total count")],
    )

    draw_box(
        chains, "mean_f_trace", r"$\bar f=M^{-1}\sum_i f_i$",
        "Posterior distribution of the mean latent field", "box_mean_f",
        [(float(true_f.mean()), "0.15", "--", "True spatial mean")],
    )
    draw_box(
        chains, "mean_lambda_trace", r"$\bar\lambda=M^{-1}\sum_i\lambda_i$",
        "Posterior distribution of mean local scale", "box_mean_lambda",
        [(float(true_lambda.mean()), "0.15", "--", "True mean local scale")],
    )
    draw_box(
        chains, "total_rate_trace", r"$T=\sum_i \lambda_i\sigma(f_i)$",
        "Posterior total intensity", "box_mean_total_rate",
        [(float(true_rate.sum()), "0.15", "--", "True total intensity"),
         (float(counts.sum()), "tab:red", ":", "Observed total count")],
    )
    save_results(chains, true_f, true_lambda, true_rate, counts)
    print(f"Figures and traces: {PLOTS_DIR}")
    if not SAVE_PLOTS:
        plt.show()

    
    # -----------------------------------------------------------
    # MCMC convergence diagnostics
    # -----------------------------------------------------------

    quantities = {
        "mean_f": "mean_f_trace",
        "mean_lambda": "mean_lambda_trace",
        "total_intensity": "total_rate_trace",
    }

    # Combine the three chains, removing burn-in
    posterior = {
        name: np.stack([
            chains[label][trace][SUMMARY_BURN_IN:]
            for label in ["low", "medium", "high"]
        ])
        for name, trace in quantities.items()
    }

    # Convert to ArviZ format
    try:
        idata = az.from_dict({"posterior": posterior})
    except TypeError:
        idata = az.from_dict(posterior=posterior)

    # Calculate R-hat, ESS and MCSE
    diagnostics = az.summary(
        idata,
        kind="diagnostics"
    )

    print("\nMCMC convergence diagnostics:")
    print(diagnostics.to_string())

    # Save results to a CSV file
    diagnostics.to_csv(
        PLOTS_DIR / "mcmc_convergence_diagnostics.csv"
    )



if __name__ == "__main__":
    main()
