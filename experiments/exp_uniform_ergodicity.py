"""
Convergence experiment with different initial values for the latent field f. 
The goal is to show that the Gibbs sampler converges to the same posterior distribution
regardless of the initial value of f.


"""


from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np
from scipy.special import expit


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

PLOTS_DIR = REPO_ROOT / "Plots" / "uniform_ergodicity" / "figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

import polyagammapoisson.catalog.catalog as catalog
import polyagammapoisson.covariance_kernels as ck
import polyagammapoisson.polyagammadensity as pgd
import polyagammapoisson.syntheticdata as sd

# ----------------------------------------------------------------------
# Figure style
# ----------------------------------------------------------------------

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

CHAIN_COLORS = {"low": "tab:blue", "medium": "tab:orange", "high": "tab:green"}
CHAIN_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}


def save_figure(fig, name):
    """Save the same figure as a vector PDF and a 300-dpi PNG."""
    if save_plots:
        for ext in ("pdf", "png"):
            target = PLOTS_DIR / f"uniform_ergodicity_{name}.{ext}"
            fig.savefig(target, bbox_inches="tight", dpi=300)
            print(f"Saved figure: {target}")
        plt.close(fig)


def plot_field(fig, ax, values, title, label, cmap, norm):
    """A clean map panel with a labelled colorbar."""
    im = ax.imshow(
        np.asarray(values).reshape(n, m),
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    return im


def style_axis(ax):
    ax.grid(axis="y", color="0.85", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def make_boxplot(ax, values, labels):
    """Style the comparison boxplots consistently across all diagnostics."""
    bp = ax.boxplot(
        values,
        positions=np.arange(1, len(labels) + 1),
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.4},
        whiskerprops={"color": "0.35", "linewidth": 1.0},
        capprops={"color": "0.35", "linewidth": 1.0},
    )
    for patch, key in zip(bp["boxes"], labels):
        patch.set_facecolor(CHAIN_COLORS[key])
        patch.set_alpha(0.50)
        patch.set_edgecolor("0.3")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels([CHAIN_LABELS[key] for key in labels])
    style_axis(ax)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

n = 20
m = 20
M = n*m
rho = 3
prior_mean = -1.0
v2 = 1.0
true_lam = 12.0
lambda_prior_shape = 1.0
lambda_prior_rate = lambda_prior_shape / true_lam

n_iter = 10000
burn_in = 0
box_plot_burn_in = 2000

save_plots = True


initial_means = {"low": -3.0, "medium": prior_mean, "high": 3.0}

# ----------------------------------------------------------------------
# Create the synthetic data
# ----------------------------------------------------------------------

precision = ck.precision_matern(
    n,
    m,
    rho=rho,
    v2=v2,
    boundary="symmetric",
)

model = pgd.PolyaGammaDensity2D(
    lam=true_lam,
    n=n,
    m=m,
)

model.set_prior_Gaussian(
    prior_mean=np.full(M, prior_mean),
    prior_precision=precision,
    sparse=True,
)  


#----------------------------------------------------------------------
# Generate the true latent field f and the corresponding Poisson rate
#----------------------------------------------------------------------

true_f = model.random_prior_parameters()
true_rate = model.field_from_f(true_f)
data = model.random_events_from_field(true_rate)
model.set_nobs(data.ravel())
observations = model.nobs.reshape(n, m)




# ----------------------------------------------------------------------
# Run the Gibbs sampler with different initial values for f
# ----------------------------------------------------------------------

chains = {}

chains_seeds = {"low": 1, "medium": 2, "high": 3}

for labels, initial_value in initial_means.items():
    print(f"Running Gibbs sampler in chain {labels} with initial f = {initial_value} ...")

    estim = pgd.PolyaGammaDensity2D(
        lam=true_lam,
        lam_prior_shape=lambda_prior_shape,
        lam_prior_rate=lambda_prior_rate,
        n=n,
        m=m,
    )   

    estim.set_prior_Gaussian(
        prior_mean=np.full(M, prior_mean),
        prior_precision=precision,
        sparse=True,
    )
    estim.set_nobs(data.ravel())

    initial_f = np.full( M, initial_value)

    mean_f_trace = []
    lambda_trace = []
    total_rate_trace = []

    posterior_f_sum = np.zeros(M)
    posterior_rate_sum = np.zeros(M)

    n_posterior = 0
    nsamples = 0

    for iteration, (f, lambda_sample) in enumerate(estim.sample_posterior(
            initial_f= initial_f, 
            n_iter = n_iter, 
            burn_in = burn_in, 
            thin = 1, 
            random_seed=chains_seeds[labels],
            sample_lam=True,
            return_lam=True,)):

       
        mean_f_trace.append(np.mean(f))
        # Use the lambda belonging to THIS draw (global scale in this experiment).
        rate = np.asarray(lambda_sample, dtype=float) * expit(f)
        lambda_trace.append(lambda_sample)
        total_rate_trace.append(np.sum(rate))   
    
        if iteration >= box_plot_burn_in:
            posterior_f_sum += f
            posterior_rate_sum += rate
            n_posterior += 1



        nsamples += 1

    # These sums contain only draws after box_plot_burn_in.
    if n_posterior == 0:
        raise ValueError("box_plot_burn_in must be less than n_iter - burn_in")
    posterior_rate_mean = posterior_rate_sum / n_posterior
    posterior_f_mean = posterior_f_sum / n_posterior


    chains[labels] = {
        "mean_f_trace": mean_f_trace,
        "total_rate_trace": total_rate_trace,
        "lambda_trace": lambda_trace,
        "posterior_f_mean": posterior_f_mean,
        "posterior_rate_mean": posterior_rate_mean,
        "nsamples": nsamples,
    }   

    print(f"Chain {labels} completed. Total samples: {nsamples}")

# ----------------------------------------------------------------------
# 1. Synthetic truth, observations, and estimates for each initialization
# ----------------------------------------------------------------------

field_arrays = [true_f] + [chains[key]["posterior_f_mean"] for key in initial_means]
f_norm = Normalize(
    vmin=min(float(np.min(arr)) for arr in field_arrays),
    vmax=max(float(np.max(arr)) for arr in field_arrays),
)
count_norm = PowerNorm(
    gamma=0.5,
    vmin=0.0,
    vmax=max(float(np.max(true_rate)), float(np.max(observations))),
)

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), constrained_layout=True)

plot_field(fig, axes[0, 0], true_f,
           "(a) True latent field", r"$f_i$", "coolwarm", f_norm)
plot_field(fig, axes[0, 1], true_rate,
           "(b) True Poisson intensity", r"$\nu_i$", "viridis", count_norm)
plot_field(fig, axes[0, 2], observations,
           "(c) Observed counts", "Events/cell", "viridis", count_norm)

for j, key in enumerate(initial_means):
    plot_field(
        fig, axes[1, j], chains[key]["posterior_f_mean"],
        f"({chr(ord('d') + j)}) Posterior mean of $f$: {key} init",
        r"$E[f_i\mid n]$", "coolwarm", f_norm,
    )

save_figure(fig, "true_and_posterior_mean_fields")


# ----------------------------------------------------------------------
# 2. Traces from the three chains
# ----------------------------------------------------------------------

def make_trace_figure(key, ylabel, title, filename, reference_lines):
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    for label, chain in chains.items():
        ax.plot(
            np.arange(1, len(chain[key]) + 1), chain[key],
            color=CHAIN_COLORS[label], linewidth=0.75, alpha=0.7,
            label=f"{CHAIN_LABELS[label]} initialization",
        )
    for y, color, ls, ref_label in reference_lines:
        ax.axhline(y, color=color, linestyle=ls, linewidth=1.5,
                   label=ref_label)
    ax.axvline(box_plot_burn_in, color="0.5", linewidth=1.1,
               linestyle=":", label="Summary burn-in")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, filename)


make_trace_figure(
    "mean_f_trace", r"$M^{-1}\sum_i f_i$",
    "Trace of the mean latent field", "trace_mean_f",
    [(prior_mean, "0.45", "--", "Prior mean"),
     (float(np.mean(true_f)), "0.15", "-.", "True spatial mean")],
)

make_trace_figure(
    "total_rate_trace", r"$T=\sum_i\nu_i$",
    "Trace of the total intensity", "trace_total_rate",
    [(float(np.sum(true_rate)), "0.15", "--", "True total intensity"),
     (float(np.sum(data)), "tab:red", ":", "Observed total count")],
)


# ----------------------------------------------------------------------
# 3. Posterior comparisons across the three initial values
# ----------------------------------------------------------------------

labels = list(chains)
mean_f_box = [np.asarray(chains[key]["mean_f_trace"][box_plot_burn_in:])
              for key in labels]
mean_rate_box = [np.asarray(chains[key]["total_rate_trace"][box_plot_burn_in:])
                 for key in labels]
lambda_box = [np.asarray(chains[key]["lambda_trace"][box_plot_burn_in:])
              for key in labels]


def make_box_figure(values, ylabel, title, filename, reference_lines):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    make_boxplot(ax, values, labels)
    for y, color, ls, ref_label in reference_lines:
        ax.axhline(y, color=color, linestyle=ls, linewidth=1.5,
                   label=ref_label)
    ax.set_xlabel("Chain initialization")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, filename)


make_box_figure(
    mean_f_box, r"$M^{-1}\sum_i f_i$",
    "Posterior mean latent-field statistic", "box_mean_f",
    [(float(np.mean(true_f)), "0.15", "--", "True spatial mean")],
)

make_box_figure(
    mean_rate_box, r"$T=\sum_i\nu_i$",
    "Posterior total intensity", "box_mean_total_rate",
    [(float(np.sum(true_rate)), "0.15", "--", "True total intensity"),
     (float(np.sum(data)), "tab:red", ":", "Observed total count")],
)

make_box_figure(
    lambda_box, r"$\lambda$",
    "Posterior global intensity scale", "box_mean_lambda",
    [(true_lam, "0.15", "--", "True global scale")],
)

if not save_plots:
    plt.show()
