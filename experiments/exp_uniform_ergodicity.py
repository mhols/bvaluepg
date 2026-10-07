"""
Convergence experiment with different initial values for the latent field f. 
The goal is to show that the Gibbs sampler converges to the same posterior distribution
regardless of the initial value of f.


"""


from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

PLOTS_DIR = REPO_ROOT / "Plots" / "uniform_ergodicity" / "figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

import polyagammapoisson.catalog.catalog as catalog
import polyagammapoisson.covariance_kernels as ck
import polyagammapoisson.polyagammadensity as pgd
import polyagammapoisson.syntheticdata as sd

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def save_plot(plot_name):
    filename = (
        f"uniform_ergodicity_{plot_name}.png")

    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path}")


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
    total_rate_trace = []

    posterior_f_sum = np.zeros(M)
    posterior_rate_sum = np.zeros(M)

    nsamples = 0

    for f in estim.sample_posterior(
            initial_f= initial_f, 
            n_iter = n_iter, 
            burn_in = burn_in, 
            thin = 1, 
            random_seed=chains_seeds[labels]):


       
        mean_f_trace.append(np.mean(f))
        rate = estim.field_from_f(f)
        total_rate_trace.append(np.sum(rate))   
    
        posterior_f_sum += f
        posterior_rate_sum += rate


        nsamples += 1

    posterior_rate_mean = posterior_rate_sum / nsamples
    posterior_f_mean = posterior_f_sum / nsamples

    chains[labels] = {
        "mean_f_trace": mean_f_trace,
        "total_rate_trace": total_rate_trace,
        "posterior_f_mean": posterior_f_mean,
        "posterior_rate_mean": posterior_rate_mean,
        "nsamples": nsamples,
    }   

    print(f"Chain {labels} completed. Total samples: {nsamples}")

# ----------------------------------------------------------------------
# Plot the true latent field and the posterior mean latent field for each chain
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(11, 6))

axes[0, 0].imshow(true_f.reshape(n, m), origin="lower")
axes[0, 0].set_title("Latent field")

axes[0, 1].imshow(true_rate.reshape(n, m), origin="lower")
axes[0, 1].set_title("Poisson-Rate")

axes[0, 2].imshow(observations, origin="lower")
axes[0, 2].set_title("Observations")

axes[1, 0].imshow(chains["low"]["posterior_f_mean"].reshape(n, m), origin="lower")
axes[1, 0].set_title("Posterior mean f (low init)")

axes[1, 1].imshow(chains["medium"]["posterior_f_mean"].reshape(n, m), origin="lower")
axes[1, 1].set_title("Posterior mean f (medium init)")

axes[1, 2].imshow(chains["high"]["posterior_f_mean"].reshape(n, m), origin="lower")
axes[1, 2].set_title("Posterior mean f (high init)")

if save_plots:
    save_plot(plot_name="true_and_posterior_mean_fields")

# ----------------------------------------------------------------------
# Trace plot: spatial mean of f
# ----------------------------------------------------------------------

plt.figure(figsize=(9, 5))

for label, chain in chains.items():

    plt.plot(
        chain["mean_f_trace"],
        label=label,
        linewidth=1.0,
    )

plt.axhline(
    prior_mean,
    linestyle="--",
    color="red",
    label="Prior mean",
)

plt.axhline(
    np.mean(true_f),
    linestyle="--",
    color="black",
    label="True mean",
)

plt.xlabel("Iteration")
plt.ylabel(r"$M^{-1}\sum_i f_i$")
plt.title("Trace of the mean latent field")
plt.legend()

plt.tight_layout()
if save_plots:  
    save_plot(plot_name="trace_mean_f")



# -----------------------------------------------------------------------
# Trace plot: total number of events
# -----------------------------------------------------------------------

plt.figure(figsize=(9, 5))

for label, chain in chains.items():

    plt.plot(
        chain["total_rate_trace"],
        label=label,
        linewidth=1.0,
    )

plt.axhline(
    np.sum(true_rate),
    linestyle="--",
    color="black",
    label="True total intensity",
)

plt.axhline(
    np.sum(data),
    linestyle=":",
    color="red",
    label="Observed total count",
)

plt.xlabel("Iteration")
plt.ylabel(r"$\sum_i \lambda_i$")
plt.title("Trace of the total intensity")
plt.legend()    
plt.tight_layout()
if save_plots:
    save_plot(plot_name="trace_total_rate")



# ----------------------------------------------------------------------
# Box plots
# ----------------------------------------------------------------------

plt.figure(figsize=(9, 5))
labels = list(chains.keys())

mean_f_box = [np.array(chains[label]["mean_f_trace"][box_plot_burn_in:]) for label in labels]
mean_rate_box = [np.array(chains[label]["total_rate_trace"][box_plot_burn_in:]) for label in labels]

plt.boxplot(
    mean_f_box,
    tick_labels=labels,
    patch_artist=True,
)

plt.axhline(
    np.mean(true_f),
    linestyle="--",
    color="black",
    label="True mean",
)

plt.xlabel("Chain")
plt.ylabel(r"$M^{-1}\sum_i f_i$")
plt.title("Posterior distribution of the mean latent field")
plt.legend()
plt.tight_layout()
if save_plots:
    save_plot(plot_name="box_mean_f")




plt.figure(figsize=(9, 5))
plt.boxplot(
    mean_rate_box,
    tick_labels=labels,
    patch_artist=True,
)   

plt.axhline(
    np.sum(true_rate),
    linestyle="--",
    color="black",
    label="True total intensity",
)   

plt.axhline(
    np.sum(data),
    linestyle=":",
    color="red",
    label="Observed total count",
)
plt.xlabel("Chain")
plt.ylabel(r"$\sum_i \lambda_i$")
plt.title("Posterior distribution of the total intensity")
plt.legend()
plt.tight_layout()
if save_plots:
    save_plot(plot_name="box_mean_total_rate")
plt.show()  












