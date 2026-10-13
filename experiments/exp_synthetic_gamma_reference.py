"""Simple Gamma-scale reference run on the synthetic bars."""

from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"

# Internal imports expect this package name
spec = importlib.util.spec_from_file_location(
    "polyagammapoisson", SOURCE_DIR / "__init__.py",
    submodule_search_locations=[str(SOURCE_DIR)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)

from polyagammapoisson import covariance_kernels as ck
from polyagammapoisson import polyagammadensity as pgd


# Gamma parameters are shape and rate

TRUTH_FILE = REPO_ROOT / "data/synthetic/bars_synthetic_catalog_truth.npz"
OUTPUT_DIR = REPO_ROOT / "results/synthetic_gamma_reference"

LAM = 12.0
LAM_PRIOR_SHAPE = 1.0
LAM_PRIOR_RATE = 1.0
RHO = 6.0
PRIOR_MEAN = -3.7
PRIOR_VARIANCE = 4.0
BOUNDARY = "symmetric"
N_ITER = 500
BURN_IN = 100
THIN = 1
RANDOM_SEED = 0  


with np.load(TRUTH_FILE) as data:
    counts = data["counts"].astype(int)
    truth = data["lambda_true"].astype(float)

n, m = counts.shape
prior_mean = np.full(counts.size, PRIOR_MEAN)
precision = ck.precision_matern(
    n=n, m=m, rho=RHO, v2=PRIOR_VARIANCE, boundary=BOUNDARY,
)

model = pgd.PolyaGammaDensity2D(
    lam=LAM, lam_prior_shape=LAM_PRIOR_SHAPE,
    lam_prior_rate=LAM_PRIOR_RATE, n=n, m=m,
)

# 2D constructor does not forward Gaussian prior arguments
# :/
# da muss ich noch mal schauen, ob ich die Prior-Parameter nicht direkt in den Konstruktor packen kann
model.set_prior_Gaussian(
    prior_mean=prior_mean, prior_precision=precision, sparse=True,
)
model.set_data(counts.ravel())

f_samples = []
lam_samples = []
for f, lam in model.sample_posterior(
    n_iter=N_ITER, burn_in=BURN_IN, thin=THIN,
    initial_f=prior_mean, random_seed=RANDOM_SEED,
    sample_lam=True, return_lam=True,
):
    f_samples.append(f.copy())
    lam_samples.append(lam)

f_samples = np.array(f_samples)
lam_samples = np.array(lam_samples)
# Use each field's own Lambda, rather than the final model.lam.
rate_samples = lam_samples[:, None] * expit(f_samples)
rate_mean = rate_samples.mean(axis=0).reshape(n, m)
rate_sd = rate_samples.std(axis=0).reshape(n, m)
iterations = np.arange(BURN_IN, N_ITER, THIN)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUTPUT_DIR / "posterior_samples.npz",
    f_samples=f_samples, lam_samples=lam_samples, rate_samples=rate_samples,
    rate_mean=rate_mean, rate_sd=rate_sd, iterations=iterations,
    counts=counts, lambda_true=truth,
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(iterations, lam_samples)
ax.set(xlabel="Gibbs iteration", ylabel="Lambda", title="Gamma scale trace")
fig.tight_layout()
# fig.savefig(OUTPUT_DIR / "lambda_trace.png", dpi=160)
plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
rate_max = max(truth.max(), rate_mean.max())
for ax, values, title, vmax in zip(
    axes, (truth, counts, rate_mean, rate_sd),
    ("True rate", "Observed counts", "Posterior mean rate", "Posterior SD rate"),
    (rate_max, None, rate_max, None),
):
    im = ax.imshow(values, origin="lower", vmin=0, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.tight_layout()
# fig.savefig(OUTPUT_DIR / "rate_summary.png", dpi=160)
plt.show()
plt.close(fig)

print("Saved results to", OUTPUT_DIR)
print("Retained samples:", len(lam_samples))
print("Lambda mean:", lam_samples.mean())
print("Rate RMSE:", np.sqrt(np.mean((rate_mean - truth) ** 2)))
