"""Gibbs sampling for Poisson observations with a softplus intensity
using a Gaussian-mixture approximation of the pixelwise likelihood.

Goal
----
We want samples from p(f | nobs) where
    f   ~ N(prior_mean, prior_cov)
    n_i ~ Poisson( softplus(f_i) )

We approximate each pixelwise likelihood p(n | f) by a Gaussian mixture in f:
    p(n | f) ∝ softplus(f)^n * exp(-softplus(f))
          ≈ sum_k w[n,k] * N(f | means[n,k], sigma[n]^2)

Then we introduce latent mixture indicators z_i (data augmentation):
    z_i | f_i, n_i  ∝ w[n_i,k] N(f_i | means[n_i,k], sigma[n_i]^2)
    f   | z, n      is Gaussian (prior Gaussian + diagonal Gaussian likelihood)

"""

from __future__ import annotations

import numpy as np
import os
import scipy.linalg as spla
import matplotlib.pyplot as plt


def gaussian_pdf(x: float | np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian N(x | mean, sigma^2), vectorized over mean."""
    z = (x - mean) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * sigma)


# =========================
# Density wrapper 
# =========================

class SoftplusMixtureDensity:
    """Minimal density container:

    Attributes
    ----------
    prior_mean : (N,) float
    prior_covariance : (N,N) float (SPD)
    mix : dict
        Mixture approximation dict with keys:
          - mix['nmax']
          - mix['means'][n]  (K_n,)
          - mix['sigma'][n]  float
          - mix['w'][n]      (K_n,) weights (sum to 1)
          - mix['active'][n] (K_active,) indices where w>thr
    nobs : (N,) int
        Observed counts.
    """

    def __init__(self, prior_mean: np.ndarray, prior_covariance: np.ndarray, mix: dict):
        self.prior_mean = np.asarray(prior_mean, dtype=float)
        self.prior_covariance = np.asarray(prior_covariance, dtype=float)
        self.mix = mix

        self._Lprior: np.ndarray | None = None
        self.nobs: np.ndarray | None = None

    @property
    def nbins(self) -> int:
        return int(self.prior_mean.shape[0])

    @property
    def Lprior(self) -> np.ndarray:
        """Cholesky factor of the prior covariance: C = L L^T."""
        if self._Lprior is None:
            self._Lprior = spla.cholesky(self.prior_covariance, lower=True)
        return self._Lprior

    def set_data(self, nobs: np.ndarray) -> None:
        nobs = np.asarray(nobs, dtype=int)
        if nobs.shape != (self.nbins,):
            raise ValueError("nobs must have shape (nbins,)")
        self.nobs = nobs


# =========================
# Step 1: sample z | f, n
# =========================

def sample_z_cond_f(f: np.ndarray, nobs: np.ndarray, mix: dict) -> np.ndarray:
    """Sample mixture indicators z.

    For each i:
        p(z_i=k | f_i, n_i) ∝ w[n_i,k] * N(f_i | means[n_i,k], sigma[n_i]^2)


    Returns
    -------
    z : (N,) int
        Indices into means[n_i] for each pixel.
    """
    N = int(f.shape[0])
    z = np.empty(N, dtype=int)

    for i in range(N):
        n_i = int(nobs[i])
        if n_i > mix["nmax"]:
            raise ValueError(f"count {n_i} exceeds precomputed nmax={mix['nmax']}")

        sigma = float(mix["sigma"][n_i])
        means = np.asarray(mix["means"][n_i], dtype=float)
        w = np.asarray(mix["w"][n_i], dtype=float)
        active = np.asarray(mix["active"][n_i], dtype=int)

        m_act = means[active]
        w_act = w[active]

        # log probabilities
        logp = np.log(w_act + 1e-300) + np.log(gaussian_pdf(f[i], m_act, sigma) + 1e-300)
        logp -= np.max(logp)
        p = np.exp(logp)
        p /= np.sum(p)

        k_local = np.random.choice(active.size, p=p)
        z[i] = int(active[k_local])

    return z


# =========================
# Step 2: sample f | z, n
# =========================

def sample_f_cond_z(
    z: np.ndarray,
    nobs: np.ndarray,
    prior_mean: np.ndarray,
    Lprior: np.ndarray,
    mix: dict,
) -> np.ndarray:
    """Sample the latent field f from its Gaussian conditional.

    Conditioned on z and nobs, the likelihood becomes a diagonal Gaussian:
        f_i ~ N(mu_i, s2_i)
    with
        mu_i = means[n_i][z_i]
        s2_i = sigma[n_i]^2

    Prior:
        f ~ N(prior_mean, C),  C = Lprior Lprior^T

    Posterior:
        Cov_post  = (C^{-1} + D^{-1})^{-1},  D = diag(s2)
        mean_post = Cov_post (C^{-1} prior_mean + D^{-1} mu)

    Sampling is done without forming C^{-1} explicitly, using
        Cov_post = L (I + L^T D^{-1} L)^{-1} L^T

    Returns
    -------
    f : (N,) float
        One sample from the conditional posterior.
    """
    N = int(prior_mean.shape[0])

    # Build mu and s2 from (n_i, z_i)
    mu = np.empty(N, dtype=float)
    s2 = np.empty(N, dtype=float)
    for i in range(N):
        n_i = int(nobs[i])
        mu[i] = float(mix["means"][n_i][int(z[i])])
        sigma = float(mix["sigma"][n_i])
        s2[i] = sigma * sigma

    dinv = 1.0 / (s2 + 1e-15)  # diagonal of D^{-1}

    L = Lprior

    # T = I + L^T D^{-1} L
    T = np.eye(N) + L.T @ (dinv[:, None] * L)

    # Cholesky factor of T
    cholT = spla.cholesky(T, lower=True)

    # Apply C^{-1} to a vector using the Cholesky of C
    def Cinv_dot(v: np.ndarray) -> np.ndarray:
        y = spla.solve_triangular(L, v, lower=True)
        x = spla.solve_triangular(L.T, y, lower=False)
        return x

    # b = C^{-1} prior_mean + D^{-1} mu
    b = Cinv_dot(prior_mean) + dinv * mu

    # mean = Cov_post * b = L T^{-1} L^T b
    Lt_b = L.T @ b
    y = spla.solve_triangular(cholT, Lt_b, lower=True)
    x = spla.solve_triangular(cholT.T, y, lower=False)  # x = T^{-1} L^T b
    mean = L @ x

    # Sample noise ~ N(0, Cov_post)
    # If u ~ N(0, I), then noise = L * (T^{-1/2} u).
    # Using cholT (T = cholT cholT^T): T^{-1/2} u = solve(cholT^T, u)
    u = np.random.normal(size=N)
    v = spla.solve_triangular(cholT.T, u, lower=False)  # v = T^{-1/2} u
    noise = L @ v

    return mean + noise


# =========================
# Gibbs sampler 
# =========================

def gibbs_sampler(
    dens: SoftplusMixtureDensity,
    n_iter: int,
    burn_in: int = 0,
    thin: int = 1,
    initial_f: np.ndarray | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """Run the Gibbs sampler.

    Returns
    -------
    f_samples : (n_keep, N)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if dens.nobs is None:
        raise ValueError("Call dens.set_data(nobs) before running the sampler.")

    N = dens.nbins

    # init
    if initial_f is None:
        f = dens.prior_mean.copy()
    else:
        f = np.asarray(initial_f, dtype=float).copy()
        if f.shape != (N,):
            raise ValueError("initial_f must have shape (nbins,)")

    n_keep = max(0, (n_iter - burn_in) // thin)
    f_samples = np.zeros((n_keep, N), dtype=float)

    idx = 0
    for it in range(n_iter):
        # Step 1: z | f, n
        z = sample_z_cond_f(f, dens.nobs, dens.mix)

        # Step 2: f | z, n
        f = sample_f_cond_z(z, dens.nobs, dens.prior_mean, dens.Lprior, dens.mix)

        if it >= burn_in and ((it - burn_in) % thin == 0):
            f_samples[idx] = f
            idx += 1

    return f_samples


# =========================
# =========================
# Mixture precomputation 
# =========================

from cvxopt import matrix as cvxmat, solvers as cvxsolvers
cvxsolvers.options["show_progress"] = False


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def poisson_like_unnormalized(t: np.ndarray, n: int) -> np.ndarray:
    lam = softplus(t)
    return (lam ** n) * np.exp(-lam)


def normalize_on_grid(y: np.ndarray) -> np.ndarray:
    s = float(np.sum(y))
    return y / (s + 1e-15)


def design_matrix(t_grid: np.ndarray, means: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_pdf(t_grid[:, None], means[None, :], sigma)


def solve_l1_lp(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    N, K = Phi.shape
    c = np.concatenate([np.zeros(K), np.ones(N)])

    G1 = np.hstack([Phi, -np.eye(N)])
    h1 = y.copy()

    G2 = np.hstack([-Phi, -np.eye(N)])
    h2 = -y.copy()

    G3 = np.hstack([-np.eye(K), np.zeros((K, N))])
    h3 = np.zeros(K)

    G4 = np.hstack([np.zeros((N, K)), -np.eye(N)])
    h4 = np.zeros(N)

    G = np.vstack([G1, G2, G3, G4])
    h = np.concatenate([h1, h2, h3, h4])

    sol = cvxsolvers.lp(cvxmat(c, tc="d"), cvxmat(G, tc="d"), cvxmat(h, tc="d"))
    x = np.array(sol["x"]).reshape(-1)
    w = np.clip(x[:K], 0.0, None)
    return w


def fit_weights_L1(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = solve_l1_lp(Phi, y)
    s = float(w.sum())
    if s > 0:
        w = w / s
    return w


def precompute_softplus_mixtures(
    nmax: int,
    *,
    t_min: float = -10.0,
    t_max: float = 50.0,
    t_N: int = 1200,
    means_min: float = -20.0,
    means_max: float = 50.0,
    K_of_n=None,
    sigma_of_n=None,
    normalize_target: bool = True,
    thr_active: float = 1e-12,
    verbose: bool = True,
) -> dict:

    if K_of_n is None:
        def K_of_n(n):
            return 60

    if sigma_of_n is None:
        sigma_of_n = {"default": 1.0}

    t = np.linspace(t_min, t_max, t_N)

    mix = {
        "t": t,
        "sigma": {},
        "means": {},
        "w": {},
        "active": {},
        "nmax": int(nmax),
    }

    Phi_cache = {}

    for n in range(nmax + 1):
        sigma_n = float(sigma_of_n.get(n, sigma_of_n.get("default", 1.0)))
        K_n = int(K_of_n(n))
        means_n = np.linspace(means_min, means_max, K_n)

        y = poisson_like_unnormalized(t, n)
        if normalize_target:
            y = normalize_on_grid(y)

        key = (sigma_n, K_n)
        if key not in Phi_cache:
            Phi_cache[key] = design_matrix(t, means_n, sigma_n)
        Phi = Phi_cache[key]

        w = fit_weights_L1(Phi, y)
        active = np.flatnonzero(w > thr_active)
        if active.size == 0:
            active = np.array([int(np.argmax(w))], dtype=int)

        mix["sigma"][n] = sigma_n
        mix["means"][n] = means_n
        mix["w"][n] = w
        mix["active"][n] = active

        if verbose:
            print(f"[mixture] n={n} | active={active.size}")

    return mix


# =========================

if __name__ == "__main__":


    np.random.seed(0)

    # --- 1) Build prior covariance ---
    try:
        import syntheticdata as sd


        n, m = 20, 20
        N = n * m

        prior_mean = np.zeros(N)
        prior_cov = sd.spatial_covariance_gaussian(n, m, rho=3, v2=1)

        # Sample f_true from the prior
        L = spla.cholesky(prior_cov, lower=True)
        f_true = prior_mean + L @ np.random.normal(size=N)

        # Observations: nobs ~ Poisson( softplus(f_true) )
        rate_true = softplus(f_true)
        nobs = np.random.poisson(rate_true)

        print(f"Synthetic 2D data: grid=({n}x{m}), max count={int(nobs.max())}")

        # --- 2) Offline mixture precomputation up to nmax ---
        nmax = int(np.max(nobs))

        def K_of_n(n_):
            return 60

        sigma_of_n = {"default": 1.0}
        sigma_of_n[0]=1.5

        mix = precompute_softplus_mixtures(
            nmax,
            t_min=-10.0,
            t_max=50.0,
            t_N=1500,
            means_min=-20.0,
            means_max=50.0,
            K_of_n=K_of_n,
            sigma_of_n=sigma_of_n,
            normalize_target=True,
            thr_active=1e-12,
            verbose=True,
        )

        # --- 3) Create density + set data ---
        dens = SoftplusMixtureDensity(prior_mean=prior_mean, prior_covariance=prior_cov, mix=mix)
        dens.set_data(nobs)
        # --- 4) Run Gibbs ---
        samples = gibbs_sampler(
            dens,
            n_iter=200,
            burn_in=100,
            thin=10,
            initial_f=None,
            random_seed=1,
        )

        f_est = samples.mean(axis=0)
        rate_est = softplus(f_est)
     


        # --- 5) Plot  ---
        nsamp = samples.shape[0]
        show_idx = [0, nsamp//3, 2*nsamp//3, nsamp-1]
        show_idx = sorted(set(i for i in show_idx if 0 <= i < nsamp))
        print("nsamp:", nsamp, "show_idx:", show_idx)

        for si in show_idx:
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111)
            ax.set_title(f"Stored sample {si}")

            field_s = softplus(samples[si])
            im = ax.imshow(sd.scanorder_to_image(field_s, n, m).T)
            fig.colorbar(im, ax=ax)

            fig.tight_layout()

            out = f"results/sample_field_{si}.png"
            os.makedirs("results", exist_ok=True)
            fig.savefig(out, dpi=180, bbox_inches="tight")
            print("Saved:", out)

            plt.show(block=True)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("True field (softplus(f_true))")
        plt.imshow(sd.scanorder_to_image(rate_true, n, m).T)
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("Observed events (nobs)")
        plt.imshow(sd.scanorder_to_image(nobs, n, m).T)
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("Posterior mean field (softplus(E[f]))")
        plt.imshow(sd.scanorder_to_image(rate_est, n, m).T)
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        # Fallback: 1D correlated demo if syntheticdata is unavailable
        print("[warn] syntheticdata import failed; using 1D fallback demo.")
        print("[warn] error:", repr(e))

        N = 100
        x = np.linspace(0.0, 1.0, N)
        ell = 0.08
        prior_cov = np.exp(-0.5 * ((x[:, None] - x[None, :]) / ell) ** 2) + 1e-6 * np.eye(N)
        prior_mean = np.zeros(N)

        L = spla.cholesky(prior_cov, lower=True)
        f_true = prior_mean + L @ np.random.normal(size=N)
        nobs = np.random.poisson(softplus(f_true))

        nmax = int(np.max(nobs))
        mix = precompute_softplus_mixtures(nmax, verbose=False)

        dens = SoftplusMixtureDensity(prior_mean=prior_mean, prior_covariance=prior_cov, mix=mix)
        dens.set_data(nobs)

        samples = gibbs_sampler(dens, n_iter=200, burn_in=100, thin=10, random_seed=1)
        f_est = samples.mean(axis=0)

        print("samples shape:", samples.shape)
        print("mean rate true:", float(np.mean(softplus(f_true))))
        print("mean rate est:", float(np.mean(softplus(f_est))))
