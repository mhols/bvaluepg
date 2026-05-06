from __future__ import annotations

import os
import time
import numpy as np
import scipy.linalg as spla
from pathlib import Path
import pickle
from cvxopt import matrix as cvxmat, solvers as cvxsolvers
cvxsolvers.options["show_progress"] = False


# =========================
# Basic Gaussian helper
# =========================
def gaussian_pdf(x: float | np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    z = (x - mean) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * sigma)


# =========================
# Step 1: sample z | f, n
# =========================
def sample_z_cond_f(f: np.ndarray, nobs: np.ndarray, mix: dict) -> np.ndarray:
    N = int(f.shape[0])
    z = np.empty(N, dtype=int)

    for i in range(N):
        n_i = int(nobs[i])
        if n_i > mix["nmax"]:
            raise ValueError(f"count {n_i} exceeds precomputed nmax={mix['nmax']}")

        w = np.asarray(mix["w"][n_i], dtype=float)

        # Gaussian tail: single component
        if w.size == 1:
            z[i] = 0
            continue

        sigma = float(mix["sigma"][n_i])
        means = np.asarray(mix["means"][n_i], dtype=float)
        active = np.asarray(mix["active"][n_i], dtype=int)

        m_act = means[active]
        w_act = w[active]

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
def prepare_f_cond_z(
    nobs: np.ndarray,
    prior_mean: np.ndarray,
    Lprior: np.ndarray,
    mix: dict,
) -> dict:
    """
    Precompute all quantities in p(f | z, n) that are fixed across Gibbs iterations.

    sigma[n_i] depends only on n_i, and nobs is fixed.
    """
    nobs = np.asarray(nobs, dtype=int)
    prior_mean = np.asarray(prior_mean, dtype=float)
    L = np.asarray(Lprior, dtype=float)

    N = int(prior_mean.shape[0])

    sigma = np.empty(N, dtype=float)
    for i in range(N):
        sigma[i] = float(mix["sigma"][int(nobs[i])])

    s2 = sigma * sigma
    dinv = 1.0 / (s2 + 1e-15)

    T = np.eye(N) + L.T @ (dinv[:, None] * L)
    cholT = spla.cholesky(T, lower=True)

    # C^{-1} prior_mean
    y = spla.solve_triangular(L, prior_mean, lower=True)
    Cinv_prior_mean = spla.solve_triangular(L.T, y, lower=False)

    return {
        "nobs": nobs,
        "L": L,
        "dinv": dinv,
        "cholT": cholT,
        "Cinv_prior_mean": Cinv_prior_mean,
    }

def sample_f_cond_z(
    z: np.ndarray,
    nobs: np.ndarray,
    prior_mean: np.ndarray,
    Lprior: np.ndarray,
    mix: dict,
) -> np.ndarray:
    """
    Conditioned on z and nobs, the likelihood becomes diagonal Gaussian:
        f_i ~ N(mu_i, s2_i)
    where
        mu_i = means[n_i][z_i]
        s2_i = sigma[n_i]^2
    """
    N = int(prior_mean.shape[0])

    mu = np.empty(N, dtype=float)
    s2 = np.empty(N, dtype=float)

    for i in range(N):
        n_i = int(nobs[i])
        mu[i] = float(mix["means"][n_i][int(z[i])])
        sigma = float(mix["sigma"][n_i])
        s2[i] = sigma * sigma

    dinv = 1.0 / (s2 + 1e-15)

    L = Lprior
    T = np.eye(N) + L.T @ (dinv[:, None] * L)
    cholT = spla.cholesky(T, lower=True)

    def Cinv_dot(v: np.ndarray) -> np.ndarray:
        y = spla.solve_triangular(L, v, lower=True)
        x = spla.solve_triangular(L.T, y, lower=False)
        return x

    b = Cinv_dot(prior_mean) + dinv * mu

    Lt_b = L.T @ b
    y = spla.solve_triangular(cholT, Lt_b, lower=True)
    x = spla.solve_triangular(cholT.T, y, lower=False)
    mean = L @ x

    u = np.random.normal(size=N)
    v = spla.solve_triangular(cholT.T, u, lower=False)
    noise = L @ v

    return mean + noise


def sample_f_cond_z_cache(
    z: np.ndarray,
    cache: dict,
    mix: dict,
) -> np.ndarray:
    """
    Sample f | z, n using precomputed quantities from prepare_f_cond_z().
    """
    z = np.asarray(z, dtype=int)

    nobs = cache["nobs"]
    L = cache["L"]
    dinv = cache["dinv"]
    cholT = cache["cholT"]
    Cinv_prior_mean = cache["Cinv_prior_mean"]

    N = int(nobs.shape[0])

    mu = np.empty(N, dtype=float)
    for i in range(N):
        n_i = int(nobs[i])
        mu[i] = float(mix["means"][n_i][int(z[i])])

    b = Cinv_prior_mean + dinv * mu

    Lt_b = L.T @ b
    y = spla.solve_triangular(cholT, Lt_b, lower=True)
    x = spla.solve_triangular(cholT.T, y, lower=False)
    mean = L @ x

    u = np.random.normal(size=N)
    v = spla.solve_triangular(cholT.T, u, lower=False)
    noise = L @ v

    return mean + noise



# =========================
# Mixture precomputation
# =========================
def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def poisson_like_unnormalized(t: np.ndarray, n: int, softplus_k: float = 1.0) -> np.ndarray:
    """
    y(t) ∝ L_k(t)^n * exp(-L_k(t))
    where L_k(t) = softplus(k t) / k.
    computed in log-space for numerical stability.
    """
    k = float(softplus_k)


    lam = softplus(k * t) / k
    lam = np.maximum(lam, 1e-300)

    logy = n * np.log(lam) - lam
    logy = logy - np.max(logy)

    y = np.exp(logy)
    y = np.where(np.isfinite(y), y, 0.0)
    y = np.maximum(y, 0.0)
    return y


def normalize_on_grid(y: np.ndarray) -> np.ndarray:
    y = np.where(np.isfinite(y), y, 0.0)
    y = np.maximum(y, 0.0)
    s = float(np.sum(y))
    if (not np.isfinite(s)) or (s <= 0.0):
        return np.full_like(y, 1.0 / y.size, dtype=float)
    return y / s


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

    G = np.asarray(G, dtype=float)
    h = np.asarray(h, dtype=float)
    c = np.asarray(c, dtype=float)

    sol = cvxsolvers.lp(cvxmat(c, tc="d"), cvxmat(G, tc="d"), cvxmat(h, tc="d"))
    x = np.array(sol["x"]).reshape(-1)
    w = np.clip(x[:K], 0.0, None)
    return w


def fit_weights_L1(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = solve_l1_lp(Phi, y)
    s = float(w.sum())
    if s > 0:
        w = w / s
    else:
        w = np.ones_like(w) / w.size
    return w

def make_means_for_kappa(means_left, means_right, K_n, kappa):
    """
    Build Gaussian component centers.

    For kappa=1, use a uniform grid.
    For larger kappa, add more centers near f=0.
    """
    kappa = float(kappa)

    if kappa <= 1.0:
        return np.linspace(means_left, means_right, K_n)

    K_local = int(0.6 * K_n)
    K_global = K_n - K_local

    local_width = max(2.0, 8.0 / kappa)

    means_global = np.linspace(means_left, means_right, K_global)
    means_local = np.linspace(-local_width, local_width, K_local)

    means = np.unique(np.concatenate([means_global, means_local]))
    means.sort()

    return means


def precompute_softplus_mixtures(
    nmax: int,
    *,
    softplus_k: float = 1.0,
    t_N: int = 1200,
    K_of_n=None,
    sigma_of_n=None,
    normalize_target: bool = True,
    thr_active: float = 1e-12,
    tail_start: int = 60,
    t_half_width: float = 20.0,
    means_half_width: float = 15.0,
    verbose: bool = True,
) -> dict:
    """
    For n <= tail_start:
        fit a Gaussian mixture on an interval centered around n.
    For n > tail_start:
        use a single Gaussian approximation N(mean=n, var=n).
    """
    softplus_k = float(softplus_k)

    

    if K_of_n is None:
        def K_of_n(n):
            return 60

    if sigma_of_n is None:
        sigma_of_n = {"default": 1.0, 0: 1.5}

    mix = {
        "sigma": {},
        "means": {},
        "w": {},
        "active": {},
        "t_grid": {},
        "nmax": int(nmax),
        "tail_start": int(tail_start),
        "softplus_k": softplus_k,
    }

    Phi_cache = {}

    for n in range(nmax + 1):

        if n > tail_start:
            sigma_n = np.sqrt(float(max(n, 1)))

            mix["sigma"][n] = sigma_n
            mix["means"][n] = np.array([float(n)], dtype=float)
            mix["w"][n] = np.array([1.0], dtype=float)
            mix["active"][n] = np.array([0], dtype=int)
            mix["t_grid"][n] = np.linspace(
                float(n) - 4.0 * sigma_n,
                float(n) + 4.0 * sigma_n,
                t_N,
            )

            if verbose:
                print(f"[mixture] n={n:3d} | Gaussian tail approx N({n:.1f}, {n:.1f})")

            continue

        K_n = int(K_of_n(n))
        sigma_n = float(sigma_of_n.get(n, sigma_of_n.get("default", 1.0)))

        t_left = min(-10.0, float(n) - t_half_width)
        t_right = float(n) + t_half_width
        t = np.linspace(t_left, t_right, t_N)

        means_left = float(n) - means_half_width
        means_right = float(n) + means_half_width

        means_n = make_means_for_kappa(
            means_left,
            means_right,
            K_n,
            softplus_k,
        )
        K_n = len(means_n)

        y = poisson_like_unnormalized(t, n, softplus_k=softplus_k)
        if normalize_target:
            y = normalize_on_grid(y)

        cache_key = (
            round(float(softplus_k), 8),
            round(t_left, 8),
            round(t_right, 8),
            int(t_N),
            round(sigma_n, 8),
            int(K_n),
            round(means_left, 8),
            round(means_right, 8),
        )

        if cache_key not in Phi_cache:
            Phi_cache[cache_key] = design_matrix(t, means_n, sigma_n)
        Phi = Phi_cache[cache_key]

        w = fit_weights_L1(Phi, y)
        active = np.flatnonzero(w > thr_active)
        if active.size == 0:
            active = np.array([int(np.argmax(w))], dtype=int)

        mix["sigma"][n] = sigma_n
        mix["means"][n] = means_n
        mix["w"][n] = w
        mix["active"][n] = active
        mix["t_grid"][n] = t

        if verbose:
            print(
                f"[mixture] n={n:3d} | K={K_n:3d} | active={active.size:3d} "
                f"| t=[{t_left:.1f},{t_right:.1f}] | means=[{means_left:.1f},{means_right:.1f}]"
            )

    return mix

def load_or_build_mix(nmax_mix: int, cache_dir: Path, softplus_k: float = 1.0,) -> dict:
    softplus_k = float(softplus_k)
    cache_dir.mkdir(parents=True, exist_ok=True)

    k_tag = str(softplus_k).replace(".", "p")
    cache_path = cache_dir / f"softplus_mix_k{k_tag}_L1_nmax{nmax_mix}_tail60.pkl"

    if cache_path.exists():
        print(f"[mix] loading cache: {cache_path}")
        with open(cache_path, "rb") as f:
            mix = pickle.load(f)

        cached_k = float(mix.get("softplus_k", 1.0))
        if not np.isclose(cached_k, softplus_k):
            raise ValueError(
                f"Loaded mixture has softplus_k={cached_k}, "
                f"but requested softplus_k={softplus_k}."
            )

        return mix

    print(f"[mix] building mix up to nmax_mix={nmax_mix}, softplus_k={softplus_k}")

    def K_of_n(n: int) -> int:
        return int(60 * max(1.0, np.sqrt(softplus_k)))

    sigma_base = max(0.25, 1.0 / np.sqrt(softplus_k))
    sigma_zero = max(0.5, 1.5 / np.sqrt(softplus_k))

    sigma_of_n = {
      "default": sigma_base,
       0: sigma_zero,
    }

    

    mix = precompute_softplus_mixtures(
        nmax_mix,
        softplus_k=softplus_k,
        t_N=1500,
        K_of_n=K_of_n,
        sigma_of_n=sigma_of_n,
        normalize_target=True,
        thr_active=1e-12,
        tail_start=60,
        t_half_width=20.0,
        means_half_width=15.0,
        verbose=True,
    )

    with open(cache_path, "wb") as f:
        pickle.dump(mix, f)

    print(f"[mix] saved cache: {cache_path}")
    return mix

