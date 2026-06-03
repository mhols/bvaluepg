from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt




from cvxopt import matrix, solvers
solvers.options["show_progress"] = False


# =========================
# Basic helpers
# =========================


def safe_exp(x: np.ndarray | float) -> np.ndarray:
    """Numerically safe exp for likelihood evaluation."""
    return np.exp(np.clip(x, -745.0, 700.0))


def gaussian_pdf(x: np.ndarray | float, mean: np.ndarray, sigma: float) -> np.ndarray:
    z = (x - mean) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * sigma)


def design_matrix(t_grid: np.ndarray, means: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_pdf(t_grid[:, None], means[None, :], sigma)


# =========================
# Exp-link likelihood target
# =========================


def poisson_like_unnormalized(t: np.ndarray, n: int) -> np.ndarray:
    """
    y(t) proportional to p(n | t), for lambda(t)=exp(t):
        log y(t) = n*t - exp(t) + constant.

    """
    t = np.asarray(t, dtype=float)
    n = int(n)

    logy = n * t - safe_exp(t)
    logy -= np.nanmax(logy)

    y = np.exp(logy)
    y = np.where(np.isfinite(y), y, 0.0)
    return np.maximum(y, 0.0)


def normalize_on_grid(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.where(np.isfinite(y), y, 0.0)
    y = np.maximum(y, 0.0)
    s = float(np.sum(y))
    if (not np.isfinite(s)) or s <= 0.0:
        return np.full_like(y, 1.0 / y.size)
    return y / s


# =========================
# L1 fit with CVXOPT
# =========================


def solve_l1_lp(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Nonnegative L1 fit using CVXOPT:
        min_{w >= 0, u >= 0} 1^T u
        s.t. -u <= Phi w - y <= u.
    """
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float)
    N, K = Phi.shape

    c = np.concatenate([np.zeros(K), np.ones(N)])

    # Phi w - u <= y
    G1 = np.hstack([Phi, -np.eye(N)])
    h1 = y.copy()

    # -Phi w - u <= -y
    G2 = np.hstack([-Phi, -np.eye(N)])
    h2 = -y.copy()

    # -w <= 0
    G3 = np.hstack([-np.eye(K), np.zeros((K, N))])
    h3 = np.zeros(K)

    # -u <= 0
    G4 = np.hstack([np.zeros((N, K)), -np.eye(N)])
    h4 = np.zeros(N)

    G = np.vstack([G1, G2, G3, G4])
    h = np.concatenate([h1, h2, h3, h4])

    sol = solvers.lp(matrix(c, tc="d"), matrix(G, tc="d"), matrix(h, tc="d"))
    if sol.get("status") != "optimal":
        raise RuntimeError(f"CVXOPT LP failed with status={sol.get('status')!r}")

    x = np.array(sol["x"]).reshape(-1)
    return np.clip(x[:K], 0.0, None)


def fit_weights_L1(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = solve_l1_lp(Phi, y)
    s = float(np.sum(w))
    if s > 0.0 and np.isfinite(s):
        return w / s
    return np.ones_like(w) / w.size


# =========================
# Exp-link mixture precomputation
# =========================


def exp_mode_and_scale(n: int) -> tuple[float, float]:
    """
    Laplace approximation for n >= 1:
        mode = log(n), curvature = n, sd = 1/sqrt(n).
    """
    n = int(n)
    if n <= 0:
        return -8.0, 3.0
    return float(np.log(n)), float(1.0 / np.sqrt(n))


def _zero_count_grid_and_centers(
    zero_left: float,
    zero_right: float,
    t_N: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Special basis for n=0.

    Uses sigma=0.85 and enough left centers to keep the plateau smooth.
    """
    t_left = float(zero_left)
    t_right = float(zero_right)

    # Grid: dense near the transition.
    t_left_grid = np.linspace(t_left, -6.0, max(250, t_N // 4), endpoint=False)
    t_mid_grid = np.linspace(-6.0, 2.5, max(700, t_N // 2), endpoint=False)
    t_right_grid = np.linspace(2.5, t_right, max(200, t_N // 5))
    t = np.unique(np.concatenate([t_left_grid, t_mid_grid, t_right_grid]))

    # Centers: spacing should be <= sigma, preferably close to sigma/2 where flatness matters.
    centers_left = np.linspace(t_left, -6.0, 35, endpoint=False)
    centers_mid = np.linspace(-6.0, 2.5, 75, endpoint=False)
    centers_right = np.linspace(2.5, t_right, 10)
    means = np.unique(np.concatenate([centers_left, centers_mid, centers_right]))

    sigma = 0.85
    return t, means, sigma


def precompute_exp_mixtures(
    nmax: int,
    *,
    t_N: int = 1500,
    K_of_n=None,
    sigma_of_n=None,
    normalize_target: bool = True,
    thr_active: float = 1e-12,
    tail_start: int = 60,
    t_half_width: float = 8.0,
    means_half_width: float = 8.0,
    zero_left: float = -25.0,
    zero_right: float = 8.0,
    verbose: bool = True,
) -> dict:
    """
    Precompute Gaussian-mixture approximations to p(n | f) for lambda=exp(f).

    For n > tail_start, use the exp-link Laplace tail approximation:
        f | n approx N(log(n), 1/n)
    as a single Gaussian component.

    For n=0, p(0|f)=exp(-exp(f)) tends to 1 as f -> -inf, so the normalized
    grid target is a finite-window approximation. The grid/centers are chosen
    separately to approximate the plateau and the cutoff around f≈0.
    """
    nmax = int(nmax)

    if K_of_n is None:
        def K_of_n(n: int) -> int:
            return 60

    if sigma_of_n is None:
        sigma_of_n = {"default": 0.35}

    mix = {
        "link": "exp",
        "version": "L1_v3_zero_smooth_plateau",
        "sigma": {},
        "means": {},
        "w": {},
        "active": {},
        "t_grid": {},
        "nmax": nmax,
        "tail_start": int(tail_start),
        "zero_window": (float(zero_left), float(zero_right)),
    }

    Phi_cache: dict[tuple, np.ndarray] = {}

    for n in range(nmax + 1):
    
        if n > tail_start:
            mu_n, sd_n = exp_mode_and_scale(n)
            sigma_n = sd_n

            mix["sigma"][n] = sigma_n
            mix["means"][n] = np.array([mu_n], dtype=float)
            mix["w"][n] = np.array([1.0], dtype=float)
            mix["active"][n] = np.array([0], dtype=int)
            mix["t_grid"][n] = np.linspace(mu_n - 4.0 * sd_n, mu_n + 4.0 * sd_n, t_N)

            if verbose:
                print(
                    f"[mixture-exp] n={n:3d} | Laplace tail "
                    f"N(log n={mu_n:.3f}, var={sd_n**2:.4g})"
                )
            continue

        if n == 0:
            t, means_n, sigma_n = _zero_count_grid_and_centers(zero_left, zero_right, t_N)
            K_n = int(means_n.size)
            t_left = float(t[0])
            t_right = float(t[-1])
        else:
            K_n = int(K_of_n(n))
            sigma_n = float(sigma_of_n.get(n, sigma_of_n.get("default", 0.35)))

            mu_n, sd_n = exp_mode_and_scale(n)
            half_t = max(float(t_half_width) * sd_n, 4.0 * sd_n, 1.5)
            half_m = max(float(means_half_width) * sd_n, 4.0 * sd_n, 1.5)

            t_left = mu_n - half_t
            t_right = mu_n + half_t
            means_left = mu_n - half_m
            means_right = mu_n + half_m

            t = np.linspace(t_left, t_right, t_N)
            means_n = np.linspace(means_left, means_right, K_n)

        y = poisson_like_unnormalized(t, n)
        if normalize_target:
            y = normalize_on_grid(y)

        cache_key = (
            int(t.size),
            int(means_n.size),
            round(float(t[0]), 8),
            round(float(t[-1]), 8),
            round(float(means_n[0]), 8),
            round(float(means_n[-1]), 8),
            round(float(sigma_n), 8),
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
                f"[mixture-exp] n={n:3d} | K={K_n:3d} | active={active.size:3d} "
                f"| sigma={sigma_n:.3f} | t=[{t_left:.3f},{t_right:.3f}]"
            )

    return mix


def load_or_build_exp_mix(
    nmax_mix: int,
    cache_dir: Path,
    *,
    force_rebuild: bool = False,
    **kwargs,
) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tail_start = int(kwargs.get("tail_start", 60))
    zero_left = float(kwargs.get("zero_left", -25.0))
    zero_right = float(kwargs.get("zero_right", 8.0))
    cache_path = cache_dir / (
        f"exp_mix_L1_v3_nmax{int(nmax_mix)}_tail{tail_start}_"
        f"z{zero_left:g}_{zero_right:g}.pkl"
    )

    if force_rebuild and cache_path.exists():
        print(f"[cache] deleting old cache: {cache_path}")
        cache_path.unlink()

    if cache_path.exists():
        print(f"[mix] loading cache: {cache_path}")
        with open(cache_path, "rb") as f:
            mix = pickle.load(f)
        if mix.get("link") != "exp":
            raise ValueError(f"Cache {cache_path} is not an exp-link mixture.")
        return mix

    print(f"[mix] building exp-link L1 mixture up to nmax_mix={nmax_mix}")
    mix = precompute_exp_mixtures(int(nmax_mix), **kwargs)
    with open(cache_path, "wb") as f:
        pickle.dump(mix, f)
    print(f"[mix] saved cache: {cache_path}")
    return mix


# =========================
# Plotting / diagnostics
# =========================


def mixture_fit_from_mix(t: np.ndarray, n: int, mix: dict):
    n = int(n)
    sigma = float(mix["sigma"][n])
    means = np.asarray(mix["means"][n], dtype=float)
    w = np.asarray(mix["w"][n], dtype=float)

    Phi = design_matrix(t, means, sigma)
    yhat = normalize_on_grid(Phi @ w)
    return yhat, Phi, w


def plot_likelihood_mixture(
    ax,
    n: int,
    nmax_mix: int,
    cache_dir: Path,
    force_rebuild: bool = False,
    xlim=None,
    comp_rel_thr: float = 1e-3,
):
    mix = load_or_build_exp_mix(
        nmax_mix=nmax_mix,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
    )

    n = int(n)
    t = np.asarray(mix["t_grid"][n], dtype=float)
    y = normalize_on_grid(poisson_like_unnormalized(t, n))
    yhat, Phi, w = mixture_fit_from_mix(t, n, mix)

    active = np.asarray(mix["active"][n], dtype=int)
    sigma = float(mix["sigma"][n])
    means = np.asarray(mix["means"][n], dtype=float)

    ax.set_xlabel("f")
    ax.set_ylabel("weighted Gaussian components")

    if active.size > 0:
        w_active = w[active]
        thr = comp_rel_thr * max(float(np.max(w_active)), 1e-300)
        display = active[w_active > thr]
    else:
        display = active

    for j in display:
        ax.plot(t, Phi[:, j] * w[j], linewidth=1.0, alpha=0.6)

    axr = ax.twinx()
    axr.set_ylabel("normalized likelihood / mixture fit")
    l1, = axr.plot(t, y, linewidth=2.8, label="exp-link likelihood")
    l2, = axr.plot(t, yhat, "--", linewidth=2.8, label="L1 Gaussian-mixture fit")

    ymax = max(float(np.max(y)), float(np.max(yhat)), 1e-15)
    axr.set_ylim(0.0, 1.10 * ymax)

    if xlim is not None:
        ax.set_xlim(*xlim)
        axr.set_xlim(*xlim)

    tv = 0.5 * float(np.sum(np.abs(y - yhat)))
    max_abs = float(np.max(np.abs(y - yhat)))

    ax.set_title(
        f"exp link, n={n}, sigma={sigma:.3g}, "
        f"K={len(means)}, active={len(active)}, shown={len(display)}\n"
        f"TV={tv:.3g}, max|err|={max_abs:.3g}"
    )
    axr.legend(handles=[l1, l2], loc="upper right", fontsize=8)

    return {"tv": tv, "max_abs": max_abs, "active": int(len(active)), "shown": int(len(display))}


def main():

    n = 4
    nmax_mix = 10
    cache_dir = Path(".mixture")
    force_rebuild = True

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.0))
    diagnostics = plot_likelihood_mixture(
        ax=ax,
        n=n,
        nmax_mix=nmax_mix,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        xlim=None,
        comp_rel_thr=1e-3,
    )

    fig.suptitle(
        "Poisson exponential-link likelihood and L1 Gaussian-mixture approximation",
        fontsize=13,
    )
    fig.tight_layout()

    out = Path("likelihood_mixture_exp_twinaxis.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved: {out}")
    print("diagnostics:", diagnostics)
    plt.show()


if __name__ == "__main__":
    main()
