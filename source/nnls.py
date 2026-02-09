import os
import time
import numpy as np
import matplotlib.pyplot as plt

from cvxopt import matrix, solvers
solvers.options["show_progress"] = False


# =========================
# Model: softplus + Poisson
# =========================
def softplus(t):
    return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0) ### very good we should use log1p in all our codes...

def poisson_like_unnormalized(t, n):
    lam = softplus(t)
    return (lam**n) * np.exp(-lam)

def normalize_on_grid(y):
    s = np.sum(y)
    return y / (s + 1e-15)

# =========================
# Dictionary
# =========================
def gaussian_pdf(x, mean, sigma):
    z = (x - mean) / sigma
    return np.exp(-0.5 * z*z) / (np.sqrt(2*np.pi) * sigma)

def design_matrix(t_grid, means, sigma):
    return gaussian_pdf(t_grid[:, None], means[None, :], sigma)

# =========================
# Metrics
# =========================
def diagnostics(y, yhat):
    diff = yhat - y
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    tv = float(0.5 * np.sum(np.abs(diff)))
    return rmse, max_abs, tv

def effective_components(w, thr=1e-6):
    return int(np.sum(w > thr))

# =========================
# Solvers
# =========================
def solve_nnls_qp(Phi, y, ridge=1e-10):
    PtP = Phi.T @ Phi
    K = PtP.shape[0]

    P = 2.0 * (PtP + ridge*np.eye(K))
    q = -2.0 * (Phi.T @ y)

    G = -np.eye(K)
    h = np.zeros(K)

    sol = solvers.qp(matrix(P, tc="d"),
                     matrix(q, tc="d"),
                     matrix(G, tc="d"),
                     matrix(h, tc="d"))
    return np.array(sol["x"]).reshape(-1)

def solve_weighted_nnls_qp(Phi, y, alpha, ridge=1e-10):
    a = np.sqrt(alpha).reshape(-1, 1)
    Phi_t = Phi * a
    y_t = y * a.reshape(-1)
    return solve_nnls_qp(Phi_t, y_t, ridge=ridge)

def solve_l1_lp(Phi, y):
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

    sol = solvers.lp(matrix(c, tc="d"),
                     matrix(G, tc="d"),
                     matrix(h, tc="d"))
    x = np.array(sol["x"]).reshape(-1)
    return x[:K]

def solve_linf_lp(Phi, y):
    N, K = Phi.shape
    c = np.concatenate([np.zeros(K), np.array([1.0])])

    ones = np.ones((N, 1))
    G1 = np.hstack([Phi, -ones])
    h1 = y.copy()

    G2 = np.hstack([-Phi, -ones])
    h2 = -y.copy()

    G3 = np.hstack([-np.eye(K), np.zeros((K, 1))])
    h3 = np.zeros(K)

    G4 = np.hstack([np.zeros((1, K)), np.array([[-1.0]])])
    h4 = np.array([0.0])

    G = np.vstack([G1, G2, G3, G4])
    h = np.concatenate([h1, h2, h3, h4])

    sol = solvers.lp(matrix(c, tc="d"),
                     matrix(G, tc="d"),
                     matrix(h, tc="d"))
    x = np.array(sol["x"]).reshape(-1)
    return x[:K]

def fit_objective(obj, Phi, y, *, rel_eps=1e-3):
    if obj == "L2":
        return solve_nnls_qp(Phi, y)
    if obj == "cappedRelL2":
        alpha = 1.0 / (y + rel_eps)
        alpha = alpha / (np.mean(alpha) + 1e-15)
        return solve_weighted_nnls_qp(Phi, y, alpha)
    if obj == "L1":
        return solve_l1_lp(Phi, y)
    if obj == "Linf":
        return solve_linf_lp(Phi, y)
    raise ValueError(f"Unknown objective: {obj}")

# =========================
# Pretty printing + saving
# =========================
def summarize(summary, objectives):
    rows = []
    for obj in objectives:
        rows.append({
            "obj": obj,
            "meanTV": float(np.mean(summary[obj]["tv"])),
            "meanRMSE": float(np.mean(summary[obj]["rmse"])),
            "worstTV": float(np.max(summary[obj]["tv"])),
            "meanEffK": float(np.mean(summary[obj]["effK"])),
            "meanTime": float(np.mean(summary[obj]["time"])),
        })
    return rows

def print_table(rows, *, title="", meta="", sort_by="meanTV"):
    if sort_by is not None:
        rows = sorted(rows, key=lambda r: r[sort_by])

    width = max(len(title), len(meta), 78)
    print("\n" + "="*width)
    if title:
        print(title)
    if meta:
        print(meta)
    print("="*width)

    header = f"{'obj':12s} | {'meanTV':>10s} {'meanRMSE':>10s} {'worstTV':>10s} | {'meanEffK':>9s} | {'meanTime(s)':>11s}"
    print(header)
    print("-"*len(header))

    for r in rows:
        print(f"{r['obj']:12s} | {r['meanTV']:10.3e} {r['meanRMSE']:10.3e} {r['worstTV']:10.3e} | {r['meanEffK']:9.1f} | {r['meanTime']:11.3e}")

def save_csv(rows, path):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def print_effk_by_n(summary, objectives, n_values):
    print("\nEffK by n (threshold = 1e-6)")
    header = "n  | " + " ".join([f"{obj:>12s}" for obj in objectives])
    print(header)
    print("-" * len(header))
    for i, n in enumerate(n_values):
        row = f"{n:2d} | " + " ".join([f"{summary[obj]['effK'][i]:12d}" for obj in objectives])
        print(row)


# =========================
# Plotting
# =========================
def plot_fits(t, y, fits_dict, n, sigma, out_dir="plots", show=True):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(11, 6))
    plt.plot(t, y, linewidth=2.5, label=f"target n={n}")

    for obj, yhat in fits_dict.items():
        plt.plot(t, yhat, linestyle="--", linewidth=1.5, label=obj)

    plt.title(f"Likelihood approximation (sigma={sigma}) for n={n}")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    fname = os.path.join(out_dir, f"fits_sigma{sigma}_n{n}.png")
    plt.savefig(fname, dpi=180)
    if show:
        plt.show()
    else:
        plt.close()
    return fname

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Config
    t = np.linspace(-10, 50, 1000)
    means = np.linspace(-15, 50, 150)
    sigma = 2
    n_values = list(range(20))
    normalize_target = True

    rel_eps = 1e-3

    objectives = ["L2", "cappedRelL2", "L1", "Linf"]

    seen = set()
    objectives = [o for o in objectives if not (o in seen or seen.add(o))]

    # Plot config
    do_plots = True
    plot_ns = [0, 3, 9, 15, 19]
    include_L1_in_plots = True

    # Precompute dictionary
    Phi = design_matrix(t, means, sigma)

    # Run
    summary = {obj: {"rmse": [], "tv": [], "max_abs": [], "effK": [], "time": []} for obj in objectives}

    # Also store fits for plotting for selected n
    fits_for_plot = {n: {} for n in plot_ns}

    for n in n_values:
        y_raw = poisson_like_unnormalized(t, n)
        y = normalize_on_grid(y_raw) if normalize_target else y_raw

        for obj in objectives:
            start = time.time()
            w = fit_objective(obj, Phi, y, rel_eps=rel_eps)
            elapsed = time.time() - start

            yhat = Phi @ w
            if normalize_target:
                yhat = normalize_on_grid(yhat)

            rmse, max_abs, tv = diagnostics(y, yhat)
            effK = effective_components(w, thr=1e-6)

            summary[obj]["rmse"].append(rmse)
            summary[obj]["tv"].append(tv)
            summary[obj]["max_abs"].append(max_abs)
            summary[obj]["effK"].append(effK)
            summary[obj]["time"].append(elapsed)

            # Save for plots (optionally skip L1)
            if do_plots and (n in plot_ns):
                if (obj == "L1") and (not include_L1_in_plots):
                    continue
                fits_for_plot[n][obj] = yhat

    # Print once, cleanly
    rows = summarize(summary, objectives)
    title = f"Objective comparison | sigma={sigma} | normalize_target={normalize_target}"
    meta = f"n_values={n_values} | means={len(means)} | grid={len(t)} | rel_eps={rel_eps}"
    print_table(rows, title=title, meta=meta, sort_by="meanTV")

    print_effk_by_n(summary, objectives, n_values)

    # Save CSV
    save_csv(rows, path=os.path.join("results", f"objective_summary_sigma{sigma}.csv"))

    # Plots
    if do_plots:
        for n in plot_ns:
            y_raw = poisson_like_unnormalized(t, n)
            y = normalize_on_grid(y_raw) if normalize_target else y_raw
            saved = plot_fits(t, y, fits_for_plot[n], n=n, sigma=sigma, out_dir="plots", show=True)
            print(f"Saved plot: {saved}")
