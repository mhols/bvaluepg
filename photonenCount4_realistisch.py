#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photonenCount4_realistic.py
---------------------------------
Simulation + Inferenz-Pipeline mit folgenden Eigenschaften:
(a) Räumlich korreliertes Rauschen wird in die Logits/Intensitäten integriert.
(b) Die Klassifikationswahrscheinlichkeiten sind nicht trivial (~0.55–0.85 statt ~0.97).
(c) Die Photonen-Zählungen sind an ein latentes Intensitätsfeld gekoppelt (Multinomial).
(d) Keine "inverse crime": Generatives Modell enthält leichte Nichtlinearität, 
    das Fit-Modell ist aber (bewusst) linear in den Features.

Falls das 'polyagamma'-Modul verfügbar ist, wird eine PG-basierte Gibbs-Sampling
für die logistische Regression verwendet. Andernfalls wird automatisch auf eine
Laplace/MAP-Approximation (Newton-Verfahren) mit Gaussian-Prior zurückgefallen.
Damit bleibt das Skript robust lauffähig, auch ohne externe PG-Library.

Abhängigkeiten (nur Standard-Stack):
- numpy, scipy (für ndimage), matplotlib
- sklearn wird NICHT benötigt; Metriken (AUC, Confusion) sind implementiert.

Autor: ChatGPT (für Toni Luhdo)
Datum: 2025-10-22
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------
def set_seed(seed=42):
    rng = np.random.default_rng(seed)
    return rng


def make_chessboard(rows, cols, low=0.3, high=1.0):
    """Erzeugt ein Schachbrett-Muster als 2D-Array mit Werten in {low, high}."""
    pattern = np.indices((rows, cols)).sum(axis=0) % 2  # 0/1
    pattern = np.where(pattern == 0, low, high).astype(float)
    return pattern


def gaussian_random_field(rows, cols, rng, sigma_pix=2.0, scale=1.0):
    """
    Erzeugt ein räumlich korreliertes Feld, indem weißes Rauschen per
    Gauß-Filter geglättet wird (effizient, kein großes Cholesky nötig).
    """
    white = rng.standard_normal((rows, cols))
    field = gaussian_filter(white, sigma=sigma_pix, mode='reflect')
    field = (field - field.mean()) / (field.std() + 1e-12)
    return scale * field


def softplus(x):
    """Weiche Positivierung, um stabile positive Intensitäten zu bekommen."""
    return np.log1p(np.exp(x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def standardize(x):
    mu = np.mean(x)
    sd = np.std(x) + 1e-12
    return (x - mu) / sd, mu, sd


def multinomial_counts_from_intensity(intensity, n_photons, rng):
    """
    Verteilt n_photons proportional zur Intensität per Multinomial.
    intensity: 2D Feld >= 0
    """
    probs = intensity.ravel()
    probs = probs / (probs.sum() + 1e-12)
    hits = rng.choice(np.arange(probs.size), size=n_photons, p=probs)
    counts = np.bincount(hits, minlength=probs.size).reshape(intensity.shape)
    return counts


def auc_roc(scores, y):
    """
    ROC-AUC ohne sklearn. y ∈ {0,1}. Ties werden korrekt behandelt via Rank-Sum.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    # Ranks der Scores (Average ranks bei ties)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Ties: average ranks
    # (Implementieren via Gruppenmittel der gleichen Werte)
    # Effizient: sortierte Scores, dann Blöcke gleicher Werte mittlerer Rang
    s_sorted = scores[order]
    start = 0
    while start < len(s_sorted):
        end = start + 1
        while end < len(s_sorted) and s_sorted[end] == s_sorted[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)  # 1-based ranks
        ranks[order[start:end]] = avg_rank
        start = end

    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    rank_sum_pos = ranks[y == 1].sum()
    # Mann-Whitney U for positives
    U1 = rank_sum_pos - n1 * (n1 + 1) / 2.0
    auc = U1 / (n1 * n0)
    return float(auc)


def confusion_matrix_01(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


# ------------------------------------------------------------
# Pólya-Gamma Gibbs (optional) oder Laplace/MAP (Fallback)
# ------------------------------------------------------------
def fit_logistic_pg_or_laplace(X, y, rng, prior_var=10.0, n_iter_pg=400, warm_start=None):
    """
    Logistic Regression mit Gaussian-Prior N(0, prior_var I).
    Versuche PG-Gibbs (polyagamma), fallback auf Laplace/MAP falls nicht verfügbar.
    """
    try:
        import polyagamma as pg  # benötigt installiertes 'polyagamma' (PyPI)
        use_pg = True
    except Exception:
        use_pg = False

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    n, p = X.shape
    prior_prec = (1.0 / prior_var) * np.eye(p)

    if use_pg:
        # PG Gibbs
        beta = np.zeros(p) if warm_start is None else warm_start.copy()
        sampler = pg.PyPolyaGamma() if hasattr(pg, 'PyPolyaGamma') else None

        betas = []
        for it in range(n_iter_pg):
            z = X @ beta
            # ω_i ~ PG(1, z_i)
            if sampler is not None:
                omega = np.array([sampler.pgdraw(1.0, zi) for zi in z])
            else:
                # Neuere polyagamma-Versionen: rvs(n, z) -> size=n
                omega = pg.random_polyagamma(1.0, z, size=n)

            # Posterior für beta: N(m, V) mit
            # V = (X^T Ω X + prior_prec)^-1, m = V X^T (y - 0.5)
            XT_omega = X.T * omega  # (p,n)
            H = XT_omega @ X + prior_prec
            b = X.T @ (y - 0.5)
            # Löse H m = b
            m = np.linalg.solve(H, b)
            # Ziehe beta ~ N(m, V) via Cholesky
            L = np.linalg.cholesky(H)
            # Ziehen aus N(m, H^-1): Lösung von L^T u = eps, L v = u, beta = m + v
            eps = rng.standard_normal(p)
            u = np.linalg.solve(L.T, eps)
            v = np.linalg.solve(L, u)
            beta = m + v
            betas.append(beta.copy())

        betas = np.array(betas)
        beta_hat = betas[int(0.5 * n_iter_pg):].mean(axis=0)  # Burn-in 50%, mean
        return beta_hat, {'method': 'PG-Gibbs', 'trace': betas}

    else:
        # Laplace/MAP (Newton-Raphson)
        beta = np.zeros(p) if warm_start is None else warm_start.copy()
        for it in range(50):
            z = X @ beta
            p_ = sigmoid(z)
            W = p_ * (1 - p_)
            # Gradient + Hessian mit Gaussian-Prior
            g = X.T @ (y - p_) - prior_prec @ beta
            H = X.T @ (X * W[:, None]) + prior_prec
            # Newton-Schritt
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                # Regularisiere minimal falls nötig
                H_reg = H + 1e-6 * np.eye(p)
                step = np.linalg.solve(H_reg, g)
            beta_new = beta + step
            # Konvergenz
            if np.linalg.norm(step) < 1e-6:
                beta = beta_new
                break
            beta = beta_new
        return beta, {'method': 'Laplace-MAP'}


# ------------------------------------------------------------
# Haupt-Simulation
# ------------------------------------------------------------
def run_simulation(rows=30, cols=30, seed=42,
                   n_photons=12000,
                   sigma_field=2.0, field_scale=0.9,
                   # Generatives Modell für Intensität (Counts):
                   a0=0.0, a1=1.2,
                   # Klassifikations-Logits: Ziel p ~ 0.55–0.85
                   b0=-0.2, b1=1.0, b2=0.9, b3=0.3,
                   noise_std_logits=0.35):
    """
    - Intensitätsfeld: eta = a0 + a1*pattern + field  -> lambda = softplus(eta)
    - Photonen: Multinomial proportional zu lambda
    - y-Logits (Nichtlinear, bewusst leicht "mis-specified" vs spätere lineare Fits):
        logits_y = b0 + b1*z + b2*counts_std + b3*(z*counts_std) + eps
    """
    rng = set_seed(seed)

    # 1) Feature (Schachbrett) und korreliertes Feld
    z = make_chessboard(rows, cols, low=0.3, high=1.0)
    field = gaussian_random_field(rows, cols, rng, sigma_pix=sigma_field, scale=field_scale)

    # 2) Latente Intensität für Photonen-Zählungen
    eta = a0 + a1 * z + field
    lam = softplus(eta)  # >= 0
    counts = multinomial_counts_from_intensity(lam, n_photons, rng)

    # 3) Zielvariable y koppeln an z und counts (mit Nichtlinearität z*counts_std)
    counts_std, mu_c, sd_c = standardize(counts)
    logits_y = b0 + b1 * z + b2 * counts_std + b3 * (z * counts_std) + rng.normal(0, noise_std_logits, size=z.shape)
    p = sigmoid(logits_y)
    y = (rng.uniform(size=p.shape) < p).astype(int)

    # 4) Designmatrix für lineares Fit-Modell (bewusst OHNE das Interaktionsterm z*counts_std)
    X = np.stack([np.ones_like(z.ravel()), z.ravel(), counts_std.ravel()], axis=1)

    return {
        'z': z, 'field': field, 'eta': eta, 'lambda': lam, 'counts': counts,
        'logits_y': logits_y, 'p': p, 'y': y, 'X': X
    }


def evaluate_and_plot(sim, beta_hat, method_label=''):
    z = sim['z']
    counts = sim['counts']
    X = sim['X']
    y = sim['y'].ravel()

    # Scores & Metriken
    scores = X @ beta_hat
    probs = sigmoid(scores)
    auc = auc_roc(probs, y)
    y_hat = (probs >= 0.5).astype(int)
    tp, fp, fn, tn = confusion_matrix_01(y, y_hat)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    print(f"[{method_label}] AUC: {auc:.3f} | Accuracy: {acc:.3f} | TP:{tp} FP:{fp} FN:{fn} TN:{tn}")

    # Plots
    fig1 = plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.title("Photonen-Zählungen (Multinomial)")
    plt.imshow(counts, origin='lower')
    plt.colorbar(shrink=0.8)
    plt.subplot(1, 2, 2)
    plt.title("Vorhersagewahrscheinlichkeit p(y=1|X)")
    plt.imshow(probs.reshape(z.shape), origin='lower', vmin=0, vmax=1)
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    plt.show()


def main():
    # ---- Simulation ----
    sim = run_simulation(
        rows=30, cols=30, seed=42,
        n_photons=15000,            # ein paar mehr Photonen für sichtbarere Struktur
        sigma_field=2.5, field_scale=1.0,
        a0=0.1, a1=1.1,             # Intensitätssteuerung
        b0=-0.3, b1=0.9, b2=1.0, b3=0.35,  # Ziel-Logits-Steuerung (nicht trivial)
        noise_std_logits=0.35
    )

    # ---- Fit: PG (falls vorhanden) oder Laplace ----
    X = sim['X']
    y = sim['y'].ravel()
    rng = set_seed(2025)

    beta_hat, meta = fit_logistic_pg_or_laplace(
        X, y, rng,
        prior_var=10.0,
        n_iter_pg=500,     # nur relevant, wenn PG verfügbar
        warm_start=None
    )

    method_label = meta.get('method', 'Unknown')
    print("Geschätzte Koeffizienten (β):", beta_hat)

    # ---- Auswertung + Plots ----
    evaluate_and_plot(sim, beta_hat, method_label=method_label)


if __name__ == "__main__":
    main()
