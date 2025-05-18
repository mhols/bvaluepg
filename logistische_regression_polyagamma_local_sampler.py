"""
Bayessche logistische Regression mit Pólya-Gamma-Datenaugmentation
==================================================================

Problemstellung:
----------------
Wir betrachten ein binäres Klassifikationsproblem mit zwei Bins/Eimern.
Ein Experiment wirft 50 Bälle nacheinander, und jeder Ball landet zufällig
in einem der beiden Eimer (Bin 0 oder Bin 1). Die Wahrscheinlichkeit, dass
ein Ball in Bin 1 landet, hängt von einem Merkmalsvektor x_i ab.

Modell:
-------
- Zielvariable: y_i ∈ {0, 1}
- Merkmale: x_i ∈ ℝ^p
- Logistische Regression:
    p(y_i = 1 | x_i, β) = σ(x_i^T β), wobei σ(z) = 1 / (1 + exp(-z))

Bayessche Inferenz:
-------------------
- Prior: β ~ N(0, τ² I)
- Likelihood wird durch Einführung latenter Variablen ω_i ∼ PG(1, x_i^T β)
  so transformiert, dass die bedingte Verteilung von β wieder Normal ist.

Gibbs-Sampler:
--------------
1. Ziehe ω_i | β  ~ Pólya-Gamma(1, x_i^T β)
2. Ziehe β | ω, y ~ Normal

Dieses Skript simuliert Daten, führt einen Gibbs-Sampler aus und speichert die Posterior-Samples.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt

# -------------------------------
# Hilfsfunktion: Approximation von PG(1, z)
# -------------------------------

def sample_pg_approx(z, trunc=200):
    """
    Approximiert einen Polya-Gamma(1, z) Draw nach Devroye (2014),
    durch Trunkierung der unendlichen Summe.
    """
    out = 0.0
    z = np.abs(z)
    for n in range(1, trunc + 1):
        lam = (n - 0.5) * np.pi
        out += np.random.gamma(1.0, 1.0) / (lam**2 + (z**2) / 4.0)
    return 0.25 * out

# -------------------------------
# 1. Simuliere die Daten
# -------------------------------

np.random.seed(42)
n = 50            # Anzahl Bälle / Beobachtungen
p = 1             # Anzahl Merkmale
X = np.random.randn(n, p)  # Zufällige Merkmalsmatrix

# beta_true = np.array([1.5, -2.0])  # Wahres β
beta_true = np.array([1.5])
logits = X @ beta_true
probs = 1 / (1 + np.exp(-logits))  # σ(Xβ)
y = np.random.binomial(1, probs)   # Binäre Zielvariablen

# -------------------------------
# 2. Gibbs-Sampler Setup
# -------------------------------

n_iter = 100
beta_samples = np.zeros((n_iter, p))
beta = np.zeros(p)  # Initialwert
# anderen Startwert
beta = np.ones(p)
tau2 = 10.0         # Priorvarianz

for t in range(n_iter):
    # a) Ziehe ω_i | β
    psi = X @ beta
    omega = np.array([sample_pg_approx(psi_i) for psi_i in psi])  # ω ∼ PG(1, x_i^T β)

    # b) Ziehe β | ω, y
    Omega = np.diag(omega)
    XT_Omega_X = X.T @ Omega @ X
    Sigma_post = np.linalg.inv(XT_Omega_X + (1 / tau2) * np.eye(p))
    mu_post = Sigma_post @ (X.T @ (y - 0.5))
    
    #wende cholesky auf weisses rauschen
    #linalg.solve
    beta = multivariate_normal.rvs(mean=mu_post, cov=Sigma_post)
    beta_samples[t, :] = beta
    beta = np.array([beta])


# -------------------------------
# 3. Plot Posteriorverteilungen
# -------------------------------

fig, axes = plt.subplots(1, p, figsize=(12, 4))
for j in range(p):
    axes[j].hist(beta_samples[:, j], bins=30, density=True, alpha=0.7, label=f"β{j}")
    axes[j].axvline(beta_true[j], color='r', linestyle='--', label='true β')
    axes[j].set_title(f"Posterior von β{j}")
    axes[j].legend()
# plt.tight_layout()
# plt.savefig("posterior_beta_bins.png")
plt.show()
