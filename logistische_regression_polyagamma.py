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
- Merkmale: x_i ∈ R^Anazahl der Merkmale}
- Logistische Regression:
    p(y_i = 1 | x_i, β) = sigma(x_i^T β), wobei sigam(z) = 1 / (1 + exp(-z))

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
from polyagamma import polyagamma

# -------------------------------
# 1. Simuliere die Daten
# -------------------------------

np.random.seed(42)
n = 50            # Anzahl Bälle / Beobachtungen
p = 2             # Anzahl Merkmale, OMG ICH MUSS MIT p aufpassen
X = np.random.randn(n, p)  # Zufällige Merkmalsmatrix

print('X', X)

beta_true = np.array([1.5, -2.0])  # Wahres β
logits = X @ beta_true
probs = 1 / (1 + np.exp(-logits))  # σ(Xβ)
y = np.random.binomial(1, probs)   # Binäre Zielvariablen

# -------------------------------
# 2. Gibbs-Sampler Setup
# -------------------------------

n_iter = 1000
beta_samples = np.zeros((n_iter, p))
beta = np.zeros(p)  # Initialwert
tau2 = 10.0         # Priorvarianz

pg = polyagamma()

for t in range(n_iter):
    # a) Ziehe ω_i | β
    psi = X @ beta
    omega = np.array([pg.pgdraw(1, psi_i) for psi_i in psi])  # ω ∼ PG(1, x_i^T β)

    # b) Ziehe β | ω, y
    Omega = np.diag(omega)
    XT_Omega_X = X.T @ Omega @ X
    Sigma_post = np.linalg.inv(XT_Omega_X + (1 / tau2) * np.eye(p))
    mu_post = Sigma_post @ (X.T @ (y - 0.5))
    beta = multivariate_normal.rvs(mean=mu_post, cov=Sigma_post)

    beta_samples[t, :] = beta

# -------------------------------
# 3. Plot Posteriorverteilungen
# -------------------------------

fig, axes = plt.subplots(1, p, figsize=(12, 4))
for j in range(p):
    axes[j].hist(beta_samples[:, j], bins=30, density=True, alpha=0.7, label=f"β{j}")
    axes[j].axvline(beta_true[j], color='r', linestyle='--', label='true β')
    axes[j].set_title(f"Posterior von β{j}")
    axes[j].legend()
plt.tight_layout()
plt.savefig("posterior_beta_bins.png")
plt.close()
