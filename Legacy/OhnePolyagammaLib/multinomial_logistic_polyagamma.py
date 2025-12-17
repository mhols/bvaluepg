"""
Multinomiale logistische Regression mit Pólya-Gamma-Datenaugmentation
======================================================================

Problemstellung:
----------------
Wir betrachten ein multinomiales Klassifikationsproblem mit 4 Klassen (Bins).
Ein Experiment wirft 50 Bälle, und jeder Ball landet zufällig in genau einem
von vier Eimern (Klassen 1 bis 4). Die Wahrscheinlichkeit hängt von einem
Merkmalsvektor x_i ab.

Modell:
-------
- Zielvariable: y_i ∈ {1, 2, 3, 4}
- Merkmale: x_i ∈ ℝ^p
- Wahrscheinlichkeiten:
    P(y_i = j) = exp(x_i^T β_j) / sum_k exp(x_i^T β_k)
- β_4 = 0 (Referenzklasse)

Bayessche Inferenz mit Pólya-Gamma:
-----------------------------------
- Für jede Klasse j = 1, 2, 3 wird ein binäres logit-Modell formuliert.
- Pólya-Gamma Latent-Variable ω_i^(j) ∼ PG(1, x_i^T β_j)
- Gibbs-Sampler über β_j und ω_i^(j) getrennt für j = 1, 2, 3

Dieses Skript simuliert Daten, führt Sampling aus und speichert Posterior-Samples.
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# -------------------------------
# Hilfsfunktion: Approximation von PG(1, z)
# -------------------------------

def sample_pg_approx(z, trunc=200):
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
n = 50
p = 2
K = 4

# Wahre Koeffizienten für Klassen 1 bis 3 (Klasse 4 ist Referenz)
beta_true = np.array([[1.0, -1.0],
                      [-1.5, 2.0],
                      [0.5, -0.5]])  # Shape (3, p)

X = np.random.randn(n, p)
logits = np.zeros((n, K))
for j in range(K - 1):  # Klasse 4 = Referenz (0-Logit)
    logits[:, j] = X @ beta_true[j]
logits[:, 3] = 0

# Softmax Wahrscheinlichkeiten
exp_logits = np.exp(logits)
probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

# Zielvariable y ∈ {0,1,2,3} → später +1 um Klassenbezeichner zu haben
y = np.array([np.random.choice(K, p=probs[i]) for i in range(n)])

# -------------------------------
# 2. Gibbs-Sampler Setup
# -------------------------------

n_iter = 1000
tau2 = 10.0
beta_samples = np.zeros((n_iter, K - 1, p))
beta = np.zeros((K - 1, p))

for t in range(n_iter):
    for j in range(K - 1):
        # a) Konstruktion von z_i^{(j)} = 1[y_i = j] − π_{ij}
        indicator = (y == j).astype(float)
        
        psi = X @ beta[j]
        omega = np.array([sample_pg_approx(psi_i) for psi_i in psi])
        Omega = np.diag(omega)
        
        XT_Omega_X = X.T @ Omega @ X
        Sigma_post = np.linalg.inv(XT_Omega_X + (1 / tau2) * np.eye(p))
        z_tilde = indicator - 0.5
        mu_post = Sigma_post @ (X.T @ z_tilde)
        beta[j] = multivariate_normal.rvs(mean=mu_post, cov=Sigma_post)
    
    beta_samples[t] = beta

# -------------------------------
# 3. Plot Posteriorverteilungen
# -------------------------------

fig, axes = plt.subplots(K - 1, p, figsize=(12, 8))
for j in range(K - 1):
    for d in range(p):
        axes[j, d].hist(beta_samples[:, j, d], bins=30, density=True, alpha=0.7, label=f"β{j+1},{d}")
        axes[j, d].axvline(beta_true[j, d], color='r', linestyle='--', label='true β')
        axes[j, d].set_title(f"Posterior von β{j+1},{d}")
        axes[j, d].legend()
plt.tight_layout()
plt.savefig("posterior_multinomial_beta.png")
plt.close()
