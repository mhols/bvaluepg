'''
author: toni
created: 2025-05-04
last modified: 2025-05-16
'''

"""
Bayessche logistische Regression mit Polya-Gamma-Datenaugmentation


Problemstellung:
----------------
Wir betrachten ein binäres Klassifikationsproblem mit zwei Bins/Eimern.
Ein Experiment wirft 50 Bälle nacheinander, und jeder Ball landet zufällig
in einem der beiden Eimer (Bin 0 oder Bin 1). Die Wahrscheinlichkeit, dass
ein Ball in Bin 1 landet, hängt von einem Merkmalsvektor x_i ab.

Modell:
-------
- Zielvariable: y_i ∈ {0, 1}
- Merkmale: x_i ∈ R^p
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

#%%

import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
##@@@import py_polya

#%%

# -------------------------------
# Hilfsfunktion: Approximation von PG(1, z)
# -------------------------------

def sample_pg_approx(z, trunc=100):
    """
    Approximiert einen Polya-Gamma(1, z) Draw nach Devroye (2014),
    nicht vergessen-> macht Trunkierung der unendlichen Summe.
    """
    out = 0.0
    z = np.abs(z)
    for n in range(1, trunc + 1):
        lam = (n - 0.5) * np.pi
        out += np.random.gamma(1.0, 1.0) / (lam**2 + (z**2) / 4.0)
    return 0.25 * out

#%%

# -------------------------------
# 1. Simuliere die Daten
# -------------------------------

np.random.seed(42)
n = 50            # Anzahl Bälle / Beobachtungen
p = 1             # Anzahl Merkmale
X = np.random.randn(n, p)  # Zufällige Merkmalsmatrix - andere Featrures vorgeben
X = (X - X.mean(axis=0)) / X.std(axis=0)  # Standardisierung von X


for beta in [0.5,1.5,3.]:
    X= np.linspace(-2,2,50).reshape([50,1])
    # beta_true = np.array([1.5, -2.0])  # Wahres β
    beta_true = np.array([beta])
    
    logits = X @ beta_true
    
    probs = 1 / (1 + np.exp(-logits))  # σ(Xβ)
    y = np.random.binomial(1, probs)   # Binäre Zielvariablen
    
    
    
    plt.plot(X,probs,'.r')


#%%
plt.plot(y,'*b')

#%%

# nur Intercept beta0



# -------------------------------
# 2. Gibbs-Sampler Setup
# -------------------------------

n_iter = 500
burn_in = 100
beta_samples = np.zeros((n_iter, p))
beta = np.zeros(p)  # Initialwert
# anderen Startwert
beta = np.ones(p)
tau2 = 0.3         # Priorvarianz

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
    beta = np.atleast_1d(multivariate_normal.rvs(mean=mu_post, cov=Sigma_post))
    beta_samples[t, :] = beta

# -------------------------------
# 3. Plot Posteriorverteilungen
# -------------------------------

fig, axes = plt.subplots(1, p, figsize=(12, 4))
if p == 1:
    axes = [axes]
for j in range(p):
    axes[j].hist(beta_samples[:, j], bins=30, density=True, alpha=0.7, label=f"β{j}")
    axes[j].axvline(beta_true[j], color='r', linestyle='--', label='true β')
    axes[j].set_title(f"Posterior von β{j}")
    axes[j].legend()
# plt.tight_layout()
# plt.savefig("posterior_beta_bins.png")
plt.show()

# -------------------------------
# 4. Traceplot (nur für β0, falls p ≥ 1)
# -------------------------------
plt.figure(figsize=(10, 4))
plt.plot(beta_samples[:, 0])
plt.axvline(burn_in, color='r', linestyle='--', label='Burn-in-Grenze')
plt.title("Traceplot von β₀")
plt.xlabel("Iteration")
plt.ylabel("Wert von β₀")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Post-Burn-in-Samples für spätere Auswertung

beta_samples_post = beta_samples[burn_in:, :]

# --------------------------------------------
# Erklärung: Einfluss der Prior-Varianz τ²
# --------------------------------------------
# Die Prior-Varianz τ² steuert, wie stark unsere Annahme über β vor der Beobachtung der Daten gewichtet wird.
# Ein kleiner Wert (z. B. τ² = 0.3) führt zu einer stärkeren Regularisierung (Zentrierung von β um 0).
# Ist τ² zu klein, kann der Prior die Posteriorverteilung dominieren und β wird vom wahren Wert weggezogen.
# Ein größerer Wert (z. B. τ² = 10) erlaubt den Daten, einen größeren Einfluss auf die Posteriorverteilung zu haben.
# ?


# --------------------------------------------
# Erklärung: Notwendigkeit des Burn-ins
# --------------------------------------------
# Die ersten Iterationen des Gibbs-Samplers (Burn-in) spiegeln oft den Einfluss der Startwerte wider
# und sind noch nicht repräsentativ für die Zielverteilung.
# Durch Entfernen der ersten "burn_in" Iterationen (z. B. 100) verbessern wir die Qualität der Posterior-Samples.

# %%
