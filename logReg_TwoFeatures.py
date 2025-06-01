# Bayessche logistische Regression mit zwei Features
# Autor: toni
# Datum: 2025-05-30

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid-Funktion
from pypolyagamma import PyPolyaGamma
from scipy.stats import gaussian_kde

#%%

# --- Beobachtungen ---
# Feature 1: Anzahl Werbeanzeigen
# Feature 2: Alter der Person
X = np.array([
    [1, 22],
    [2, 25],
    [3, 27],
    [3, 30],
    [4, 32],
    [5, 35],
    [6, 38],
    [7, 40],
    [8, 43],
    [9, 45]
])

# Zielvariable: Kaufentscheidung
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

#%%

# --- Hilfsfunktionen ---
def sigmoid(z):
    return expit(z)

def likelihood(beta1, beta2):
    z = X[:,0] * beta1 + X[:,1] * beta2
    p = sigmoid(z)
    return np.prod(p ** y * (1 - p) ** (1 - y))

def log_likelihood(beta1, beta2):
    z = X[:,0] * beta1 + X[:,1] * beta2
    p = sigmoid(z)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

#%%

# --- Log-Likelihood-Visualisierung ---
beta1_vals = np.linspace(-1.5, 1.5, 100)
beta2_vals = np.linspace(-0.2, 0.2, 100)
B1, B2 = np.meshgrid(beta1_vals, beta2_vals)

LL = np.array([[log_likelihood(b1, b2) for b1, b2 in zip(row1, row2)]
               for row1, row2 in zip(B1, B2)])

plt.figure(figsize=(8, 6))
cp = plt.contourf(B1, B2, LL, levels=30, cmap="viridis")
plt.colorbar(cp)
plt.xlabel(r'$\beta_1$ (Werbung)')
plt.ylabel(r'$\beta_2$ (Alter)')
plt.title('Log-Likelihood für zwei Feature-Koeffizienten')
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# --- Likelihood-Visualisierung (nicht log) ---
L = np.array([[likelihood(b1, b2) for b1, b2 in zip(row1, row2)]
              for row1, row2 in zip(B1, B2)])

plt.figure(figsize=(8, 6))
cp = plt.contourf(B1, B2, L, levels=30, cmap="plasma")
plt.colorbar(cp)
plt.xlabel(r'$\beta_1$ (Werbung)')
plt.ylabel(r'$\beta_2$ (Alter)')
plt.title('Likelihood für zwei Feature-Koeffizienten')
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# --- Gibbs-Sampling mit Pólya-Gamma (ohne Intercept) ---
def gibbs_sampler_pg(X, y, tau2=10.0, num_samples=1000, burnin=200):
    n, d = X.shape
    sampler = PyPolyaGamma()
    beta_samples = np.zeros((num_samples, d))
    beta = np.zeros(d)

    for t in range(num_samples + burnin):
        omega = np.array([sampler.pgdraw(1, X[i] @ beta) for i in range(n)])
        Omega = np.diag(omega)
        Sigma_inv = X.T @ Omega @ X + (1 / tau2) * np.eye(d)
        Sigma = np.linalg.inv(Sigma_inv)
        mu = Sigma @ X.T @ (y - 0.5)
        beta = np.random.multivariate_normal(mu, Sigma)
        if t >= burnin:
            beta_samples[t - burnin] = beta

    return beta_samples

#%%

# Gibbs-Sampling ausführen
samples = gibbs_sampler_pg(X, y, tau2=10.0, num_samples=1000)

#%%

# Posterior + Log-Likelihood als Konturplot mit Samples
plt.figure(figsize=(8, 6))
cp = plt.contourf(B1, B2, LL, levels=30, cmap="viridis")
plt.colorbar(cp)
plt.scatter(samples[:, 0], samples[:, 1], color='red', alpha=0.3, s=10, label="Posterior Samples")
plt.xlabel(r'$\beta_1$ (Werbung)')
plt.ylabel(r'$\beta_2$ (Alter)')
plt.title('Log-Likelihood + Posterior-Samples')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# --- Querschnitt durch Posterior bei festem beta_1 ---
fixed_beta1 = 0.8
beta2_grid = np.linspace(-0.2, 0.2, 300)
log_post_beta2 = [log_likelihood(fixed_beta1, b2) for b2 in beta2_grid]
post_beta2 = [likelihood(fixed_beta1, b2) for b2 in beta2_grid]

mask = (np.abs(samples[:, 0] - fixed_beta1) < 0.05)
kde_beta2 = gaussian_kde(samples[mask, 1]) if np.any(mask) else None

plt.figure(figsize=(7, 4))
plt.plot(beta2_grid, post_beta2, label="Likelihood bei $\\beta_1 = {:.2f}$".format(fixed_beta1))
if kde_beta2:
    plt.plot(beta2_grid, kde_beta2(beta2_grid) * max(post_beta2), label="Posterior-Dichte (skaliert)", linestyle="--")
plt.xlabel(r'$\beta_2$')
plt.title("Likelihood-Querschnitt für festes $\\beta_1$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# --- Querschnitt durch Posterior bei festem beta_2 ---
fixed_beta2 = 0.03
beta1_grid = np.linspace(-1.5, 1.5, 300)
log_post_beta1 = [log_likelihood(b1, fixed_beta2) for b1 in beta1_grid]
post_beta1 = [likelihood(b1, fixed_beta2) for b1 in beta1_grid]

mask2 = (np.abs(samples[:, 1] - fixed_beta2) < 0.01)
kde_beta1 = gaussian_kde(samples[mask2, 0]) if np.any(mask2) else None

plt.figure(figsize=(7, 4))
plt.plot(beta1_grid, post_beta1, label="Likelihood bei $\\beta_2 = {:.2f}$".format(fixed_beta2))
if kde_beta1:
    plt.plot(beta1_grid, kde_beta1(beta1_grid) * max(post_beta1), label="Posterior-Dichte (skaliert)", linestyle="--")
plt.xlabel(r'$\beta_1$')
plt.title("Likelihood-Querschnitt für festes $\\beta_2$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%