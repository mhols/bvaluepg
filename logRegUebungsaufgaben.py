"""
Bayessche logistische Regression – Übungsaufgaben

Author: Toni Luhdo
Created: 2025-05-20

Dieses Skript enthält die Lösungen zu den Übungsaufgaben aus der Datei 'uebungsaufgaben.md'.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid-Funktion

# Beobachtungen
# x: Werbekontakte
# zweimal drei :) erinner dich kann alles sein
x = np.array([0, 1, 2, 3, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
n = len(x)
#%%
# Aufgabe 0 – Mittelwert und Varianz
mean_x = np.mean(x)
var_x = np.var(x, ddof=1)  # Stichprobenvarianz

print(f"Mittelwert x: {mean_x:.2f}")
print(f"Stichprobenvarianz x: {var_x:.2f}")

#%%
# Aufgabe 1 & 2 – Likelihood und log-Likelihood

# Eigene Sigmoidfunktion
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Likelihood-Funktion
def likelihood(beta):
    p = sigmoid(x * beta)
    return np.prod(p ** y * (1 - p) ** (1 - y))

def log_likelihood(beta):
    p = sigmoid(x * beta)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
#%%

# Aufgabe 5 – Visualisierung der Beobachtungen
plt.figure()
plt.scatter(x, y, color='black')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Beobachtungen: Werbekontakte vs. Reaktion")
plt.grid(True)
plt.tight_layout()
plt.show()

# Aufgabe 5 – Sigmoid-Fit über Beobachtungen
beta_hat = beta_vals[np.argmax(likelihood_vals)]
x_grid = np.linspace(min(x) - 1, max(x) + 1, 200)
p_pred = sigmoid(x_grid * beta_hat)

plt.figure()
plt.scatter(x, y, color='black', label='Beobachtungen')
plt.plot(x_grid, p_pred, color='blue', label=f'Sigmoid (β ≈ {beta_hat:.2f})')
plt.xlabel("x")
plt.ylabel("P(y=1 | x)")
plt.title("Sigmoid-Fit zur Modellierung von P(y=1 | x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Aufgabe 5 – Likelihood-Plot
beta_vals = np.linspace(-10, 10, 500)
likelihood_vals = np.array([likelihood(b) for b in beta_vals])
plt.plot(beta_vals, likelihood_vals)
plt.xlabel("β")
plt.ylabel("Likelihood")
plt.title("Likelihood-Verlauf")
plt.grid(True)
plt.tight_layout()
plt.show()

# Aufgabe 5 – Posterior (bis auf Normalisierung) plotten
log_lik_vals = np.array([log_likelihood(b) for b in beta_vals])
log_prior = np.where((beta_vals >= -10) & (beta_vals <= 10), 0, -np.inf)
log_post = log_lik_vals + log_prior

# Aufgabe 5 – Posterior (unnormalisiert) plotten
posterior = np.exp(log_post - np.max(log_post))  # für numerische Stabilität
plt.plot(beta_vals, posterior)
plt.xlabel("β")
plt.ylabel("Posterior (unnormalisiert)")
plt.title("Posterior-Verlauf (unnormalisiert)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.plot(beta_vals, log_post)
plt.xlabel("β")
plt.ylabel("log Posterior (unnormalisiert)")
plt.title("Posterior-Verlauf (bis auf Normierung)")
plt.grid(True)
plt.tight_layout()
plt.show()
#%%

# Aufgabe 6 – Gibbs-Sampling mit Pólya-Gamma (optional)
# ich muss die Vollständige Implementation Pólya-Gamma-Sampling mit externem Paket machen
# Hier nur Struktur des Algorithmus als Platzhalter:

# def gibbs_sampler_pg(...):
#     for iter in range(N):
#         # 1. Ziehe ω_i ∼ PG(1, x_i^T β)
#         # 2. Ziehe β ∼ N(m, V)
#     return beta_samples