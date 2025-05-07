'''
irgendwas stimmt noch nicht.
'''

import numpy as np
from scipy.stats import gamma
from scipy.special import expit

# Funktion, um aus der Pólya-Gamma-Verteilung zu samplen
# das muss ich nochmal gegenchecken

def sample_pg(b, c, trunc=200):
    """
    Simuliert eine Pólya-Gamma-Zufallsvariable.
    b: Formparameter der Gamma-Verteilung
    c: Tilting-Parameter
    trunc: Anzahl der Summanden in der unendlichen Serie
    """
    omega = 0
    for k in range(1, trunc + 1):
        # Gewichteter Beitrag jedes Terms
        coef = 1 / ((k - 0.5) ** 2 + (c / (2 * np.pi)) ** 2)
        omega += gamma.rvs(b, scale=coef)
    return omega

#%%

# Simulation von Daten zum Testen
# muss noch gegen Sampler von 241019 ausgetaucht werden
np.random.seed(42)

N = 100  # Anzahl der Beobachtungen
p = 2    # Anzahl der Prädiktoren






# Design-Matrix X
# einsen und Zufallsnormal verteilt
X = np.hstack([np.ones((N, 1)), np.random.normal(0, 1, size=(N, p - 1))])

#%%



# "wahre" Parameter für Simulation
beta_true = np.array([0.5, -1.0])  # Wahre Betas
psi = X @ beta_true                # Log-Odds
p_true = expit(psi)                # Wahre Wahrscheinlichkeiten

# Generiere binäre Zielvariable y
y = np.random.binomial(1, p_true)

# Bayesianische Prior-Parameter
B_inv = np.eye(p) * 1  # Prior-Präzision (Diagonalmatrix)
mu_prior = np.zeros(p) # Prior-Mittelwert

# Gibbs-Sampling Parameter
n_iter = 1000   # Anzahl der Iterationen
burn_in = 200   # Burn-In Period

# Initialisierung der Parameter
beta = np.zeros(p)       # Startwert für Beta
samples_beta = []        # Speichere Beta-Samples

# Gibbs-Sampling
for i in range(n_iter):
    # 1. Berechne ψ
    psi = X @ beta
    
    # 2. Sample ω (Pólya-Gamma)
    omega = np.array([sample_pg(1, psi[j]) for j in range(N)])
    Omega = np.diag(omega)  # Diagonalmatrix
    
    # 3. Berechne Posterior-Prior-Präzision
    V_inv = X.T @ Omega @ X + B_inv
    V = np.linalg.inv(V_inv)  # Posterior-Kovarianzmatrix
    
    # 4. Berechne Posterior-Mittelwert
    kappa = y - 0.5
    mu_post = V @ (X.T @ kappa + B_inv @ mu_prior)
    
    # 5. Sample β aus multivariater Normalverteilung
    beta = np.random.multivariate_normal(mu_post, V)
    
    # Speichere die Werte nach Burn-In
    if i >= burn_in:
        samples_beta.append(beta)

# Konvertiere Samples in numpy-Array
samples_beta = np.array(samples_beta)

# Posterior-Analyse
mean_beta = samples_beta.mean(axis=0)
ci_95 = np.percentile(samples_beta, [2.5, 97.5], axis=0)

# Ergebnisse ausgeben
print("Posterior Mittelwerte von Beta:", mean_beta)
print("95% Konfidenzintervalle von Beta:", ci_95)

# Visualisierung der Posterior-Samples
import matplotlib.pyplot as plt
for j in range(p):
    plt.hist(samples_beta[:, j], bins=30, density=True, alpha=0.6, label=f"Beta {j}")
    plt.axvline(beta_true[j], color='r', linestyle='--', label=f"Wahrer Wert Beta {j}")
    plt.title(f"Posterior von Beta {j}")
    plt.legend()
    plt.show()