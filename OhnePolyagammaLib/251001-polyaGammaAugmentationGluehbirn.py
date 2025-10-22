#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Fri 14 11:41:36 2025

@author: toni
"""

import numpy as np
# from polya_gamma import PolyaGamma

# Implementierung der Polya-Gamma-Simulation
def sample_polya_gamma(b, c, size=1):
    """Simuliert Polya-Gamma-Zufallsvariablen."""
    trunc = 200  # Anzahl der Terme in der Summe
    result = np.zeros(size)
    for k in range(1, trunc + 1):
        rate = (k - 0.5) ** 2 + (c / (2 * np.pi)) ** 2
        result += np.random.exponential(1 / rate, size=size)
    result *= (np.pi ** 2) / 8
    return result





# Simulierte Daten
n, p = 1000, 3
X = np.random.normal(0, 1, (n, p))  # Eigenschaften der Glühbirnen
beta_true = np.array([0.5, -1.0, 0.8])  # Parameter
eta = X @ beta_true
p_y = 1 / (1 + np.exp(-eta))
y = np.random.binomial(1, p_y)  # Zustand der Glühbirnen

# Initialisierung
beta = np.zeros(p)
omega = np.ones(n)
# pg_sampler = PolyaGamma(seed=42)

# Hyperparameter
Sigma0_inv = np.eye(p)  # Prior-Kovarianzmatrix

# Gibbs-Sampling
# n_iter = 5000
n_iter = 50
beta_samples = []  # Um Posterior-Samples zu speichern

for t in range(n_iter):
    # 1. Ziehe ω aus Polya-Gamma-Verteilung
    eta = X @ beta
    # omega = np.array([pg_sampler.pgdraw(1, eta_i) for eta_i in eta])
    omega = np.array([sample_polya_gamma(1, eta_i, size=1)[0] for eta_i in eta])
    
    # 2. Aktualisiere β
    Omega = np.diag(omega)
    Sigma_beta = np.linalg.inv(X.T @ Omega @ X + Sigma0_inv)
    mu_beta = Sigma_beta @ X.T @ (y - 0.5)
    beta = np.random.multivariate_normal(mu_beta, Sigma_beta)
    
    # Speichere das aktuelle Sample
    beta_samples.append(beta)
    
# Konvertiere die Samples in ein Array für weitere Analysen
beta_samples = np.array(beta_samples)


# Ergebnisse
posterior_mean_beta = beta_samples.mean(axis=0)
print("Posterior Mean von Beta:", posterior_mean_beta)


# Beispiel: Konvergenzprüfung (z. B. Verlauf der ersten Beta-Komponente)
import matplotlib.pyplot as plt

plt.plot(beta_samples[:, 0])
plt.title("Verlauf der ersten Beta-Komponente")
plt.xlabel("Iteration")
plt.ylabel("Beta-Wert")
plt.show()


### Rechenbeispiel manuelll




'''
Die Posterior-Verteilung beschreibt jetzt meine Unsicherheit über die Parameter beta_1, beta_2
 nach Berücksichtigung der Daten. Im Kontext, wie wahrscheinlich bestimmte Werte der Parameter in Anbetracht
 der beobachteten Glühbirnenzustände und ihrer Eigenschaften sind.
'''
#%%

n, p = 10, 2  # 10 Glühbirnen, 2 features



###  Werte für Spannung und Drahtdicke (standar.)

X_birnen = np.array([[0.497, -0.138], [0.648, 1.523], [-0.234, -0.234],[
1.579, 0.767],[
-0.469, 0.543],[
-0.463,-0.466],[
0.242, -1.913],[
-1.725, -0.562],[
-1.013, 0.314],[
-0.908, -1.412]])
    
# waehle beta
# Idee :)
# beta_1 = 0.8: positive Spannung  stärkeren Einfluss darauf, dass eine Glühbirne kaputtgeht
# beta_2 = -0.5: dickere Draht reduziert die Wahrscheinlichkeit, dass die Glühbirne kaputtgeht

beta_true2 = np.array([0.8, -0.5])
eta2 = X_birnen @ beta_true2

# p_y = 1 / (1 + np.exp(-eta))  # Wahrscheinlichkeiten für y
# y = np.random.binomial(1, p_y)  # Zustand der Glühbirnen (0: heil, 1: kaputt)

y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 1])


# Initialisierung - Start
beta = np.zeros(2)
omega = np.ones(10)

# Parameter
Sigma0_inv = np.eye(2)  # Prior-Inverse-Kovarianzmatrix fuer 2 features



# Gibbs-Sampling
n_iter = 10  # Anzahl der Iterationen
beta_samples = []  # Um die Posterior-Samples zu speichern

for t in range(n_iter):
    # 1. Ziehe omega aus Polya-Gamma-Verteilung (siehe funtkion oben)
    eta = X_birnen @ beta
    omega = np.array([sample_polya_gamma(1, eta_i, size=1)[0] for eta_i in eta])
    
    # 2. Aktualisiere β
    Omega = np.diag(omega)
    Sigma_beta = np.linalg.inv(X_birnen.T @ Omega @ X_birnen + Sigma0_inv)
    mu_beta = Sigma_beta @ X_birnen.T @ (y - 0.5)
    beta = np.random.multivariate_normal(mu_beta, Sigma_beta)
    
    # # Speichere das aktuelle Sample
    beta_samples.append(beta)
    
# Konvertiere die Samples in Array
beta_samples = np.array(beta_samples)

# Ergebnisse
posterior_mean_beta = beta_samples.mean(axis=0)
posterior_std_beta = beta_samples.std(axis=0)

# Verlauf der Beta-Komponenten
plt.plot(beta_samples[:, 0], label="Beta 1 (Spannung)")
plt.plot(beta_samples[:, 1], label="Beta 2 (Drahtdicke)")
plt.title("Verlauf der Beta-Komponenten")
plt.xlabel("Iteration")
plt.ylabel("Beta-Wert")
plt.legend()
plt.show()

# Ergebnisse ausgeben
print("Wahre Werte von Beta:", beta_true)
print("Posterior-Mittelwerte von Beta:", posterior_mean_beta)
print("Posterior-Standardabweichungen von Beta:", posterior_std_beta)
print("Zustände der Glühbirnen (y):", y)
print("Eigenschaften der Glühbirnen (X):\n", X)

#%%
