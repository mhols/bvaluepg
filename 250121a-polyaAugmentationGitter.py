#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:51:34 2025

@author: toni
"""

import numpy as np
import matplotlib.pyplot as plt

# Eigene Implementierung der Polya-Gamma-Simulation
def sample_polya_gamma(b, c, size=1):
    trunc = 200
    result = np.zeros(size)
    for k in range(1, trunc + 1):
        rate = (k - 0.5) ** 2 + (c / (2 * np.pi)) ** 2
        result += np.random.exponential(1 / rate, size=size)
    result *= (np.pi ** 2) / 8
    return result

# Simulierte Erdbebendaten: 10x10 Gitter (nur x- und y-Koordinaten)
n_cells = 100  # Anzahl der Zellen

'''
TODOs
bedenke benachbarte Zellen ...
2D
noch kein ernst zunehmendes Vorwissen
verarbeite noch richtiges sampling
'''


p = 2  # Nur 2 Eigenschaften: x- und y-Koordinaten
np.random.seed(42)  # Reproduzierbarkeit

# Zufällige Eigenschaften für jede Zelle
X = np.random.normal(0, 1, (n_cells, p))  # Eigenschaften: x- und y-Koordinaten
beta_true = np.array([1.0, -1.0])  # Wahre Einflüsse der Eigenschaften
eta = X @ beta_true  # Linearkombination
p_y = 1 / (1 + np.exp(-eta))  # Wahrscheinlichkeiten für Erdbeben
y = np.random.binomial(1, p_y)  # Beobachtete Erdbeben (0 oder 1)

# Initialisierung
beta = np.zeros(p)
omega = np.ones(n_cells)

# parameterstart
Sigma0_inv = np.eye(p)  # Prior-Inverse-Kovarianzmatrix

# Gibbs-Sampling
n_iter = 100  # Anzahl der Iterationen
beta_samples = []  # Um die Posterior-Samples zu speichern

for t in range(n_iter):
    # 1. Ziehe ω aus Polya-Gamma-Verteilung
    eta = X @ beta
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
posterior_std_beta = beta_samples.std(axis=0)

# Ergebnisse anzeigen
print("Wahre Werte von Beta:", beta_true)
print("Posterior-Mittelwerte von Beta:", posterior_mean_beta)
print("Posterior-Standardabweichungen von Beta:", posterior_std_beta)

# Plotte die Schätzungen von Beta
plt.plot(beta_samples[:, 0], label="Beta 1 (x-Koordinate)")
plt.plot(beta_samples[:, 1], label="Beta 2 (y-Koordinate)")
plt.title("Verlauf der Beta-Schätzungen")
plt.xlabel("Iteration")
plt.ylabel("Beta-Wert")
plt.legend()
plt.show()

# Beispiel: Wahrscheinlichkeit für neue Gitterzellen berechnen
new_X = np.random.normal(0, 1, (5, p))  # Eigenschaften von 5 neuen Gitterzellen
new_eta = new_X @ posterior_mean_beta
new_p_y = 1 / (1 + np.exp(-new_eta))  # Wahrscheinlichkeiten für Erdbeben
print("Wahrscheinlichkeiten für neue Zellen:", new_p_y)