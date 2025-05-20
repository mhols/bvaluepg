#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:30:52 2024

@author: toni
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Parameter
lambdaa = 1.0  # Wert lambda für Exponentialverteilung
b = np.log(10) * lambdaa  # Wert b in log_10-Skala

# Anzahl der synthetischen Daten
N = 1000

# 1. Synthetische Magnituden generieren
M_i = np.random.exponential(scale=1/lambdaa, size=N)

# 2. Wert von beta (nach Definition)
beta = b

# 3. Maximum-Likelihood-Schartzung für beta (1/mean von M_i)
beta_MLE = 1 / np.mean(M_i)

# 4. Definiere die Prior für beta (Gamma-Verteilung als Beispiel)
# Nehme an, dass Prior Gamma verteilt ist, mit Parametern alpha und beta_prior
alpha_prior = 2.0
beta_prior = 1.0

# 5. Likelihood-Funktion
def likelihood(beta, M):
    return np.prod(beta * np.exp(-beta * M))

# Posterior-Schätzung fuer beta (Bayesian Update)
# Verwende die Konjugierte Gamma-Verteilung
alpha_post = alpha_prior + N
beta_post = beta_prior + np.sum(M_i)

# Posterior-Schaetzung für beta
beta_posterior_mean = alpha_post / beta_post

# 7. Plot der Prior und Posterior
x = np.linspace(0.01, 5, 100)
prior = gamma.pdf(x, a=alpha_prior, scale=1/beta_prior)
posterior = gamma.pdf(x, a=alpha_post, scale=1/beta_post)

plt.plot(x, prior, label='Prior (Gamma)')
plt.plot(x, posterior, label='Posterior (Gamma)', color='r')
plt.axvline(beta_MLE, color='g', linestyle='--', label='MLE von beta')
plt.axvline(beta_posterior_mean, color='b', linestyle='--', label='Posterior Mean von beta')

# bissel schick machen
plt.legend()
plt.xlabel('beta')
plt.ylabel('Density')
plt.title('Prior und Posterior von beta')
plt.show()

# Ergebnise
print(f"Wahrer Wert von beta: {beta:.4}")
print(f"MLE von beta: {beta:.4}")
print(f"Posterior Mean von beta: {beta:.4}")