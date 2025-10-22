#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:51:51 2025

@author: toni
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

# gauß-Prozess: Sample generieren

def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """RBF Kernel"""
    sqdist = np.subtract.outer(x1, x2) ** 2
    return variance * np.exp(-0.5 * sqdist / length_scale**2)

# Erzeuge diskrete Punkte/Puxel
x = np.linspace(0, 10, 100)
K = rbf_kernel(x, x)

# Cholesky-Zerlegung
L = cholesky(K + 1e-6 * np.eye(len(K)), lower=True)  # Numerische Stabilität prüfen?

# Weißes Rauschen
z = np.random.randn(len(x))

# Korrelierte Samples durch GP
f = L @ z

# Visualisierung des GP-Samples
plt.figure(figsize=(10, 5))
plt.plot(x, f, label='Gauß-Prozess Sample')
plt.fill_between(x, -2, 2, color='gray', alpha=0.1, label='Unkorrelierte Schwankungen')
plt.legend()
plt.title("Beispiel für ein Gauß-Prozess-Sample")
plt.show()





# fuer spaeter Idee Kalman-Filter-Simulation

# Simulierte Zustände: Ein Objekt bewegt sich mit Beschleunigung
n_steps = 50
true_state = np.zeros((n_steps, 2))  # [Position, Geschwindigkeit]
measurements = np.zeros(n_steps)

dt = 1.0  # Zeitschritt
a = 0.5  # Beschleunigung
process_noise_std = 0.2
measurement_noise_std = 1.0

for t in range(1, n_steps):
    true_state[t, 0] = true_state[t-1, 0] + true_state[t-1, 1] * dt + 0.5 * a * dt**2
    true_state[t, 1] = true_state[t-1, 1] + a * dt
    measurements[t] = true_state[t, 0] + np.random.randn() * measurement_noise_std

# Kalman-Filter-Implementierung
x_est = np.zeros((n_steps, 2))  # Geschätzte Position und Geschwindigkeit
P = np.eye(2)  # Anfangskovarianz
F = np.array([[1, dt], [0, 1]])  # Zustandsübergangsmatrix
Q = np.array([[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]]) * process_noise_std**2  # Prozessrauschen
H = np.array([[1, 0]])  # Messmatrix
R = np.array([[measurement_noise_std**2]])  # Messrauschen

for t in range(1, n_steps):
    # Vorhersage
    x_pred = F @ x_est[t-1]
    P_pred = F @ P @ F.T + Q
    
    # Messupdate
    y = measurements[t] - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est[t] = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred

# Ergebnisse plotten
plt.figure(figsize=(10, 5))
plt.plot(range(n_steps), true_state[:, 0], label='Wahre Position', linestyle='dashed')
plt.plot(range(n_steps), measurements, 'ro', label='Messungen', markersize=3)
plt.plot(range(n_steps), x_est[:, 0], label='Gefilterte Position', linewidth=2)
plt.legend()
plt.title("Kalman-Filter Schätzung einer Bewegung")
plt.show()
