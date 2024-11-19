#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:37:31 2024

@author: toni
"""

import numpy as np
import matplotlib.pyplot as plt

# Kovarianzfunktion (RBF-Kernel)
def rbf_kernel(x1, x2, length_scale=0.2, variance=1.0):
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return variance * np.exp(-0.5 * sqdist / length_scale**2)

# Daten für Regression
np.random.seed(42)
X_train = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train).ravel() + 0.1 * np.random.randn(len(X_train))
X_test = np.linspace(0, 1, 100).reshape(-1, 1)

# Hyperparameter des GP
length_scale = 0.2
variance = 1.0
noise_variance = 0.01

# Berechnung der Kovarianzmatrizen
K = rbf_kernel(X_train, X_train, length_scale, variance) + noise_variance * np.eye(len(X_train))
K_s = rbf_kernel(X_train, X_test, length_scale, variance)
K_ss = rbf_kernel(X_test, X_test, length_scale, variance)

# Posterior-Berechnungen
K_inv = np.linalg.inv(K)
mu_s = K_s.T @ K_inv @ y_train
cov_s = K_ss - K_s.T @ K_inv @ K_s
std_s = np.sqrt(np.diag(cov_s))

# Visualisierung der Regression
plt.figure(figsize=(10, 6))
plt.plot(X_test, np.sin(2 * np.pi * X_test), 'r:', label="True function")
plt.plot(X_train, y_train, 'r.', markersize=10, label="Training points")
plt.plot(X_test, mu_s, 'b-', label="Mean prediction")
plt.fill_between(X_test.ravel(), mu_s - 2 * std_s, mu_s + 2 * std_s, color="blue", alpha=0.2, label="Confidence interval (2 std)")
plt.title("Gaussian Process Regression: Prior and Posterior")
plt.xlabel("Input (x)")
plt.ylabel("Output (f(x))")
plt.legend()
plt.show()

# Klassifikationsaufgabe
from scipy.special import expit  # Sigmoid-Funktion für Wahrscheinlichkeiten

# Labels für Klassifikation (1 und -1)
y_class = np.array([1, -1, 1, -1])

# Posterior-Berechnungen für Klassifikation
mu_class = K_s.T @ K_inv @ y_class
std_class = np.sqrt(np.diag(cov_s))
prob_class = expit(mu_class)  # Logistische Funktion

# Visualisierung der Klassifikation
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_class, 'ro', label="Training points (class labels)")
plt.plot(X_test, prob_class, 'b-', label="Probability prediction (sigmoid)")
plt.fill_between(X_test.ravel(), prob_class - 2 * std_class, prob_class + 2 * std_class, color="blue", alpha=0.2, label="Confidence interval (2 std)")
plt.title("Gaussian Process Classification with Sigmoid Function")
plt.xlabel("Input (x)")
plt.ylabel("Probability P(class=1)")
plt.legend()
plt.show()

#%%
'''
das laeuft mit sklearn wahrscheinlich schneller
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Daten für Regression generieren
np.random.seed(42)
X_train = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train).ravel() + 0.1 * np.random.randn(len(X_train))
X_test = np.linspace(0, 1, 100).reshape(-1, 1)

# Kernel definieren: Kombination aus RBF (Radial Basis Function) und konst. Kern
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2, n_restarts_optimizer=10)

# GP-Modell trainieren
gp.fit(X_train, y_train)

# Vorhersagen mit Posterior-Verteilung
y_pred, sigma = gp.predict(X_test, return_std=True)

# Visualisierung
plt.figure(figsize=(10, 6))
plt.plot(X_test, np.sin(2 * np.pi * X_test), 'r:', label="True function")
plt.plot(X_train, y_train, 'r.', markersize=10, label="Training points")
plt.plot(X_test, y_pred, 'b-', label="Mean prediction")
plt.fill_between(X_test.ravel(), 
                 y_pred - 2 * sigma, 
                 y_pred + 2 * sigma, 
                 color="blue", alpha=0.2, label="Confidence interval (2 std)")

plt.title("Gaussian Process Regression: Prior and Posterior")
plt.xlabel("Input (x)")
plt.ylabel("Output (f(x))")
plt.legend()
plt.show()

# Klassifikationsaufgabe mit Sigmoid-Funktion
from scipy.special import expit  # Logistische Funktion

# Daten für Klassifikation
X_class = np.array([[0.2], [0.4], [0.6], [0.8]])
y_class = np.array([1, -1, 1, -1])

# GP für Klassifikation
gp_class = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2, n_restarts_optimizer=10)
gp_class.fit(X_class, y_class)

# Vorhersagen
y_class_pred, sigma_class = gp_class.predict(X_test, return_std=True)
prob = expit(y_class_pred)  # Logistische Funktion anwenden

# Visualisierung der Klassifikation
plt.figure(figsize=(10, 6))
plt.plot(X_class, y_class, 'ro', label="Training points (class labels)")
plt.plot(X_test, prob, 'b-', label="Probability prediction (sigmoid)")
plt.fill_between(X_test.ravel(), 
                 prob - 2 * sigma_class, 
                 prob + 2 * sigma_class, 
                 color="blue", alpha=0.2, label="Confidence interval (2 std)")

plt.title("Gaussian Process Classification with Sigmoid Function")
plt.xlabel("Input (x)")
plt.ylabel("Probability P(class=1)")
plt.legend()
plt.show()


