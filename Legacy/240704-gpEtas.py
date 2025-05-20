#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 4 17:17:47 2024

@author: toni
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Definition der Triggerfunktion
def triggering_function(t, x, y, t_i, x_i, y_i, m_i, params):
    g_m = params['g'](m_i)
    h_t = params['h'](t - t_i)
    f_xy = params['f'](x - x_i, y - y_i)
    return g_m * h_t * f_xy

# Implementierung der Komponenten der Triggerfunktion
# Test Exp funktion
def g(m):
    return np.exp(m)

def h(t):
    return np.exp(-t)

def f(x, y):
    return np.exp(-(x**2 + y**2))

# Definition Hintergrundintensitaet mittels Gaußschen Prozess
def background_intensity_gp(X, Y, gp):
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    mu, sigma = gp.predict(xy, return_std=True)
    return mu.reshape(X.shape), sigma.reshape(X.shape)

# Bedingte Intensitätsfunktion
def conditional_intensity(t, x, y, history, background_intensity, params, gp):
    mu, _ = background_intensity_gp(np.array([[x]]), np.array([[y]]), gp)
    intensity = mu[0][0]
    
    for event in history:
        t_i, x_i, y_i, m_i = event
        intensity += triggering_function(t, x, y, t_i, x_i, y_i, m_i, params)
    
    return intensity

# Beispiel historische Daten (t, x, y, m)
history = [
    (1, 0.5, 0.5, 2.0),
    (2, 1.0, 1.0, 3.0),
    (3, 1.5, 1.5, 2.5)
]

# Parameter für die Triggerfunktion
params = {
    'g': g,
    'h': h,
    'f': f
}

# Beispeil Daten für den Gaußschen Prozess
X_train = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
y_train = np.array([0.2, 0.4, 0.6])

# Definition Gaußschen Prozesses
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
gp.fit(X_train, y_train)

# Berechnung der bedingten Intensitaet an einem bestimmten Punkt (t, x, y)
t = 4
x = 1.0
y = 1.0
intensity = conditional_intensity(t, x, y, history, background_intensity_gp, params, gp)

print(f'Bedingte Intensität bei (t={t}, x={x}, y={y}): {intensity}')
