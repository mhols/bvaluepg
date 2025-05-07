#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 04:36:25 2025

@author: toni
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def polya_gamma_density(x, b, c, k_max=50):
    """
    Berechnet die Dichtefunktion der Polya-Gamma Verteilung für gegebene b und c.
    Die Berechnung basiert auf der unendlichen Summe, die auf k_max Terme begrenzt wird.
    
    Parameters:
        x (array): Werte, für die die Dichte berechnet wird.
        b (float): Parameter b der Polya-Gamma-Verteilung.
        c (float): Parameter c der Polya-Gamma-Verteilung.
        k_max (int): Anzahl der Summationstermen für die Approximation.
    
    Returns:
        Array der Dichtewerte für die eingegebenen x.
    """
    density = np.zeros_like(x)
    
    for k in range(k_max):
        coef = (-1)**k * (2*k + b) * gamma(k + b) / (gamma(k + 1) * gamma(b))
        term = coef * (x**(k + b - 1) * np.exp(-x * c**2 / 2)) / (2**(2*k + b))
        density += term
    
    return density

# Parameter setzen
b = 1.0  # Beispielwert für b
c = 0.0  # Beispielwert für c
x_values = np.linspace(0.01, 5, 500)  # Wertebereich für x
density_values = polya_gamma_density(x_values, b, c)

# Plot der Dichtefunktion
plt.figure(figsize=(8, 6))
plt.plot(x_values, density_values, label=f"PG({b}, {c})", color='b', linewidth=2)
plt.title(f"Polya-Gamma Dichtefunktion für b={b}, c={c}")
plt.xlabel("x")
plt.ylabel("Dichte f(x)")
plt.legend()
plt.grid()
plt.show()