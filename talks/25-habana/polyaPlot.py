#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 01:57:05 2025

@author: toni
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#%%

# Werte für die Parameter b und c
b_values = [1, 2, 5]  # Verschiedene Werte für b
c_values = [0, 1, 2]  # Verschiedene Werte für c


#%%

# Erzeugung der x-Werte für die Verteilung
x = np.linspace(0, 5, 1000)

# Farben für die Plots
colors = ['blue', 'green', 'red']

# Erstellen der Plots für verschiedene b- und c-Werte
plt.figure(figsize=(8, 6))
for i, b in enumerate(b_values):
    for j, c in enumerate(c_values):
        # Näherung der Polya-Gamma-Dichte mit einer Gamma-Verteilung
        shape = b  # Formparameter (b)
        scale = 1 / (c**2 + 1)  # Skalenparameter (vereinfachte Approximation)
        
        pdf_values = stats.gamma.pdf(x, shape, scale=scale)  # Gamma-Dichte als Approximation
        plt.plot(x, pdf_values, label=f'b={b}, c={c}', color=colors[i], linestyle=['-', '--', ':'][j])

plt.xlabel(r'$\omega$')
plt.ylabel('Dichtefunktion')
plt.title('Polya-Gamma Verteilung für verschiedene b- und c-Werte')
plt.legend()
plt.grid(True)

# Speichern des Bildes
plt.savefig("polya_gamma_distribution.png")
plt.show()
# %%