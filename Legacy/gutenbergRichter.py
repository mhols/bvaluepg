#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:50:55 2024

@author: toni
"""

import matplotlib.pyplot as plt
import numpy as np

# Parameter für die Gutenberg-Richter-Verteilung
a = 4
b = 1

# Erdbebenmagnitude von 0 bis 9
M = np.arange(0, 10, 0.1)
# Berechnung der Anzahl der Erdbeben mit Magnitude größer oder gleich M
N = 10**(a - b * M)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(M, N, label=r'$\log_{10} N(M) = a - bM$')
# plt.yscale('log')
plt.xlabel('Magnitude (M)')
plt.ylabel('Anzahl der Erdbeben $N(M)$')
plt.title('Gutenberg-Richter-Verteilung')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()