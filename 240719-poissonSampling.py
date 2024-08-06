#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:49:18 2024

@author: toni
"""

import numpy as np

# Parameter
lambda_rate = 5  # Ereignisrate (λ)
T = 100  # Zeitintervall [0, T]

# Idee 1: Samplen der Ereignisanzahl
N = np.random.poisson(lambda_rate * T)  # zufaellige Anzahl der Ereignisse
events1 = np.sort(np.random.uniform(0, T, N))  # Ereigniszeitpunkte

print("Ereignisse mit Ansatz 1:", events1)



# Idee 2: Samplen der Ereigniszeiten
t = 0
events2 = []
while t < T:
    t += np.random.exponential(1 / lambda_rate)
    if t < T:
        events2.append(t)

print("Ereignisse mit Ansatz 2:", events2)


# hmm da muss ich nochmal drueber
