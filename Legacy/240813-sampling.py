#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:35:57 2024

@author: toni
"""

import math
import random

def generate_poisson_random(lambda_value):
    # Schritt 1: Berechne L = exp(-λ)
    L = math.exp(-lambda_value)
    
    # Schritt 2: Initialisiere k = 0 und p = 1
    k = 0
    p = 1
    
    # Schleife zur Generierung der Zufallszahl
    while p > L:
        # Schritt 5: Erhöhe k um 1
        k += 1
        
        # Schritt 3 und 4: Generiere eine Zufallszahl u und multipliziere p mit u
        p *= random.random()
    
    # Schritt 7: Rückgabe der Anzahl der Ereignisse
    return k - 1

# Beispielaufruf
lambda_value = 3.0  # Erwartungswert
random_numbers = [generate_poisson_random(lambda_value) for _ in range(10)]

print(random_numbers)