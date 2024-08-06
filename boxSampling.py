#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:15:50 2024

@author: toni

REJECTION SAMPLING
(ich nenn's liebevoll Boxsampling)
"""

import numpy as np
import plotly.graph_objects as go


# Definiere Dichtefunktion f(x)
def f(x):
    return x**2 * np.exp(-x**2 / 2)

# Funktion zur Erzeugung von n Stichproben, die Dichtefunktion f(x) folgen
def generate_samples(n):
    samples = []  # Initialisiere leere Liste
    while len(samples) < n:  # Wiederhole, bis n erreicht
        x = np.random.uniform(0, 10)  # ziehe eine Zufallszahl x aus gleichmäßiger Verteilung [0, 10]
        y = np.random.uniform(0, 1)   # Ziehe eine Zufallszahl y aus gleichmäßiger Verteilung [0, 1]
        if y <= f(x):  # Überprüfe, y kleiner/ gelich f(x) 
            samples.append(x)  #füge x der Liste der Stichproben hinzu
    return np.array(samples)  #gib die liste der Stichproben zurück

# Verwende die Funktion, um 1000 Zufallszahlen zu erzeugen, die der Dichtefunktion f(x) folgen
zufallszahlen = generate_samples(1000)


'''
ok laeuft und ist verstanden
'''


# Bereich von x-Werten für die Dichtefunktion
x_values = np.linspace(0, 10, 1000)
# Berechne die y-Werte der Dichtefunktion für die x-Werte
y_values = f(x_values)

# Erstelle ein Histogramm der generierten Zufallszahlen
histogram = go.Histogram(x=zufallszahlen, nbinsx=50, name='Generierte Stichproben', histnorm='probability density')

# Erstelle die Linie der Dichtefunktion
density_function = go.Scatter(x=x_values, y=y_values, mode='lines', name='Dichtefunktion f(x)', line=dict(color='red'))

# Erstelle das Layout für den Plot
layout = go.Layout(
    title='Histogramm der generierten Zufallszahlen und Dichtefunktion',
    xaxis=dict(title='x'),
    yaxis=dict(title='Dichte'),
    bargap=0.2,
)

# Erstelle die Figur und füge das Histogramm und die Dichtefunktion hinzu
fig = go.Figure(data=[histogram, density_function], layout=layout)

# Zeige die Figur an
fig.show()


import plotly.offline as pyo

# Speicher Plot als HTML
pyo.plot(fig, filename='zufallszahlen_plot.html')