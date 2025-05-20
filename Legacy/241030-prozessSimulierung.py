#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:51:11 2024

@author: toni
"""


#%%%%%%%%
# lade bibs
import numpy as np
import matplotlib.pyplot as plt


#%%%

# Parameter für den Prozess
lambda_rate = 0.5  # Ereignisrate (λ), z.B., durchschnittliche Anzahl Event pro Zeiteinheit, naja oder Raum

# Anzahl der Events (zu simulieren)
num_events = 100

# 1. Zeitabstaende zwischen den Ereignissen
# Abstaende folgen einer Exponentialverteilung mit Erwartungswert 1/lambda_rate
# mach ich's mal zeitlich
time_intervals = np.random.exponential(1 / lambda_rate, num_events)

# 2. Ankunftszeiten Aufsummieren der Zeitabstände
arrival_times = np.cumsum(time_intervals)

# Plotten der Ergebnisse
plt.figure(figsize=(10, 6))
plt.step(arrival_times, np.arange(1, num_events + 1), where='post', color='b', label='Anzahl der Ereignisse')
plt.xlabel('Zeit')
plt.ylabel('Anzahl der Ereignisse')
plt.title('Simulation Poisson-Prozesses')
plt.grid(True)
plt.legend()
plt.show()

#%%%%%%%%

'''
Erzeugen Rechteck/Raster) bestimmten Größe 
simulieren die Anzahl der Ereignisse in jedem Bin (basierend aPoisson-Verteilung)
'''


# Parameter für das Gitter
grid_size = (20, 20)  # Anzahl der Kacheln (Bins) in x- und y-Richtung
lambda_rate = 2       # Durchschnittliche Ereignisrate pro Kachel

# Erzeuge eine leere Matrix für das Gitter
events_grid = np.random.poisson(lambda_rate, grid_size)

# Erstelle eine Grafik zur Darstellung des Gitters
plt.figure(figsize=(8, 8))
plt.imshow(events_grid, cmap='Blues', origin='upper')
plt.colorbar(label='Anzahl der Ereignisse')
plt.title('Simulation von Ereignissen in einem Gitter')
plt.xlabel('X-Koordinate')
plt.ylabel('Y-Koordinate')

# Anzahl der Ereignisse in jeder Kachel
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        plt.text(j, i, str(events_grid[i, j]), ha='center', va='center', color='black')

plt.show()


#%%%

'''
Um dies zu simulieren, können wir einen Prozess erstellen, bei dem wir Zeitabstände zwischen
 den Ereignissen simulieren und die kumulierten Ankunftszeiten berechnen. 
 Zusätzlich berechnen wir die erwartete Ankunftszeit und die tatsächliche 
 Ankunftszeit basierend auf den zufällig generierten Ereignissen.
 
1 Erstelle Prozess der Abstaende zwischen Ereignissen simuliert.
2 kumuliere Abstaende
3 berechne expected 
4 berechne tatsaechlich basierend auf generierten events
'''

# Parameter für den Prozess
lambda_rate = 0.5  # Ereignisrate, durchschnittliche Anzahl der Ereignisse pro Zeiteinheit

#ich mache mal Zeit
time_limit = 10    # Ende des Simulationszeit oder Raumgrenze

# 1. Simuliere die Zeitabstände zwischen Ereignissen
time_intervals = np.random.exponential(1 / lambda_rate, 100)  # Exponentialverteilte Zeitabstände
arrival_times = np.cumsum(time_intervals)  # Kumulierte Zeiten

# Begrenze die Zeiten auf den Simulationszeitraum
arrival_times = arrival_times[arrival_times < time_limit]

# 2. Berechne die erwartete Anzahl der Ereignisse
expected_events = lambda_rate * time_limit  # Erwartungswert basierend auf λ und der Zeit

# 3. Plotten der Ankunftszeiten und der erwarteten Anzahl
plt.figure(figsize=(10, 6))
plt.step(arrival_times, np.arange(1, len(arrival_times) + 1), where='post', color='b', label='Anzahl der Ereignisse')
plt.axhline(expected_events, color='r', linestyle='--', label=f'Erwartete Anzahl der Ereignisse ({expected_events})')
plt.xlabel('Zeit')
plt.ylabel('Kumulative Anzahl der Ereignisse')
plt.title('Simulation der Ankunftszeiten in einem Poisson-Prozess')
plt.grid(True)
plt.legend()
plt.show()

#%%%%%%%
'''
teste mal
Intensitaet Poisson-Prozesses ueber raeumliche oder zeitliche Domaene zu berechnen 
simuliere Szenario, Anzahl der Ereignisse in verschiedenen Zeitintervallen berechnen (Frequenz)
Funktion alpha(x)=exp(a(x)) Poissondichte der Ereignis (Anzahl der Ereignisse)

'''

# Parameter für zeitabh Modell
lambda_base = 0.5   # Basis-Ereignisrate
a = 0.1             # Wachstumskonstante für die Rate
time_intervals = 10 # Anzahl der Zeitintervalle
time_limit = 10     # Gesamtdauer

# Berechnung der zeitabh. Rate in jedem Intervall
time_points = np.linspace(0, time_limit, time_intervals + 1)
rates = lambda_base * np.exp(a * time_points)

# Simulation von Ereignissen in jedem Intervall
events_in_intervals = [np.random.poisson(rate * (time_points[i+1] - time_points[i])) for i, rate in enumerate(rates[:-1])]

# Berechnung der Frequenzen (Summe der Ereignisse in jedem Intervall)
frequencies = np.array(events_in_intervals)

# Plotten der Frequenzen
plt.figure(figsize=(10, 6))
plt.bar(range(time_intervals), frequencies, width=0.8, color='skyblue', label='Anzahl der Ereignisse')
plt.plot(range(time_intervals), rates[:-1] * (time_limit / time_intervals), 'r--', label='Theoretische Rate (skaliert)')
plt.xlabel('Zeitintervall')
plt.ylabel('Anzahl der Ereignisse')
plt.title('Simulation der Frequenz der Ereignisse in Zeitintervallen')
plt.grid(True)
plt.legend()
plt.show()

#%%%%%%%
'''
Kernfunktionen 
um die Abhängigkeit zwischen Parametern zu modellieren
Gaussian Process Regression (GPR)
Kernfunktion  - Korrelation zwischen Punkten beschreiben
Prior-Verteilung für geschätzte Werte 

'''

from sklearn.neighbors import KernelDensity

# Beispiel-Daten generieren
# zum testen randomseed nicht vergessen np.random.seed(0) 

data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])

# Bandbreite für die Kernel-Dichte-Schätzung
bandwidth = 0.5  # Breite des Kerns (bestimmt die Glättung)

# Kernel-Dichte-Schätzung durchführen
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
kde.fit(data[:, None])

# Erstelle einen Bereich für die x-Werte
x_values = np.linspace(-3, 8, 1000)[:, None]

# Berechne die Wahrscheinlichkeitsdichtefunktion (PDF) für jeden Punkt
log_density = kde.score_samples(x_values)
density = np.exp(log_density)  # Rücktransformation, da score_samples den Log-Wert liefert

# Plot der geschätzten Dichte und der Daten
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Daten-Histogramm')
plt.plot(x_values, density, color='blue', label='Kernel-Dichte-Schätzung')
plt.xlabel('Datenwerte')
plt.ylabel('Dichte')
plt.title('Kernel-Dichte-Schätzung mit Gaußschem Kern')
plt.legend()
plt.grid(True)
plt.show()


#%%%%%%
'''
bauen wri die kernel dichte schaetzung mal nach
'''

# Beispiel-Daten generieren
np.random.seed(0)
data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])

# Bandbreite für die Kernel-Dichte-Schätzung
bandwidth = 0.5  # Breite des Kerns (Glättungsparameter)

# Erstelle einen Bereich für die x-Werte
x_values = np.linspace(-3, 8, 1000)

# Gauss scher Kernel-Dichte-Schätzer
def gaussian_kernel_density(x, data, bandwidth):
    n = len(data)
    # Berechne den Gauss schen Kern für jeden Datenpunkt und jeden x-Wert
    kernels = np.exp(-0.5 * ((x[:, None] - data[None, :]) / bandwidth) ** 2)
    kernels /= (bandwidth * np.sqrt(2 * np.pi))
    # Summiere über alle Datenpunkte und berechne die Dichte
    density = kernels.sum(axis=1) / n
    return density

# Berechne die Dichte für jeden Punkt in x_values
density = gaussian_kernel_density(x_values, data, bandwidth)

# Plot der geschätzten Dichte und der Daten
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Daten-Histogramm')
plt.plot(x_values, density, color='blue', label='Kernel-Dichte-Schätzung')
plt.xlabel('Datenwerte')
plt.ylabel('Dichte')
plt.title('Kernel-Dichte-Schätzung mit Gaußschem Kern (ohne sklearn)')
plt.legend()
plt.grid(True)
plt.show()

