#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:49:18 2024

@author: toni
"""
#%%
'''
Idee1 Samplen der Ereignisanzahl

'''
import numpy as np

# Parameter
lambda_rate = 5  # Ereignisrate (λ)
T = 100  # Zeitintervall [0, T]

#Samplen der Ereignisanzahl
N = np.random.poisson(lambda_rate * T)  # zufaellige Anzahl der Ereignisse
events1 = np.sort(np.random.uniform(0, T, N))  # Ereigniszeitpunkte

print("Ereignisse mit Idee1:", events1)


#%%
'''
Idee2 Samplen der Ereigniszeiten
'''
t = 0
events2 = []
while t < T:
    t += np.random.exponential(1 / lambda_rate)
    if t < T:
        events2.append(t)

print("Ereignisse mit Ansatz 2:", events2)


#%%
'''
Spielerei Inverionsmethode
Idee3: nutze nicht np.random.poisson und np.random.exponentiell
Dichtefunktion exponential
f_X(x) = \lambda e^{-\lambda x}

\int f(x)

F_X(x) = 1 - e^{-\lambda x}

ziehe uniform/gleichverteilt U_X und setze gleich

U = 1 - e^{-\lambda x}   nach x umstellen

x = -\frac{1}{\lambda} \ln(1 - U)
'''

import numpy as np

# Parameter
lambda_rate = 5  # Ereignisrate
T = 100  # Zeitintervall [0, T]

# Funktion Generierung exponentiell verteilter Zufallszahlen
def toni_exponential(lambda_rate):
    U = np.random.uniform(0, 1)  # Gleichmäßig verteilte Zufallszahl
    return -np.log(1 - U) / lambda_rate

# Funktion Generierung Anzahl der Ereignisse
def toni_poisson(lambda_rate, T):
    events = []
    time = 0

    while True:
        # Zeit bis zum nächsten Ereignis, mit benutzerdefinierter Exponentialfunktion
        time_to_next_event = toni_exponential(lambda_rate)
        time += time_to_next_event

        if time > T:
            break

        events.append(time)

    return len(events), np.array(events)

# Nutzung der benutzerdefinierten Funktion
N, events1 = toni_poisson(lambda_rate, T)

# Sortieren der Ereigniszeitpunkte (falls nötig)
events1 = np.sort(events1)

print("Ereignisse mit IDee3:", events1)