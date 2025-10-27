'''
neue Version mit grösserer Abweichung
Damit die Variation im Poissonprozess deutlicher wird.
erzeuge schachbrettmuster 30 mal 30 mit 5*5 feldern
jede zelle 5x5 pixel
jede zelle hat einen basiswert + abweichung*pattern
'''

#%% Imports

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import polyagamma as polyga

#%% create pattern
# --- Parameter ---
rows, cols = 30, 30
poisson_lambda = 20 # Erwartungswert pro Feld
noise_std = 2
np.random.seed(42)

x = np.indices((rows, cols)).sum(axis=0) % 2
pattern = np.where(x == 0, 0.3, 1.0)


for i in range(rows):
    ii=(i // 5)%2
    for j in range(cols):
        jj=(j // 5)%2
        if (ii + jj)%2 == 0:
            pattern[i, j] = 0
        else:
            pattern[i, j] = 1

#plot pattern
plt.figure(figsize=(6, 5))
plt.imshow(pattern, cmap="gray")
plt.title("Schachbrettmuster")
plt.xlabel("Spalte")
plt.ylabel("Zeile")
plt.colorbar(label="Musterwert")
plt.show()  

#%%
# Poisson-Verteilung base rate + bild mit Schachbrettmuster
#base_rate = poisson_lambda * (1 + pattern)

base_rate = 100
deviation = 40  # darf nicht zu klein sein

feld=base_rate+deviation*pattern
# plot feld
plt.figure(figsize=(6, 5))
plt.imshow(feld, cmap="viridis")
plt.title("Photonenrate pro Feld (λ) mit Abweichung")
plt.xlabel("Spalte")
plt.ylabel("Zeile")
plt.colorbar(label="Photonenrate λ")
plt.show()
#%%
# ergebnis ist gleich np.zeros like feld

ergebnis = np.zeros_like(feld)
for i in range(rows):
    for j in range(rows):
        ergebnis[i,j]=np.random.poisson(feld[i,j],size=1)
# plot ergebnis
plt.figure(figsize=(6, 5))
plt.imshow(ergebnis, cmap="viridis")
plt.title("Anzahl detektierter Photonen pro Feld")
plt.xlabel("Spalte")
plt.ylabel("Zeile")

# %%
# sample posterior aus ergebnis
# iteration mindestens 100 mal
# 1. prior definieren
# 2. likelihood definieren
# 3. posterior berechnen
# 4. sample aus posterior ziehen
# wird ein ziehen aus 1 dimensionaler gaussverteilung
