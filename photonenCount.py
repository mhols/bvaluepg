import numpy as np
import matplotlib.pyplot as plt
from pypolyagamma import PyPolyaGamma

#%%
# --- Parameter ---
rows, cols = 4, 4  # Schachbrettgrößex
poisson_lambda = 20  # Erwartungswert für Photonen pro Feld
noise_std = 2  # Standardabweichung für weißes Rauschen


#%%

# --- Schachbrettmuster erzeugen ---
# x= np.indices((rows, cols)).sum(axis=0) % 2 erzeugt ein Schachbrettmuster
# 0 für weiße Felder, 1 für schwarze Felder
# Hier verwenden wir zwei verschiedene Wahrscheinlichkeiten für die Felder
# z.B. 30% der Basisrate für weiße Felder und 100%
# für schwarze Felder, um ein Schachbrettmuster zu simulieren

x = np.indices((rows, cols)).sum(axis=0) % 2
pattern = np.where(x == 0, 0.3, 1.0)  # z.B. 30% und 100% der Basisrate

#%%

# --- Photonen-Counts simulieren ---
np.random.seed(42)  # Für Reproduzierbarkeit
photon_counts = np.random.poisson(lam=poisson_lambda * (1 + pattern))

#%%

# --- Weißes Rauschen hinzufügen ---
noise = np.random.normal(loc=0, scale=noise_std, size=photon_counts.shape)
counts_noisy = photon_counts + noise

# --- Linkfunktion anwenden (z. B. log) ---
# Hier wird eine Linkfunktion verwendet, um die Photonen-Counts zu transformieren
# In diesem Fall verwenden wir log(1+x), um negative Werte zu vermeiden
counts_link = np.log1p(np.maximum(counts_noisy, 0))  # log(1+x) und keine negativen Werte
# koennte ich spaeter zu sigmoid machen, wenn ich will



# --- Visualisierung ---
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(pattern, cmap='gray')
axs[0].set_title('Schachbrett-Muster')

axs[1].imshow(photon_counts, cmap='viridis')
axs[1].set_title('Photonen-Counts')

im = axs[2].imshow(counts_link, cmap='plasma')
axs[2].set_title('Linkfunktion (log(1+x))')

plt.colorbar(im, ax=axs[2])
plt.show()
#%%

# next step, verbindng mit raumkorrelation

# --- Erweiterung: Korrelation einbauen über Cholesky ---
from math import exp

def corr_function(x1, y1, x2, y2):
    """
        soll abhaengen von x1,y1 und x2 x2,y2
        muss positiv definit sein 
        Gaussglocke
    """
    sigma = 0.1
    dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return exp(-dist_sq / (2 * sigma ** 2))

#%%

# Gitter definieren
x_vals = np.linspace(0, 1, rows)
y_vals = np.linspace(0, 1, cols)
X, Y = np.meshgrid(x_vals, y_vals)
rX = X.ravel()
rY = Y.ravel()

#%%

# Kovarianzmatrix berechnen
Cova_matrix = np.zeros((len(rX), len(rY)))
for i in range(len(rX)):
    for j in range(len(rY)):
        Cova_matrix[i, j] = corr_function(rX[i], rY[i], rX[j], rY[j])

# Cholesky-Zerlegung und korreliertes Rauschen
L = np.linalg.cholesky(Cova_matrix + 1e-10 * np.eye(Cova_matrix.shape[0]))
random_vector = np.random.randn(Cova_matrix.shape[0])
correlated_noise = L @ random_vector
correlated_noise_image = correlated_noise.reshape(X.shape)

# Korrelierte Photonen-Counts simulieren
photon_counts_corr = np.random.poisson(lam=poisson_lambda * (1 + pattern) + correlated_noise_image)

# Visualisierung
plt.figure(figsize=(6,5))
plt.imshow(photon_counts_corr, cmap='magma')
plt.title('Photonen-Counts mit Raumkorrelation')
plt.colorbar()
plt.show()
#%%
# next step, add link function to correlated photon counts
# und teste verschiedene Wahrscheinlichkeiten im Schachbrettmuster
# aktuell ist es nur 0 und 1 und das bringt keine Korrelation

#%%
'''
polya gamma und gibs
'''
# next step, add polya gamma and gibbs sampling
# und teste verschiedene Wahrscheinlichkeiten im Schachbrettmuster
