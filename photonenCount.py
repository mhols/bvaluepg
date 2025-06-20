import numpy as np
import matplotlib.pyplot as plt
#%%
# --- Parameter ---
rows, cols = 32, 32  # Schachbrettgröße
poisson_lambda = 20  # Erwartungswert für Photonen pro Feld
noise_std = 2  # Standardabweichung für weißes Rauschen


#%%

# --- Schachbrettmuster erzeugen ---
x = np.indices((rows, cols)).sum(axis=0) % 2
pattern = x  # 0 und 1 als Schachbrettmuster

#%%

# --- Photonen-Counts simulieren ---
np.random.seed(42)  # Für Reproduzierbarkeit
photon_counts = np.random.poisson(lam=poisson_lambda * (1 + pattern))

#%%

# --- Weißes Rauschen hinzufügen ---
noise = np.random.normal(loc=0, scale=noise_std, size=photon_counts.shape)
counts_noisy = photon_counts + noise

# --- Linkfunktion anwenden (z. B. log) ---
counts_link = np.log1p(np.maximum(counts_noisy, 0))  # log(1+x) und keine negativen Werte

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
# %%

# next step, verbindng mit raumkorrelation