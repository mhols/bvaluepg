# Logistische Regression mit 4 Klassen (One-vs-Rest)
# Autor: Toni Luhdo
# Modell: 1 Feature, 4 Klassen, One-vs-Rest


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid
from scipy.stats import gaussian_kde

#%%
# --- Daten (x: Werbeanzeigen, y: Klasse) ---
x = np.array([0, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 10, 10])
y = np.array([0, 0, 1, 1, 2, 2, 1, 1, 2, 3, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3])

#%%
# --- Klassenlabels (One-vs-Rest Targets) ---
K = 4
n = len(x)
y_ovr = np.zeros((K, n))
for k in range(K):
    y_ovr[k] = (y == k).astype(int)

#%%
# --- Pólya-Gamma Gibbs-Sampler für One-vs-Rest ---
from pypolyagamma import PyPolyaGamma

def gibbs_pg_1d(x, y_bin, tau2=10.0, num_samples=1000, burnin=200):

#%%
# --- Sampling durchführen für alle Klassen ---
beta_samples = []
for k in range(K):
    beta_k = gibbs_pg_1d(x, y_ovr[k], tau2=10.0)
    beta_samples.append(beta_k)

#%%
# --- Vorhersagewahrscheinlichkeiten ---
x_new = np.array([0, 2, 5, 8, 10])
probs = np.zeros((len(x_new), K))

for k in range(K):
    beta_mean = np.mean(beta_samples[k])
    probs[:, k] = expit(x_new * beta_mean)

# --- Normalisieren, sodass Zeilensummen 1 ergeben ---
probs /= probs.sum(axis=1, keepdims=True)

# # --- Visualisierung ---
# plt.figure(figsize=(8, 5))
# for k in range(K):
#     plt.plot(x_new, probs[:, k], marker='o', label=f"Klasse {k}")
# plt.xlabel("x (Anzahl Werbeanzeigen)")
# plt.ylabel("Vorhersagewahrscheinlichkeit")
# plt.title("One-vs-Rest: Klassenvorhersage für neue x-Werte")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Dichteplot der Posterioren
# fig, ax = plt.subplots(1, K, figsize=(12, 3))
# for k in range(K):
#     kde = gaussian_kde(beta_samples[k])
#     grid = np.linspace(-2, 2, 300)
#     ax[k].plot(grid, kde(grid))
#     ax[k].set_title(f"Posterior $\\beta_{k}$")
# plt.tight_layout()
# plt.show()