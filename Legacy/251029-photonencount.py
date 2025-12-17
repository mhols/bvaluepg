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
    for j in range(cols):
        ergebnis[i,j]=np.random.poisson(feld[i,j],size=1)
# plot ergebnis
plt.figure(figsize=(6, 5))
plt.imshow(ergebnis, cmap="viridis")
plt.title("Anzahl detektierter Photonen pro Feld")
plt.xlabel("Spalte")
plt.ylabel("Zeile")










#%% Posterior-Verteilung für eine Poisson-Rate λ (Gamma-Prior → Gamma-Posterior)
import math

# Eine exemplarische Zelle auswählen (z. B. mittig)
i_sel, j_sel = 15, 15
k = int(ergebnis[i_sel, j_sel])  # beobachtete Photonen in dieser Zelle

# Gamma-Prior (schwach informativ, Mittelwert ≈100)
alpha0 = 2.0
beta0 = 0.02  # Rate-Parameterisierung (nicht Skala): E[λ] = α/β = 100

# Posterior-Parameter (Poisson-Gamma-Konjunktion)
alpha_post = alpha0 + k
beta_post = beta0 + 1.0

def gamma_pdf_rate_param(lam, alpha, beta):
    """Dichte der Gamma-Verteilung in Rate-Parameterisierung."""
    if lam <= 0:
        return 0.0
    return (beta**alpha) / math.gamma(alpha) * (lam**(alpha - 1)) * math.exp(-beta * lam)

lam_max = max(300, int(feld.max() + 4*np.sqrt(feld.max())))
lam_grid = np.linspace(0.01, lam_max, 1500)

prior_vals = np.array([gamma_pdf_rate_param(l, alpha0, beta0) for l in lam_grid])
post_vals  = np.array([gamma_pdf_rate_param(l, alpha_post, beta_post) for l in lam_grid])

# Plot Prior vs. Posterior
plt.figure(figsize=(8, 5))
plt.plot(lam_grid, prior_vals, label=f"Prior Γ(α0={alpha0:.1f}, β0={beta0:.2f})")
plt.plot(lam_grid, post_vals, label=f"Posterior Γ(α={alpha_post:.0f}, β={beta_post:.2f})")
plt.title(f"Posterior der Poisson-Rate λ für Zelle ({i_sel},{j_sel}) mit Beobachtung k={k}")
plt.xlabel("λ")
plt.ylabel("Dichte")
plt.legend()
plt.tight_layout()
plt.show()

# Kenngrößen ausgeben
mean_prior = alpha0 / beta0
var_prior = alpha0 / (beta0**2)
mean_post = alpha_post / beta_post
var_post = alpha_post / (beta_post**2)

print(f"Beobachtung k = {k}")
print(f"E[λ_prior] = {mean_prior:.2f}, Var[λ_prior] = {var_prior:.2f}")
print(f"E[λ_post]  = {mean_post:.2f}, Var[λ_post]  = {var_post:.2f}")
