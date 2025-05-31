#%%
"""
Bayessche logistische Regression – Übungsaufgaben

Author: Toni Luhdo
Created: 2025-05-20

Dieses Skript enthält die Lösungen zu den Übungsaufgaben aus der Datei 'uebungsaufgaben.md'.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid-Funktion

# Beobachtungen
# x: Werbekontakte
# zweimal drei :) erinner dich kann alles sein
x = np.array([0, 1, 2, 3, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

''''
Ergaenzung:
Wenn ich die Daten nicht selbst generiere, sondern simuliere, dann
kann ich die Daten wie folgt generieren:
beta_true = 1.5
x = np.random.randint(0, 9, size=50)
p = 1 / (1 + np.exp(-x * beta_true))
y = np.random.binomial(1, p)
'''


n = len(x)
#%%
# Aufgabe 0 – Mittelwert und Varianz
mean_x = np.mean(x)
var_x = np.var(x, ddof=1)  # Stichprobenvarianz

print(f"Mittelwert x: {mean_x:.2f}")
print(f"Stichprobenvarianz x: {var_x:.2f}")

#%%
# Aufgabe 1 & 2 – Likelihood und log-Likelihood

# Eigene Sigmoidfunktion
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Likelihood-Funktion
def likelihood(beta):
    p = sigmoid(x * beta)
    return np.prod(p ** y * (1 - p) ** (1 - y))

def log_likelihood(beta):
    p = sigmoid(x * beta)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
#%%

# Aufgabe 5 – Visualisierung der Beobachtungen
plt.figure()
plt.scatter(x, y, color='black')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Beobachtungen: Werbekontakte vs. Reaktion")
plt.grid(True)
plt.tight_layout()
plt.show()

# Aufgabe 5 – Sigmoid-Fit über Beobachtungen

beta_vals = np.linspace(-10, 10, 500)
likelihood_vals = np.array([likelihood(b) for b in beta_vals])
beta_hat = beta_vals[np.argmax(likelihood_vals)]  # einfachster Maximum-Likelihood-Schätzer

x_grid = np.linspace(min(x) - 1, max(x) + 1, 200)
p_pred = sigmoid(x_grid * beta_hat)

plt.figure()
plt.scatter(x, y, color='black', label='Beobachtungen')
plt.plot(x_grid, p_pred, color='blue', label=f'Sigmoid (β ≈ {beta_hat:.2f})')
plt.xlabel("x")
plt.ylabel("P(y=1 | x)")
plt.title("Sigmoid-Fit zur Modellierung von P(y=1 | x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Aufgabe 5 – Likelihood-Plot


plt.plot(beta_vals, likelihood_vals)
plt.xlabel("β")
plt.ylabel("Likelihood")
plt.title("Likelihood-Verlauf")
plt.grid(True)
plt.tight_layout()
plt.show()

# Aufgabe 5 – Posterior (bis auf Normalisierung) plotten
log_lik_vals = np.array([log_likelihood(b) for b in beta_vals])
log_prior = np.where((beta_vals >= -10) & (beta_vals <= 10), 0, -np.inf)
log_post = log_lik_vals + log_prior

# Aufgabe 5 – Posterior (unnormalisiert) plotten

#posterior = np.exp(log_post - np.max(log_post))  # für numerische Stabilität

prior_vals = np.where((beta_vals >= -10) & (beta_vals <= 10), 1.0, 0.0)  # Uniform-Prior
# BEDENKE: Integral muesste 1 sein, also normieren
# hier egal da unnormierte Posterior-Verteilung

# Visualisierung des Prior-Verlaufs
plt.plot(beta_vals, prior_vals)
plt.xlabel("β")
plt.ylabel("Prior-Dichte")
plt.title("Uniformer Prior über [-10, 10]")
plt.grid(True)
plt.tight_layout()
plt.show()

posterior = likelihood_vals * prior_vals  # Posterior ∝ Likelihood * Prior

plt.plot(beta_vals, posterior)
plt.xlabel("β")
plt.ylabel("Posterior (unnormalisiert)")
plt.title("Posterior-Verlauf (unnormalisiert)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.plot(beta_vals, log_post)
plt.xlabel("β")
plt.ylabel("log Posterior (unnormalisiert)")
plt.title("Posterior-Verlauf (bis auf Normierung)")
plt.grid(True)
plt.tight_layout()
plt.show()
#%%

# Aufgabe 6 – Gibbs-Sampling mit Polya-Gamma
# ich muss die Vollstaendige Implementation Polya-Gamma-Sampling mit externem Paket machen

# BE CAREFUL: es gibt verschiedene Implementationen von Pólya-Gamma-Sampling
# polyagamma scheint nicht mehr gepflegt zu sein
# pypolyagamma laeuft mit aktieller numpy und scipy version

from pypolyagamma import PyPolyaGamma


def gibbs_sampler_pg(x, y, tau2=10, n_samples=1000, burn_in=200):
    """
    Führt Gibbs-Sampling zur Schätzung des Posterior-Verlaufs von β
    mit Hilfe der Pólya-Gamma-Datenaugmentation durch.

    Parameter:
    - x: Array der Feature-Werte (n,)
    - y: Array der Zielwerte (0 oder 1) (n,)
    - tau2: Prior-Varianz von β (Standard: 10)
    - n_samples: Anzahl der Posterior-Samples nach Burn-in
    - burn_in: Anzahl an Samples zum Verwerfen (Burn-in-Phase)

    Rückgabe:
    - Array der β-Samples aus dem Posterior

    Teste mal verschiedene tau2-Werte
    """
    n = len(x)
    X = x.reshape(-1, 1)  # Designmatrix (n x 1)
    pg = PyPolyaGamma()
    
    beta_samples = []
    beta = 1.0  # Initialwert

    for t in range(n_samples + burn_in):
        # Schritt 1: Ziehe ω_i ∼ PG(1, x_i * β)
        omega = np.array([pg.pgdraw(1, xi * beta) for xi in x])
        Omega = np.diag(omega)
        
        # Schritt 2: Ziehe β ∼ N(m, V)
        XT_Omega_X = X.T @ Omega @ X
        V = 1 / (XT_Omega_X + 1 / tau2)
        m = V * (X.T @ (y - 0.5))
        beta = np.random.normal(loc=m.item(), scale=np.sqrt(V.item()))
        
        if t >= burn_in:
            beta_samples.append(beta)
    
    return np.array(beta_samples)

# Aufruf und Plot
beta_samples = gibbs_sampler_pg(x, y)
plt.hist(beta_samples, bins=30, density=True)
plt.xlabel("β")
plt.ylabel("Dichte")
plt.title("Posterior von β (Gibbs-Sampler mit Pólya-Gamma)")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%





#################################################################
''' Wie schaut's mit Intercept aus?

Erweiterte Modellierung mit Intercept
Modell: P(y=1|x) = sigmoid(β₀ + β₁ x)
'''

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def likelihood_intercept(beta0, beta1):
    p = sigmoid(beta0 + beta1 * x)
    return np.prod(p ** y * (1 - p) ** (1 - y))

def log_likelihood_intercept(beta0, beta1):
    p = sigmoid(beta0 + beta1 * x)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

#%%

# Likelihood-Gitter über β₀ und β₁
beta0_vals = np.linspace(-10, 10, 100)
beta1_vals = np.linspace(-4, 5, 100)
B0, B1 = np.meshgrid(beta0_vals, beta1_vals)
log_lik_grid = np.array([[log_likelihood_intercept(b0, b1) for b0 in beta0_vals] for b1 in beta1_vals])
# Vorsicht mit verschachtelten Listen-Komprehensionen, kann man sich gut mit austicksen

plt.contourf(B0, B1, log_lik_grid, levels=30, cmap="viridis")
plt.xlabel("β₀ (Intercept)")
plt.ylabel("β₁ (Steigung)")
plt.title("Log-Likelihood für β₀ und β₁")
plt.colorbar(label="Log-Likelihood")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Querschnitt für festes beta0 (z.B. beta0 = 0)
beta0_fixed = 0
idx_beta0 = np.abs(beta0_vals - beta0_fixed).argmin()
log_lik_beta1 = log_lik_grid[:, idx_beta0]

plt.plot(beta1_vals, log_lik_beta1)
plt.xlabel("β₁ (Steigung)")
plt.ylabel("Log-Likelihood")
plt.title(f"Log-Likelihood für festes β₀ = {beta0_fixed}")
plt.grid(True)
plt.tight_layout()
plt.show()

# Querschnitt für festes beta1 (z.B. beta1 = 1)
beta1_fixed = 1
idx_beta1 = np.abs(beta1_vals - beta1_fixed).argmin()
log_lik_beta0 = log_lik_grid[idx_beta1, :]

plt.plot(beta0_vals, log_lik_beta0)
plt.xlabel("β₀ (Intercept)")
plt.ylabel("Log-Likelihood")
plt.title(f"Log-Likelihood für festes β₁ = {beta1_fixed}")
plt.grid(True)
plt.tight_layout()
plt.show()
#%%

# nur fuer mich 
# Beispiel-Sigmoidkurve mit β₀ = -1, β₁ = 0.8
x_plot = np.linspace(0, 10, 100)
beta0 = -1
beta1 = 0.8
p_curve = sigmoid(beta0 + beta1 * x_plot)

plt.plot(x_plot, p_curve)
plt.xlabel("x")
plt.ylabel("P(y=1 | x)")
plt.title("Sigmoid mit Intercept β₀ = -1, β₁ = 0.8")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# Gibbs-Sampler mit Intercept
def gibbs_sampler_pg_intercept(x, y, tau2=10, n_samples=1000, burn_in=200):
    '''auch hier mal verschidene tau2-Werte testen
    Beschreiung kommt noch werd muede... :(
    '''    
    n = len(x)
    X = np.column_stack((np.ones_like(x), x))  # Intercept-Spalte
    pg = PyPolyaGamma()


    beta_samples = []
    beta = np.array([0.0, 1.0])  # Startwerte: [β₀, β₁]

    for t in range(n_samples + burn_in):
        omega = np.array([pg.pgdraw(1, X[i] @ beta) for i in range(n)])
        Omega = np.diag(omega)

        XT_Omega_X = X.T @ Omega @ X
        V = np.linalg.inv(XT_Omega_X + (1 / tau2) * np.eye(2))
        m = V @ (X.T @ (y - 0.5))
        beta = np.random.multivariate_normal(mean=m, cov=V)

        if t >= burn_in:
            beta_samples.append(beta)

    return np.array(beta_samples)

#%%

# Ausführen
beta_samples_intercept = gibbs_sampler_pg_intercept(x, y)

# Plotten der Posterior-Samples
plt.hist(beta_samples_intercept[:, 0], bins=30, density=True, alpha=0.7, label="β₀ (Intercept)")
plt.hist(beta_samples_intercept[:, 1], bins=30, density=True, alpha=0.7, label="β₁ (Steigung)")
plt.xlabel("β")
plt.ylabel("Dichte")
plt.title("Posteriorverteilungen von β₀ und β₁")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%


#%%
'''
nur intercept
Modell: P(y=1) = sigmoid(β₀)

'''

# Modell nur mit Intercept: P(y=1) = sigmoid(β₀)

# Likelihood nur mit Intercept
def likelihood_only_intercept(beta0):
    p = sigmoid(beta0)
    return np.prod(p**y * (1 - p)**(1 - y))

def log_likelihood_only_intercept(beta0):
    p = sigmoid(beta0)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


# Likelihood und log-Likelihood über β₀
beta0_vals = np.linspace(-5, 5, 500)
log_lik_vals_only_intercept = np.array([log_likelihood_only_intercept(b0) for b0 in beta0_vals])

plt.plot(beta0_vals, log_lik_vals_only_intercept)
plt.xlabel("β₀")
plt.ylabel("log-Likelihood")
plt.title("Log-Likelihood für Modell mit nur Intercept")
plt.grid(True)
plt.tight_layout()
plt.show()

# Likelihood (nicht log) über β₀
lik_vals_only_intercept = np.array([likelihood_only_intercept(b0) for b0 in beta0_vals])

plt.plot(beta0_vals, lik_vals_only_intercept)
plt.xlabel("β₀")
plt.ylabel("Likelihood")
plt.title("Likelihood für Modell mit nur Intercept")
plt.grid(True)
plt.tight_layout()
plt.show()

# Maximum-Likelihood-Schätzer für β₀
p_hat = np.mean(y)
beta0_hat = np.log(p_hat / (1 - p_hat))
print(f"MLE-Schätzer für β₀ (nur Intercept): {beta0_hat:.3f}")

# Plot Sigmoidkurve für geschätzten β₀
x_dummy = np.linspace(0, 10, 100)
p_const = sigmoid(beta0_hat) * np.ones_like(x_dummy)

plt.plot(x_dummy, p_const, label=f"P(y=1) = {p_const[0]:.2f}")
plt.scatter(x, y, color='black', alpha=0.6, label="Daten")
plt.xlabel("x (ignoriert)")
plt.ylabel("P(y=1)")
plt.title("Modell mit nur Intercept – konstante Wahrscheinlichkeit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Gibbs-Sampler für Modell mit nur Intercept
def gibbs_sampler_pg_only_intercept(y, tau2=10, n_samples=1000, burn_in=200):
    """
    Gibbs-Sampler zur Schätzung von β₀ (Intercept) mit Pólya-Gamma-Data-Augmentation.
    Denk an die verschiedenen tau2-Werte!
    """
    n = len(y)
    pg = PyPolyaGamma()
    
    beta0_samples = []
    beta0 = 0.0  # Startwert
    
    for t in range(n_samples + burn_in):
        omega = np.array([pg.pgdraw(1, beta0) for _ in range(n)])
        V = 1 / (np.sum(omega) + 1 / tau2)
        m = V * np.sum(y - 0.5)
        beta0 = np.random.normal(loc=m, scale=np.sqrt(V))
        
        if t >= burn_in:
            beta0_samples.append(beta0)
    
    return np.array(beta0_samples)
#%%

# Ausfuehrung und Visualisierung des Gibbs-Samplers fuer das Intercept-only Modell
beta0_samples = gibbs_sampler_pg_only_intercept(y)

plt.hist(beta0_samples, bins=30, density=True, alpha=0.8, label="β₀ Posterior")
plt.axvline(beta0_hat, color='red', linestyle='--', label='MLE β₀')
plt.xlabel("β₀")
plt.ylabel("Dichte")
plt.title("Posteriorverteilung von β₀ (Intercept-only Modell)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

