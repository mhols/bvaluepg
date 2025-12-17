import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from pypolyagamma import PyPolyaGamma

#%%
# Ziel: Simulation und Bayes'sche logistische Regression mit Pólya-Gamma-Augmentation.
# Wir modellieren binäre Daten, deren Wahrscheinlichkeiten von einer latenten Photonenrate abhängen.
# Die Photonenrate wird durch ein Schachbrettmuster und raumkorreliertes Rauschen beeinflusst.

# --- Parameter ---
rows, cols = 8, 8
poisson_lambda = 20 # Erwartungswert pro Feld
noise_std = 2
np.random.seed(42)

#---
# Schachbrettmuster als Prior für die latente Struktur.
# Dieses Muster erzeugt zwei unterschiedliche Wahrscheinlichkeiten (0.3 und 1.0),
# um eine klare, strukturierte Variation in der Photonenrate zu modellieren.
# Die Wahl des Schachbrettmusters dient dazu, einen einfachen, aber aussagekräftigen Effekt darzustellen,
# der später durch die logistische Regression erkannt werden soll.

x = np.indices((rows, cols)).sum(axis=0) % 2
pattern = np.where(x == 0, 0.3, 1.0)

#---
# Die Photonenrate pro Feld basiert auf dem Schachbrettmuster.
# λ wird als Basisrate plus Muster skaliert.
# Dies modelliert die Intensität der Photonen, die später als Input für die Binomialverteilung dient.

base_rate = poisson_lambda * (1 + pattern)

#---
# Raumkorreliertes Rauschen wird erzeugt, um realistischere Daten zu simulieren.
# Die Korrelation zwischen Feldern wird durch eine Gaußsche Korrelationsfunktion definiert,
# die Abhängigkeiten zwischen benachbarten Feldern modelliert.
# Die Cholesky-Zerlegung der Kovarianzmatrix ermöglicht das Ziehen korrelierter Zufallsvariablen.

def corr_function(x1, y1, x2, y2, sigma=0.1):
    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
    return np.exp(-dist_sq / (2 * sigma**2))

x_vals = np.linspace(0, 1, rows)
y_vals = np.linspace(0, 1, cols)
Xgrid, Ygrid = np.meshgrid(x_vals, y_vals)
rX, rY = Xgrid.ravel(), Ygrid.ravel()

Cova_matrix = np.array([[corr_function(rX[i], rY[i], rX[j], rY[j])
                         for j in range(len(rX))] for i in range(len(rX))])

# Cholesky-Zerlegung und korreliertes Rauschen
L = np.linalg.cholesky(Cova_matrix + 1e-10 * np.eye(len(rX)))
correlated_noise = (L @ np.random.randn(len(rX))).reshape((rows, cols))

#%%
# --- Simulation der Photonenzählungen mit Raumkorrelation ---
# Die Poisson-Parameter werden durch die Basisrate plus korreliertes Rauschen bestimmt.
# Durch Clipping stellen wir sicher, dass die Raten positiv bleiben.
poisson_input = base_rate + correlated_noise
photon_counts_corr = np.random.poisson(lam=np.clip(poisson_input, 0.1, None))

#---
# Zielvariable y wird als Binomialvariable mit Wahrscheinlichkeiten aus einer logistischen Funktion generiert.
# Die Logits basieren auf log(1 + λ), um die Beziehung zwischen Photonenrate und binären Outcomes zu modellieren.
# Die Sigmoidfunktion transformiert die Logits in Wahrscheinlichkeiten.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

logits = np.log1p(poisson_input)           # log(1 + λ)
probs = sigmoid(logits)                    # p = sigmoid(log(1 + λ))
y = np.random.binomial(n=1, p=probs.ravel())

#---
# Designmatrix X mit Bias-Term und Muster als Prädiktor.
# Diese Matrix wird für die logistische Regression verwendet.
X = np.stack([
    np.ones_like(pattern.ravel()),         # Bias
    pattern.ravel()                        # Muster: 0.3 oder 1.0
], axis=1)

#%%
# --- Bayes'sche logistische Regression mit Pólya-Gamma-Augmentation ---
# Ziel: Posterior-Verteilung der Regressionsparameter β schätzen.
# Pólya-Gamma-Variablen ω werden eingeführt, um die nicht-konjugierte logistische Likelihood in eine
# bedingt normale Form umzuwandeln, was Gibbs-Sampling ermöglicht.

N, D = X.shape
beta = np.zeros(D)
B0inv = np.eye(D) * 1.0                    # Prior Precision (Inverse Kovarianz) für β
pg = PyPolyaGamma()
samples = []

n_iter = 500
for i in range(n_iter):
    # 1. Sampling von ω_i ~ PG(1, xᵢᵗ β)
    # ω sind Pólya-Gamma-verteilte Hilfsvariablen, die die Likelihood vereinfachen.
    omega = np.array([pg.pgdraw(1, X[j] @ beta) for j in range(N)])
    Omega = np.diag(omega)

    # 2. Sampling von β ~ N(m, V)
    # Die bedingte Verteilung von β ist multivariate Normalverteilung mit:
    # V = (Xᵀ Ω X + B⁻¹)⁻¹ als Kovarianzmatrix,
    # m = V Xᵀ ((y - 0.5) / ω) als Mittelwert.
    # Dies folgt aus der Pólya-Gamma-Augmentation, die die logistische Regression in eine
    # bedingt normale Form transformiert.
    V_inv = X.T @ Omega @ X + B0inv
    V = inv(V_inv)
    z = (y - 0.5) / omega
    m = V @ (X.T @ z)

    # Cholesky-Zerlegung wird verwendet, um aus der Normalverteilung effizient zu sampeln.
    # Sie ermöglicht das Ziehen von korrelierten Normalvariablen durch Transformation von Standardnormalen.
    eps = 1e-6
    L = np.linalg.cholesky(V + eps * np.eye(D))
    beta = m + L @ np.random.randn(D)
    samples.append(beta)

samples = np.array(samples)

#%%
# --- Visualisierung der Posteriorverteilungen der Regressionsparameter ---
# Die Histogramme zeigen die Unsicherheit und Verteilung der geschätzten Parameter β₀ (Bias) und β₁ (Muster-Effekt).
# Ein schmaler, gut definierter Peak deutet auf hohe Sicherheit hin,
# während breite Verteilungen auf Unsicherheit oder schwache Effekte hindeuten.
# ob das so passt :?

plt.hist(samples[:, 0], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Posteriorverteilung für β₀ (Bias)")
plt.xlabel("β₀")
plt.ylabel("Häufigkeit")
plt.grid(True)
plt.show()

plt.hist(samples[:, 1], bins=30, color='salmon', edgecolor='black', alpha=0.7)
plt.title("Posteriorverteilung für β₁ (Muster-Effekt)")
plt.xlabel("β₁")
plt.ylabel("Häufigkeit")
plt.grid(True)
plt.show()


# next step
# vergleich mit dem Muster