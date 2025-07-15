import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from pypolyagamma import PyPolyaGamma

#%%
# Ziel: Simulation und Bayes'sche logistische Regression mit Pólya-Gamma-Augmentation.
# Wir modellieren binäre Daten, deren Wahrscheinlichkeiten von einer latenten Photonenrate abhängen.
# Die Photonenrate wird durch ein Schachbrettmuster und raumkorreliertes Rauschen beeinflusst.

# --- Parameter ---
rows, cols = 30, 30
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
# --- Simulation der Photonenzählungen mit Multinomialverteilung ---
# Die Photonen werden entsprechend der Wahrscheinlichkeiten aus dem Schachbrettmuster verteilt.
num_photons = 10000
weights = pattern.ravel()
probabilities = weights / weights.sum()
indices = np.arange(len(probabilities))
photon_hits = np.random.choice(indices, size=num_photons, p=probabilities)
counts = np.bincount(photon_hits, minlength=len(probabilities)).reshape((rows, cols))
# --- Plot der Photonenzählungen pro Feld ---
plt.figure(figsize=(6, 5))
plt.imshow(counts, cmap="viridis")
plt.title("Photonenzählungen pro Feld")
plt.xlabel("Spalte")
plt.ylabel("Zeile")
plt.colorbar(label="Anzahl Photonen")
plt.show()

#---
# Zielvariable y wird als Binomialvariable mit Wahrscheinlichkeiten aus einer logistischen Funktion generiert.
# Die Logits basieren auf log(1 + λ), um die Beziehung zwischen Photonenrate und binären Outcomes zu modellieren.
# Die Sigmoidfunktion transformiert die Logits in Wahrscheinlichkeiten.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

logits = np.log1p(base_rate)           # log(1 + λ)
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

#%%

# --- 1-gegen-Rest Logistische Regression: Für jedes Pixel Posterior mit X_rest vs. X_i ---
logits_map = np.zeros(N)
n_iter_pixel = 10  # Iterationen pro Pixel

for i in range(N):
    # X_rest: Alle anderen Pattern-Werte als Feature-Vektor
    x_i = X[i, 1]  # Nur das Muster-Feature des aktuellen Pixels
    x_rest = np.delete(X[:, 1], i)  # Ohne das aktuelle Pixel
    y_rest = np.delete(y, i)
    
    # Neue Designmatrix: Bias + x_rest als Feature
    X_design = np.stack([
        np.ones_like(x_rest),
        x_rest
    ], axis=1)

    beta_i = np.zeros(D)
    pg = PyPolyaGamma()

    # for _ in range(n_iter_pixel):
    
    omega = np.array([pg.pgdraw(1, X_design[j] @ beta_i) for j in range(N - 1)])
    Omega = np.diag(omega)
    V_inv = X_design.T @ Omega @ X_design + B0inv
    V = inv(V_inv)
    z = (y_rest - 0.5) / omega
    m = V @ (X_design.T @ z)
    eps = 1e-6
    L = np.linalg.cholesky(V + eps * np.eye(D))
    beta_i = m + L @ np.random.randn(D)
    
    # Vorhersage: Logit für das aktuelle Pixel (x_i)
    logits_map[i] = beta_i[0] + beta_i[1] * x_i

# Plot
plt.imshow(logits_map.reshape((rows, cols)), cmap="plasma")
plt.colorbar(label="Log-Odds (1-gegen-Rest)")
plt.title("Pixelweise Log-Odds (Pixel gegen alle anderen)")
plt.show()

# %%
# --- Vergleich: 1-gegen-Rest-LogOdds mit wahrer Zielvariable y ---
'''das sieht mir noch komisch aus'''


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

probs_map = sigmoid(logits_map)
y_pred_pixelwise = (probs_map >= 0.5).astype(int)

print("Pixelweise Accuracy:", accuracy_score(y, y_pred_pixelwise))
print("AUC (1-gegen-Rest):", roc_auc_score(y, probs_map))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_pixelwise))

# Visualisierungen
plt.figure()
plt.imshow(y.reshape(rows, cols), cmap="Greys", vmin=0, vmax=1)
plt.title("Wahre Zielvariable y (binär)")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(probs_map.reshape(rows, cols), cmap="viridis")
plt.title("Vorhergesagte Wahrscheinlichkeit (1-gegen-Rest)")
plt.colorbar()
plt.show()




# ---
# Inverse Crime: Hinweis und Diskussion
#
# "Inverse Crime" bezeichnet in der Inversen Problematik (z.B. Bildrekonstruktion, inverse Modellierung)
# den Fehler, dass man synthetische Daten mit demselben Modell erzeugt, das später auch zur Rekonstruktion verwendet wird.
# Dadurch wird das Problem künstlich vereinfacht, weil das Rekonstruktionsverfahren exakt weiß, wie die Daten entstanden sind.
# Die resultierende Leistung ist dann unrealistisch hoch und nicht auf echte Daten übertragbar.
#
# In diesem Skript:
# - Die Daten (y) werden mit einer bekannten logistischen Funktion und festen Parametern erzeugt.
# - Die logistische Regression verwendet exakt dieselbe Modellstruktur (Designmatrix, Linkfunktion).
# - Es gibt keine Modellabweichung, Messrauschen oder Modellierungsfehler.
#
# Fazit: Das hier ist ein klassisches Beispiel für einen "Inverse Crime".
# Für realistische Tests sollte man:
#   - Daten mit einem komplexeren oder abweichenden Modell generieren,
#   - Messrauschen oder Modellfehler einführen,
#   - oder echte Messdaten verwenden.
# print("Achtung: Die Simulation und das Inferenzmodell sind identisch (Inverse Crime).")
