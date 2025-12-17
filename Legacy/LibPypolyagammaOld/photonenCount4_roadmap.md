---
**Title:** Photonenzählung & Bayes‑Logit mit Pólya‑Gamma – Leitfaden  
**Author:** Toni Luhdo (prepared with GPT‑5 Thinking)  
**Created:** 2025-10-01  
**Version:** 1.0
---


# Ziel & Überblick

Dieses Dokument erklärt das Vorgehen im Skript `photonenCount4.py` Schritt für Schritt – so, dass andere es **nachbauen** können.  
Wir simulieren (1) Photonenkarten auf einem 30×30‑Gitter, (2) generieren daraus binäre Ziele via logistischer Funktion und (3) schätzen eine **bayessche logistische Regression** mit **Pólya‑Gamma‑Augmentation** per Gibbs‑Sampling.  
Zum Schluss zeigen wir eine (experimentelle) *1‑gegen‑Rest*‑Log‑Odds‑Karte und diskutieren den **Inverse‑Crime**‑Aspekt.

---

# Voraussetzungen

## Python‑Pakete

```bash
pip install numpy matplotlib pypolyagamma scikit-learn
```

> Getestet mit: Python ≥3.9, `numpy`, `matplotlib`, `pypolyagamma`, `scikit-learn`.

## Reproduzierbarkeit

Im Skript wird ein fester Seed gesetzt:
```python
np.random.seed(42)
```
Damit werden die Ergebnisse (bis auf numerische Rundung) reproduzierbar.

---

# Parameter & Intuition

- Gittergröße: `rows=30`, `cols=30`  → **N = 900** Pixel.
- Basisrate für Photonen: `poisson_lambda = 20`.
- **Muster**: Schachbrett (`pattern ∈ {0.3, 1.0}`) erzeugt zwei Intensitätsniveaus.
- Photonenanzahl gesamt: `num_photons = 10000`.

Ziel: Ein einfaches, aber strukturreiches Szenario, in dem das **Muster** einen messbaren Effekt hat.

---

# 1) Schachbrett‑Muster und (optionales) raumkorreliertes Rauschen

## Muster

```python
x = np.indices((rows, cols)).sum(axis=0) % 2
pattern = np.where(x == 0, 0.3, 1.0)  # zwei Niveaus
```

Das **Schachbrett** erzeugt alternierende Blöcke mit Werten 0.3 und 1.0 – als latente Struktur.

## Raumkorreliertes Rauschen (erstellt, aber im weiteren Verlauf nicht genutzt)

Zur Demonstration wird eine Kovarianzmatrix via Gauß‑Korrelation \(k(d)=\exp(-d^2/(2\sigma^2))\) aufgebaut
und mit Cholesky‑Zerlegung korreliertes Rauschen gezogen:

```python
def corr_function(x1, y1, x2, y2, sigma=0.1):
    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
    return np.exp(-dist_sq / (2 * sigma**2))

# ... Cova_matrix -> L -> correlated_noise
```

> **Hinweis:** In der aktuellen Fassung wird `correlated_noise` **nicht** zur Daten­generierung genutzt.
Man könnte es z. B. additiv in eine latente Log‑Rate aufnehmen, um realistischere Strukturen zu erzeugen.

---

# 2) Photonenzählungen über Multinomial

Wir verteilen `num_photons` über das Gitter proportional zu den **Muster‑Gewichten**:

\[
p_i \;=\; \frac{w_i}{\sum_j w_j},\qquad w_i \in \{0.3, 1.0\}.
\]

```python
num_photons = 10000
weights = pattern.ravel()
probabilities = weights / weights.sum()
indices = np.arange(len(probabilities))
photon_hits = np.random.choice(indices, size=num_photons, p=probabilities)
counts = np.bincount(photon_hits, minlength=len(probabilities)).reshape((rows, cols))
```

`counts[i,j]` ist die Zahl der Photonen, die im Feld \((i,j)\) gelandet sind.  
Anschließend wird das Zählbild visualisiert (`imshow`).

---

# 3) Binäre Zielvariable über logistische Transformation

Wir definieren eine **Photonenrate** als Basisrate skaliert mit dem Muster:
\[
\lambda_{ij} \;=\; \texttt{poisson\_lambda}\,\bigl(1 + \texttt{pattern}_{ij}\bigr).
\]

Daraus bilden wir Logits und Wahrscheinlichkeiten:
\[
\eta_{ij} \;=\; \log(1+\lambda_{ij}),\qquad
p_{ij} \;=\; \sigma(\eta_{ij}) \;=\; \frac{1}{1+\exp(-\eta_{ij})}.
\]

```python
def sigmoid(z): return 1 / (1 + np.exp(-z))

base_rate = poisson_lambda * (1 + pattern)
logits = np.log1p(base_rate)         # η = log(1+λ)
probs  = sigmoid(logits)             # p = σ(η)
y      = np.random.binomial(n=1, p=probs.ravel())  # y ∈ {0,1}
```

> **Interpretation:** Je höher die (latente) Intensität \( \lambda \), desto höher \( \eta \) und desto höher die Erfolgs­wahrscheinlichkeit \(p\).

---

# 4) Designmatrix

Wir schätzen eine logistische Regression **nur** mit Bias und Muster:

\[
\Pr(y_i=1\mid x_i,\beta) = \sigma(x_i^\top\beta),\quad
x_i = (1,\; \texttt{pattern}_i)^\top,\; \beta\in\mathbb{R}^2.
\]

```python
X = np.stack([np.ones_like(pattern.ravel()), pattern.ravel()], axis=1)  # N×2
N, D = X.shape  # N=900, D=2
```

---

# 5) Bayessche Logit per Pólya‑Gamma‑Augmentation (Gibbs)

## 5.1 Logistische Likelihood

Für unabhängige Beobachtungen \((y_i,x_i)\):
\[
p(\mathbf{y}\mid\beta) \;=\; \prod_{i=1}^N \sigma(\psi_i)^{y_i}\,\bigl(1-\sigma(\psi_i)\bigr)^{1-y_i},
\quad \psi_i=x_i^\top\beta.
\]

Mit dem Trick von **Pólya & Gamma (2013)** lässt sich der logistische Term mittels einer
**Pólya‑Gamma‑Verteilung** \( \omega_i\sim \mathrm{PG}(1,\psi_i) \) so darstellen, dass
die bedingte Posterior von \(\beta\) **normal** wird.

## 5.2 Augmentiertes Modell

Die Identität
\[
\frac{(e^{\psi_i})^{y_i}}{(1+e^{\psi_i})}
\;=\;
\frac{1}{2} e^{\kappa_i \psi_i}\int_0^\infty 
e^{-\frac{\omega_i \psi_i^2}{2}}\, p(\omega_i)\,d\omega_i,
\quad \kappa_i=y_i-\tfrac{1}{2},\;\omega_i\sim\mathrm{PG}(1,0),
\]
impliziert nach **Conditionierung auf \(\omega_i\)** eine **Gaussian‑Form** in \(\psi_i\).

Mit Normal‑Prior
\[
\beta \sim \mathcal{N}(0,\,B_0),\quad B_0^{-1}= \texttt{B0inv},
\]
ergibt sich die bedingte Posterior:
\[
\beta \mid \mathbf{y},\boldsymbol{\omega}
\;\sim\;
\mathcal{N}\bigl(m,\;V\bigr),
\quad
V = \bigl(X^\top \Omega X + B_0^{-1}\bigr)^{-1},\quad
m = V\,X^\top \mathbf{z},
\]
mit
\[
\Omega=\mathrm{diag}(\omega_1,\dots,\omega_N),\qquad
\mathbf{z} = \frac{\mathbf{y}-\tfrac12}{\boldsymbol{\omega}}\quad (\text{elementweise Division}).
\]

## 5.3 Gibbs‑Sampler

Für \(t=1,\dots,T\):

1. **Augmentieren:**  
   Ziehe \(\omega_i^{(t)} \sim \mathrm{PG}(1, x_i^\top \beta^{(t-1)})\) für alle \(i\).
2. **Koefﬁzienten:**  
   Setze \(\Omega^{(t)}=\mathrm{diag}(\omega^{(t)})\), berechne
\[
   V^{(t)} = \bigl(X^\top \Omega^{(t)} X + B_0^{-1}\bigr)^{-1},\qquad
   m^{(t)} = V^{(t)} X^\top \mathbf{z}^{(t)},\quad
   \mathbf{z}^{(t)} = \frac{\mathbf{y}-\tfrac12}{\boldsymbol{\omega}^{(t)}},
\]
   und ziehe \(\beta^{(t)} \sim \mathcal{N}(m^{(t)}, V^{(t)})\).

**Implementierungsausschnitt:**
```python
B0inv = np.eye(D) * 1.0
pg = PyPolyaGamma()
beta = np.zeros(D)
samples = []

for _ in range(500):
    omega = np.array([pg.pgdraw(1, X[j] @ beta) for j in range(N)])
    Omega = np.diag(omega)
    V_inv = X.T @ Omega @ X + B0inv
    V = np.linalg.inv(V_inv)
    z = (y - 0.5) / omega
    m = V @ (X.T @ z)

    # Ziehung via Cholesky
    L = np.linalg.cholesky(V + 1e-6*np.eye(D))
    beta = m + L @ np.random.randn(D)
    samples.append(beta)
```

> **Praxistipp:** Nach dem Sampling Burn‑in und Thinning in Betracht ziehen; Konvergenz z. B. mit Trace‑Plots prüfen.

---

# 6) Posterior‑Analyse

Histogramme der Posterior‑Samples für \(\beta_0\) (Bias) und \(\beta_1\) (Muster‑Effekt) zeigen
Lage & Unsicherheit. Ein **schmaler Peak** deutet auf präzise Schätzung hin.

```python
plt.hist(samples[:, 0], bins=30)
plt.title("Posterior für β₀ (Bias)")
plt.show()

plt.hist(samples[:, 1], bins=30)
plt.title("Posterior für β₁ (Muster)")
plt.show()
```

---

# 7) *1‑gegen‑Rest*‑Log‑Odds (experimentell)

Für jedes Pixel \(i\) wird (einmalig) eine Logit‑Regression auf **alle anderen Pixel** (ohne \(i\)) geschätzt:
- Feature ist **nur** das Muster der übrigen Pixel.
- Anschließend wird der Logit für \(x_i\) mit den geschätzten \(\beta_i\) berechnet.

```python
logits_map = np.zeros(N)
for i in range(N):
    x_i = X[i, 1]
    x_rest = np.delete(X[:, 1], i)
    y_rest = np.delete(y, i)
    X_design = np.stack([np.ones_like(x_rest), x_rest], axis=1)

    # ein Schritt PG‑Gibbs:
    beta_i = np.zeros(D)
    omega = np.array([pg.pgdraw(1, X_design[j] @ beta_i) for j in range(N - 1)])
    Omega = np.diag(omega)
    V_inv = X_design.T @ Omega @ X_design + B0inv
    V = np.linalg.inv(V_inv)
    z = (y_rest - 0.5) / omega
    m = V @ (X_design.T @ z)
    L = np.linalg.cholesky(V + 1e-6*np.eye(D))
    beta_i = m + L @ np.random.randn(D)

    logits_map[i] = beta_i[0] + beta_i[1] * x_i
```

**Wichtig:** Das ist **kein** klassisches „Pixel‑spezifisches Posterior“, sondern eine Heuristik.
Eine robuste, räumlich explizite Modellierung würde z. B. ein **räumliches GLM/GP** oder **CAR/ICAR‑Priors** nutzen.

Zur Bewertung:
```python
probs_map = sigmoid(logits_map)
y_pred = (probs_map >= 0.5).astype(int)

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
print("Accuracy:", accuracy_score(y, y_pred))
print("AUC:", roc_auc_score(y, probs_map))
print("Confusion:\n", confusion_matrix(y, y_pred))
```

---

# 8) Inverse Crime – Warnhinweis

**Inverse Crime**: Man generiert Daten mit **demselben** Modell, das man zur Inferenz verwendet.
Hier:  
- \(y\) wurde mit \(p=\sigma(\log(1+\lambda))\) und Feature **pattern** erzeugt.  
- Die Regression nutzt **genau dieses** Feature & Link.  

**Folge:** Unterschätzte Unsicherheit & zu optimistische Leistung.  
**Gegenmittel:**  
- Abweichende Daten‑Generierung (z. B. zusätzliche nichtlineare Effekte, anderes Link, Rauschen).  
- Räumliche Korrelationen **wirklich** nutzen (siehe `correlated_noise`).  
- Mit echten Daten testen.

---

# 9) Komplettes Minimalbeispiel

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from pypolyagamma import PyPolyaGamma
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

np.random.seed(42)
rows, cols = 30, 30
poisson_lambda = 20

# Muster
x = np.indices((rows, cols)).sum(axis=0) % 2
pattern = np.where(x == 0, 0.3, 1.0)

# Photonenzählungen (Multinomial)
num_photons = 10000
weights = pattern.ravel()
probabilities = weights / weights.sum()
idx = np.random.choice(len(probabilities), size=num_photons, p=probabilities)
counts = np.bincount(idx, minlength=len(probabilities)).reshape((rows, cols))

plt.imshow(counts, cmap="viridis"); plt.title("Photonenzählungen"); plt.colorbar(); plt.show()

# Binärziel aus log(1+λ) via Sigmoid
def sigmoid(z): return 1/(1+np.exp(-z))
base_rate = poisson_lambda * (1 + pattern)
logits = np.log1p(base_rate)
probs  = sigmoid(logits)
y      = np.random.binomial(1, probs.ravel())

# Designmatrix
X = np.stack([np.ones_like(pattern.ravel()), pattern.ravel()], axis=1)
N, D = X.shape

# PG-Gibbs
B0inv = np.eye(D) * 1.0
pg = PyPolyaGamma()
beta = np.zeros(D)
samples = []
for _ in range(500):
    omega = np.array([pg.pgdraw(1, X[j] @ beta) for j in range(N)])
    Omega = np.diag(omega)
    V_inv = X.T @ Omega @ X + B0inv
    V = inv(V_inv)
    z = (y - 0.5) / omega
    m = V @ (X.T @ z)
    L = np.linalg.cholesky(V + 1e-6*np.eye(D))
    beta = m + L @ np.random.randn(D)
    samples.append(beta)

samples = np.array(samples)
plt.hist(samples[:,1], bins=30); plt.title("Posterior β1"); plt.show()
```

---

# 10) Erweiterungen & Optionen

- **Räumliche Priors:** CAR/ICAR oder GP auf \(\eta\) oder \(\beta\).  
- **Mehr Features:** Z. B. Glättungen, Interaktionen.  
- **Alternative Links:** Probit mit Albert‑Chib‑Augmentation.  
- **Konvergenzdiagnostik:** Trace‑Plots, R‑hat, ESS.

---

# 11) Häufige Stolpersteine

- `pypolyagamma` benötigt C‑Erweiterungen; beim Installieren auf Compiler‑Fehler achten.  
- Numerische Stabilität: kleine **Jitter**‑Terme (z. B. `1e-6*I`) bei Cholesky.  
- Nicht vergessen: Burn‑in und ausreichend Iterationen; 500 ist oft nur ein Startwert.

---

# 12) Lizenz & Zitat

Wenn du den PG‑Trick zitieren möchtest:  
**Polson, Scott & Windle (2013)**: *Bayesian Inference for Logistic Models Using Pólya‑Gamma Latent Variables.*
