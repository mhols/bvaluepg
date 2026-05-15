# Benchmark-Plan: Dense vs. Sparse Cholesky für PG-Gibbs-Schritt

## Ziel

Wir wollen experimentell prüfen, wie sich verschiedene Matrix-Repräsentationen und Bibliotheken beim Cholesky-artigen Schritt verhalten.

Motivation aus `polyagammadensity.py`:

```python
L = self.Lprior
I = np.eye(self.nbins)

X = spla.solve_triangular(L, I, lower=True)
Sigma0_inv = spla.solve_triangular(L.T, X, lower=False)

A = Sigma0_inv + np.diag(w)
chol = np.linalg.cholesky(A)
```

Die zentrale Matrix im Gibbs-Sampler ist:

\[
A = \Sigma_0^{-1} + \operatorname{diag}(w)
\]

bzw. später bei sparse Prior:

\[
A = Q + \operatorname{diag}(w)
\]

---

## Fragestellungen

1. Wie groß ist der Speicherbedarf der Matrix in verschiedenen Formaten?
2. Wie schnell ist eine Cholesky-Zerlegung bzw. äquivalente Faktorisierung?
3. Bringt sparse Speicherung allein etwas?
4. Ab welcher Matrixgröße lohnt sich sparse wirklich?
5. Welche Bibliothek ist für unseren Anwendungsfall praktisch nutzbar?

---

## Getestete Varianten

### 1. NumPy / SciPy dense

Referenzvariante wie aktuell:

```python
A_dense = Sigma0_inv + np.diag(w)
np.linalg.cholesky(A_dense)
```

oder:

```python
scipy.linalg.cholesky(A_dense, lower=True)
```

Ziel: Baseline für Geschwindigkeit und Speicher.

---

### 2. SciPy sparse

Speicherung als:

```python
scipy.sparse.csr_matrix
scipy.sparse.csc_matrix
```

Matrix:

```python
A_sparse = Q_sparse + scipy.sparse.diags(w)
```

Wichtig: SciPy hat sparse Speicher und sparse Solver, aber keine direkte native sparse Cholesky wie `np.linalg.cholesky`.

Getestet werden kann:

```python
scipy.sparse.linalg.splu(A_sparse)
```

als Vergleich, aber das ist LU, nicht Cholesky.

---

### 3. scikit-sparse / CHOLMOD

Eigentlich die wichtigste sparse-Cholesky-Variante:

```python
from sksparse.cholmod import cholesky

factor = cholesky(A_sparse)
```

Ziel: echte sparse Cholesky testen.

---

### 4. PyTorch

Dense:

```python
torch.linalg.cholesky(A_torch_dense)
```

Sparse Speicherung möglich, aber sparse Cholesky ist nicht direkt Standard wie bei CHOLMOD.

Zu testen:

- Speicher als dense Tensor
- Speicher als sparse COO/CSR
- Cholesky nur dense als realistische Baseline

---

### 5. TensorFlow

Dense:

```python
tf.linalg.cholesky(A_tf_dense)
```

Sparse Tensor möglich, aber sparse Cholesky nicht direkt als Standardoperation.

Zu testen:

- Speicher als `tf.sparse.SparseTensor`
- Cholesky dense

---

### 6. scikit-learn

scikit-learn ist keine Cholesky-Bibliothek für diesen Zweck.

Nützlich höchstens zum Erzeugen von Testmatrizen, z. B.:

```python
sklearn.datasets.make_sparse_spd_matrix
```

Für die eigentliche Faktorisierung eher nicht relevant.

---

## Testmatrizen

Wir sollten zwei Matrixklassen testen.

### A) Dense covariance-artige Matrix wie aktuell

Ähnlich zu:

```python
covar = ck.spatial_covariance_matern_2_3(n, n, rho, v2)
```

Dann:

```python
L = cholesky(covar)
Sigma0_inv = ...
A = Sigma0_inv + diag(w)
```

Erwartung:

- Matrix ist mathematisch dicht.
- Sparse Speicherung bringt wenig.
- Cholesky bleibt teuer.

---

### B) Sparse precision-artige Matrix

Zum Beispiel GMRF/Laplace-Prior:

\[
Q = \tau I + \alpha L_{\text{grid}}
\]

Dann:

```python
A = Q + diag(w)
```

Erwartung:

- Matrix ist wirklich sparse.
- Speicherbedarf sinkt massiv.
- CHOLMOD sollte deutlich profitieren.

---

## Matrixgrößen

Start klein, dann steigern:

```text
n = 32  -> N = 1024
n = 64  -> N = 4096
n = 96  -> N = 9216
n = 128 -> N = 16384
```

Falls Speicher knapp wird, bei 96 oder 128 abbrechen.

---

## Wiederholungen

Für jede Variante:

```text
warmup: 1-2 Läufe
repetitions: 5-20 Läufe
```

In jeder Wiederholung:

```python
w = positive random vector
A = Q + diag(w)
factorize(A)
```

Gemessen wird:

```text
Zeit pro Faktorisierung
mittlere Zeit
Standardabweichung
Speicherbedarf Matrix
Anzahl Nicht-Null-Einträge
Dichte nnz / N²
```

---

## Speicher-Messung

Dense:

```python
A.nbytes
```

SciPy sparse:

```python
A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
```

PyTorch dense:

```python
tensor.element_size() * tensor.nelement()
```

PyTorch sparse:

```python
values + indices Speicher
```

TensorFlow:

nur näherungsweise über Werte + Indizes.

---

## Erwartete Erkenntnis

Sparse lohnt sich vermutlich nicht für die aktuelle dense Kovarianzmatrix.

Sparse lohnt sich wahrscheinlich stark, wenn wir den Prior direkt als sparse Präzisionsmatrix formulieren:

\[
f \sim \mathcal N(\mu, Q^{-1})
\]

statt:

\[
f \sim \mathcal N(\mu, \Sigma)
\]

Dann wird der Gibbs-Schritt:

```python
A = Q_sparse + sparse.diags(w)
```

und nicht mehr:

```python
A = Sigma0_inv_dense + np.diag(w)
```

---

## Ergebnisformat

Am Ende eine Tabelle:

| backend | matrix type | N | nnz | density | memory MB | mean chol time | std time |
|---|---:|---:|---:|---:|---:|---:|---:|

Zusätzlich optional ein Plot:

- Cholesky-Zeit gegen N
- Speicherbedarf gegen N
- dense vs sparse comparison

---

## Zusätzliche Ideen / Erweiterungen

### 1. Vergleich unterschiedlicher Speicherformate

Innerhalb von SciPy zusätzlich vergleichen:

```python
CSR
CSC
COO
```

CHOLMOD bevorzugt typischerweise CSC.

---

### 2. Einfluss der Matrixdichte

Künstlich unterschiedliche Sparsity testen:

```text
1%
5%
10%
20%
50%
```

um den Übergang dense ↔ sparse besser zu verstehen.

---

### 3. Nur Faktorisierung vs. vollständiger Solve

Zusätzlich messen:

```python
x = factor.solve_A(b)
```

Denn im echten Gibbs-Sampler wird nicht nur faktorisiert, sondern auch gelöst.

---

### 4. Reordering testen

Sparse Cholesky hängt stark vom Reordering ab.

Zum Beispiel:

```python
natural
amd
metis
```

Fill-In könnte großen Einfluss haben.

---

### 5. GPU-Test

Optional später:

- PyTorch CUDA
- TensorFlow GPU

Interessant vor allem für dense Matrizen.

---

### 6. Speicher während Faktorisierung

Nicht nur Matrixgröße messen, sondern Peak-Memory während Cholesky.

Zum Beispiel mit:

```python
memory_profiler
psutil
tracemalloc
```

---

## Fazitkriterium

Für unser PG-Projekt ist nicht entscheidend, welche Bibliothek sparse speichern kann.

Entscheidend ist:

> Welche Variante kann `A = Q + diag(w)` schnell und stabil faktorisieren?

Vermutlich beste Kandidaten:

1. dense SciPy/NumPy als Baseline
2. scikit-sparse/CHOLMOD für echte sparse Cholesky
3. PyTorch/TensorFlow nur dann interessant, wenn wir später GPU dense testen wollen
