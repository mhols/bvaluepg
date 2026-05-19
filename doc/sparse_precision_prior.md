# Sparse Precision Prior in `polyagammadensity.py`

## Motivation

Der bisherige Polya-Gamma-Sampler verwendet einen Gaussian Prior ueber eine
dichte Kovarianzmatrix:

```python
Sigma0_inv = Sigma0^{-1}
A = Sigma0_inv + np.diag(w)
chol = np.linalg.cholesky(A)
```

Bei den Matern-Kovarianzen aus `source/syntheticdata.py` ist `Sigma0_inv`
numerisch dicht. Sparse Speicherung hilft in diesem Fall nicht, weil die
Posterior-Precision `A` ebenfalls dicht bleibt.

Der neue Pfad ist fuer Priors gedacht, die direkt als sparse Precision
formuliert werden:

```text
f ~ N(mu, Q^{-1})
```

Dann wird im Gibbs-Schritt:

```python
A = Q + scipy.sparse.diags(w)
```

und `A` bleibt sparse.

## Geaenderte Stellen

Die Aenderungen liegen in `source/polyagammadensity.py`.

### 1. Sparse Precision als Prior

`Density.set_prior_Gaussian(...)` akzeptiert jetzt weiterhin den alten
Covariance-Pfad:

```python
set_prior_Gaussian(prior_mean=mu, prior_covariance=Sigma)
```

Zusaetzlich kann eine Precision-Matrix gesetzt werden:

```python
set_prior_Gaussian(prior_mean=mu, prior_precision=Q, sparse=True)
```

Als klarerer Wrapper existiert:

```python
set_prior_precision_sparse(prior_mean=mu, prior_precision=Q)
```

Intern wird `Q` als `scipy.sparse.csc_matrix` gespeichert.

### 2. Precision-Modus

Wenn ein Precision-Prior gesetzt ist, wird der Prior-Term direkt als

```python
Q @ (f - mu)
```

verwendet. Der dense Covariance-Pfad bleibt unveraendert.

### 3. Sparse Gibbs-Schritt

In `PolyaGammaDensity.sample_posterior(...)` gibt es jetzt zwei Pfade:

```python
# alter dense Covariance-Pfad
A = Sigma0_inv + np.diag(w)

# neuer sparse Precision-Pfad
A = Q + scipy.sparse.diags(w, format="csc")
```

Die Matrix bleibt dabei sparse. Fuer die eigentliche sparse Cholesky-
Faktorisierung wird `scikit-sparse/CHOLMOD` verwendet, weil SciPy zwar sparse
Matrixformate und sparse LU (`splu`) anbietet, aber keine native sparse
Cholesky-Faktorisierung.

Falls `scikit-sparse` nicht installiert ist, wirft der sparse Sampling-Pfad
eine klare Fehlermeldung.

### 4. Sparse Gaussian-Mixture-Sampler

`RampDensity`/`RampDensity2D` und `ExponentialDensity`/`ExponentialDensity2D`
verwenden keinen Polya-Gamma-Schritt, sondern einen Gaussian-Mixture-Sampler.
Auch dieser Pfad kann jetzt einen sparse Precision-Prior nutzen.

Bei festem Mixture-Zustand `z` ist die Likelihood diagonal-gaussian:

```text
diag(dinv), mu_z
```

Mit sparse Prior-Precision `Q` wird die bedingte Posterior-Precision:

```python
A = Q + scipy.sparse.diags(dinv, format="csc")
```

und das Sample wird wieder ueber CHOLMOD gezogen:

```text
f | z, n ~ N(A^{-1} b, A^{-1})
```

## Installation von CHOLMOD

Der sparse Sampling-Pfad braucht CHOLMOD. CHOLMOD ist nicht Teil von
scikit-learn und auch nicht direkt Teil von SciPy. Die Abhaengigkeiten sind:

```text
SuiteSparse
  -> CHOLMOD
       -> scikit-sparse Python-Wrapper
```

Das Python-Paket heisst `scikit-sparse` und wird so importiert:

```python
from sksparse.cholmod import cholesky
```

### Ubuntu

Auf Ubuntu zuerst SuiteSparse und Build-Werkzeuge installieren:

```bash
sudo apt update
sudo apt install build-essential python3-dev libsuitesparse-dev
```

Dann in der Python-venv:

```bash
source /path/to/venv/bin/activate
pip install scikit-sparse
```

Installation testen:

```bash
python -c "from sksparse.cholmod import cholesky; print('CHOLMOD OK')"
```

### macOS

Auf macOS mit Homebrew:

```bash
brew install suite-sparse
```

Dann in der Python-venv:

```bash
source /path/to/venv/bin/activate
pip install scikit-sparse
```

Falls `pip install scikit-sparse` CHOLMOD nicht findet:

```bash
export SUITESPARSE_INCLUDE_DIR="$(brew --prefix suite-sparse)/include/suitesparse"
export SUITESPARSE_LIBRARY_DIR="$(brew --prefix suite-sparse)/lib"
pip install scikit-sparse
```

## Beispiel: Sparse Grid-Prior

Ein typischer 2D-Gitter-Prior ist eine Laplace/GMRF-artige Precision:

```python
import numpy as np
import scipy.sparse as sps
from polyagammadensity import PolyaGammaDensity2D


def grid_precision(n, tau=1.0, alpha=0.2):
    one_dim = sps.diags(
        [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    eye = sps.eye(n, format="csr")
    laplacian = (
        sps.kron(eye, one_dim, format="csr")
        + sps.kron(one_dim, eye, format="csr")
    )
    return (tau * sps.eye(n * n, format="csr") + alpha * laplacian).tocsc()


n = 64
Q = grid_precision(n)
prior_mean = np.zeros(n * n)

pgd = PolyaGammaDensity2D(
    prior_mean=prior_mean,
    prior_precision=Q,
    sparse=True,
    lam=10.0,
    n=n,
    m=n,
)

pgd.set_data(nobs)

for f in pgd.sample_posterior(
    n_iter=100,
    initial_f=np.zeros(n * n),
    random_seed=123,
):
    field = pgd.field_from_f(f)
```

Alternativ kann ein bestehendes Objekt erst ohne Prior erzeugt und danach
gesetzt werden:

```python
pgd = PolyaGammaDensity2D(lam=10.0, n=n, m=n)
pgd.set_prior_precision_sparse(prior_mean, Q)
```

## Nutzung in `source/syntheticdata.py`

`experiment_1()` bleibt unveraendert und nutzt weiterhin die dichte
Matern-Kovarianz:

```python
covar = ck.spatial_covariance_matern_2_3(n, n, rho, v2)
estim.set_prior_Gaussian(pm, covar)
```

Fuer den sparse Precision-Pfad gibt es zusaetzlich:

```python
syntheticdata.experiment_1_sparse_precision(...)
```

Diese Variante erzeugt dieselben checkerboard-artigen synthetischen Daten wie
`experiment_1()`, setzt den Prior aber als sparse Grid-Precision:

```python
precision = grid_precision_laplacian(n, tau=tau, alpha=alpha)
estim.set_prior_precision_sparse(pm, precision)
```

Ein minimaler Aufruf:

```python
import syntheticdata as sd
import polyagammadensity as pgd

sd.experiment_1_sparse_precision(
    EstimatorClass=pgd.PolyaGammaDensity2D,
    n=64,
    tau=1.0,
    alpha=0.2,
    lam=10,
)
```

Die Funktion kann aktuell mit diesen Klassen genutzt werden:

```python
sd.experiment_1_sparse_precision(
    EstimatorClass=pgd.PolyaGammaDensity2D,
)

sd.experiment_1_sparse_precision(
    EstimatorClass=pgd.RampDensity2D,
)

sd.experiment_1_sparse_precision(
    EstimatorClass=pgd.ExponentialDensity2D,
)
```

Bei `RampDensity2D` und `ExponentialDensity2D` muss `nmax_mix` mindestens so
gross sein wie die groesste beobachtete Zellzaehlung. Sonst bricht der
Mixture-Sampler mit einer Meldung wie `count ... exceeds precomputed nmax`
ab. In diesem Fall `nmax_mix` erhoehen.

Die Parameter bedeuten:

```text
tau   -> Diagonal-/Ridge-Anteil; macht Q strikt positiv definit
alpha -> Staerke der Gitter-Laplace-Kopplung; groesser bedeutet glatterer Prior
```

Die sparse Variante verwendet nicht dieselbe Prior-Verteilung wie die alte
Matern-Covariance. Sie ist der rechnerisch schnelle GMRF/Precision-Prior, der
den sparse Cholesky-Pfad aktiviert.

## Was gleich bleibt

Der alte Aufruf mit dichter Kovarianz bleibt weiterhin gueltig:

```python
pgd = PolyaGammaDensity2D(
    prior_mean=prior_mean,
    prior_covariance=Sigma,
    lam=10.0,
    n=n,
    m=n,
)
```

Dieser Pfad verwendet weiterhin dense Cholesky. Dadurch sollten bestehende
Experimente mit Matern-Kovarianz dieselbe Modellklasse und dasselbe Verhalten
behalten.

## Erwartete Performance

Der sparse Pfad ist nur schneller, wenn der Prior wirklich als sparse
Precision `Q` vorliegt. Eine dichte Matern-Kovarianz nachtraeglich sparse zu
speichern bringt keinen Vorteil, weil ihre Inverse dicht ist.

Aus den Benchmarks:

```text
n=64, N=4096

dense Matern Precision:
  Matrix: 128 MB
  dense Cholesky: ca. 0.55 s

sparse Grid Precision:
  Matrix: ca. 0.25 MB
  CHOLMOD sparse Cholesky: ca. 0.004 s
```

Die Ergebnisse sind nicht bitweise identisch, weil die sparse Cholesky mit
Permutation arbeitet und Zufallszahlen in anderer Reihenfolge in den Faktor
eingehen koennen. Die Zielverteilung ist dieselbe fuer denselben Precision-
Prior `Q`.
