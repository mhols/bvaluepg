# Dokumentation zu `source/polyagammadensity.py`


Diese Datei implementiert Modelle fuer Poisson-Zaehldaten auf Bins. Die zentrale unbekannte Groesse ist ein latentes Feld

```text
f = (f_1, ..., f_N)
```

mit einem Gauß-Prior. Aus `f` wird ueber eine Link-Funktion ein nichtnegatives Poisson-Intensitaetsfeld

```text
lambda_i = field_from_f(f_i)
```

berechnet. Beobachtet werden Zaehldaten

```text
nobs_i ~ Poisson(lambda_i).
```

Die Datei enthaelt drei Link-Varianten:

- `PolyaGammaDensity`: beschraenkte Sigmoid-Intensitaet `lambda_i = lam * sigmoid(f_i)`.
- `RampDensity`: Softplus-/Smooth-Ramp-Intensitaet `lambda_i = softplus(k f_i) / k`.
- `ExponentialDensity`: Exponential-Link `lambda_i = exp(f_i)`.

Fuer 2D-Daten gibt es jeweils 2D-Klassen, die nur zwischen Bildform `(n, m)` und linearer Scan-Reihenfolge vermitteln.

## Grundlegende Hilfsfunktionen

### `to_numpy(x, dtype=float)`

Wandelt skalare Werte, NumPy-Arrays oder iterierbare Objekte in einen flachen NumPy-Vektor um.

Gespeicherte Bedeutung:

- Skalare werden zu einem Vektor der Laenge 1.
- NumPy-Arrays werden mit `.ravel()` linearisiert.
- Iterable werden mit `np.fromiter` eingelesen.

Diese Funktion wird vor allem fuer `prior_mean` genutzt. `prior_mean` ist danach immer ein eindimensionaler Vektor der Laenge `N = nbins`.

### `sigmoid(f)`, `inv_sigmoid(u)`, `der_sigmoid(f)`

Mathematisch:

```text
sigmoid(f) = 1 / (1 + exp(-f))
inv_sigmoid(u) = log(u / (1-u))
der_sigmoid(f) = sigmoid(f) sigmoid(-f)
```

`sigmoid(f)` bildet reelle Werte auf `(0, 1)` ab. Im Sigmoid-Modell wird daraus durch Multiplikation mit `lam` eine beschraenkte Intensitaet in `(0, lam)`.

### `softplus(t)`, `inv_softplus(t)`

Mathematisch:

```text
softplus(t) = log(1 + exp(t))
```

Die Implementierung nutzt eine numerisch stabilere Form:

```text
log1p(exp(-abs(t))) + max(t, 0)
```

`softplus` bildet reelle Werte auf positive Werte ab und wird im Ramp-Modell als glatte Approximation von `max(t, 0)` verwendet.

### `sample_polya_gamma(b, c)`

Zieht unabhaengige Polya-Gamma-Zufallsvariablen

```text
w_i ~ PG(b_i, c_i)
```

mit dem externen Paket `polyagamma`.

Variablen:

- `b`: Array der Formparameter. Im Sigmoid-Gibbs-Sampler ist `b_counts = nobs + k`.
- `c`: Array der Tilt-/Argumentwerte. Im Sampler ist das aktuelle latente Feld `f`.
- `b` wird auf mindestens 1 geklippt, weil das Paket positive ganzzahlige `h` erwartet.
- Rueckgabe `w`: Vektor der Polya-Gamma-Gewichte. Diese Gewichte werden als diagonale Zusatzpraezision `diag(w)` im Gaußschen Vollkonditional fuer `f` verwendet.

## Cholesky, Prior und Permutation

### Zwei Arten, den Gauß-Prior zu speichern

Die Klasse `Density` erlaubt zwei aequivalente Prior-Eingaben:

```text
Covariance mode:
f ~ N(mu0, Sigma0)
prior_covariance = Sigma0
Lprior Lprior.T = Sigma0

Precision mode:
f ~ N(mu0, Q^-1)
prior_precision = Q
Vprior Vprior.T = Q
```

Gespeicherte Attribute:

- `prior_mean`: `mu0`, Vektor der Laenge `N`.
- `prior_covariance`: `Sigma0`, falls der Prior als Kovarianz uebergeben wurde.
- `prior_precision`: `Q`, falls der Prior als Praezision uebergeben wurde.
- `mode`: `Density.COVARIANCE` oder `Density.PRECISION`.
- `sparse`: `True`, wenn eine sparse Praezision genutzt wird.
- `_Lprior`: gecachter dichter Cholesky-Faktor von `Sigma0`.
- `_Vprior`: gecachter Cholesky-Faktor von `Q`; bei sparse Praezision ist das ein CHOLMOD-Factor.

### `sigma0_inv_dot(v, L)`

Berechnet

```text
Sigma0^-1 v
```

ohne `Sigma0^-1` explizit zu bilden. Wenn

```text
Sigma0 = L L.T
```

dann loest die Funktion

```text
L y = v
L.T x = y
```

und gibt `x = Sigma0^-1 v` zurueck.

### `_sparse_cholesky(A)`

Sparse Cholesky wird ueber `sksparse.cholmod.cholesky(A, lower=True)` berechnet. SciPy selbst hat keine native sparse Cholesky-Zerlegung fuer symmetrisch positive definite Matrizen.

CHOLMOD nutzt intern eine Permutation, um Fill-in zu reduzieren. Das ist wichtig: Es wird nicht direkt

```text
A = L L.T
```

zerlegt, sondern in permutierter Reihenfolge

```text
L L.T = P A P.T
```

`P` ist die Permutationsmatrix. In der Datei wird diese Permutation als Integer-Array `perm` gespeichert.

Quelle: Die scikit-sparse-CHOLMOD-Dokumentation beschreibt genau diese Faktorisierung mit einer fill-reducing permutation `P`, naemlich `L L^T = P A P^T`. Sie zeigt ausserdem im Beispiel, dass die Permutation als Indexvektor `p` auf die Matrix mit `A[p][:, p]` angewandt wird. Siehe Quellenabschnitt am Ende.

### `_as_cholmod_lower_factor(factor)`

Normalisiert verschiedene CHOLMOD-Rueckgabeformen zu

```text
lower, perm
```

Bedeutung:

- `lower`: sparse unterer Cholesky-Faktor `L`.
- `perm`: Integer-Array der CHOLMOD-Permutation.

Falls CHOLMOD einen Factor mit Methode `P()` liefert, wird `perm = factor.P()` gespeichert. Falls keine Permutation verfuegbar ist, wird `perm = [0, 1, ..., N-1]` verwendet.

Interpretation von `perm` in dieser Datei:

```text
v_permuted = v[perm]
```

Das heisst: `perm` speichert die Reihenfolge, in der die alte/originale Variable in das permutierte CHOLMOD-System gelesen wird. Der Eintrag `perm[j]` ist der originale Index, der an permutierter Position `j` steht.

Diese Interpretation folgt aus der scikit-sparse-Konvention `P = I[p]` und daraus, dass die permutierte Matrix als `A[p][:, p]` geschrieben wird. In NumPy-Syntax bedeutet das: Die neue/permutierte Position `j` bekommt den alten/originalen Index `p[j]`. Deshalb ist fuer Vektoren die entsprechende Vorwaertspermutation `v[p]`.

Rueckabbildung:

```text
x[perm] = z
```

Dabei ist `z` die Loesung in permutierter Reihenfolge. Durch `x[perm] = z` wird jeder Wert zurueck an seinen originalen Index geschrieben. Damit ist `x` wieder in der urspruenglichen Reihenfolge der Bins gespeichert.

Kurz:

- Vor dem Loesen: originale rechte Seite `b` wird mit `b[perm]` in neue/permutierte Reihenfolge gebracht.
- Nach dem Loesen: permutierte Loesung `z` wird mit `x[perm] = z` wieder in alte/originale Reihenfolge gebracht.
- `perm` ist also kein Array "neuer Index pro altem Index", sondern wird im Code als "alter Index pro neuer/permutierter Position" benutzt.

### `_cholmod_solve_A(factor, b)`

Loest

```text
A x = b
```

mit einer CHOLMOD-Zerlegung.

Falls `factor.solve_A` existiert, wird die CHOLMOD-eigene Loesung verwendet. Sonst wird explizit mit der gespeicherten Permutation gearbeitet:

```text
rhs = b[perm]
L y = rhs
L.T z = y
x[perm] = z
```

Mathematisch passt das zu

```text
L L.T = P A P.T.
```

Die Loesung im permutierten System ist `z = P x`; die Ruecktransformation ist `x = P.T z`. Im Array-Code entspricht diese Ruecktransformation `x[perm] = z`.

Hinweis zur Quelle: scikit-sparse dokumentiert fuer `CholeskyFactor.solve(..., system="A")`, dass nur das `A`-System die Permutation automatisch beruecksichtigt. Die manuelle Loesung in dieser Datei macht diesen Schritt deshalb explizit: erst `b[perm]`, dann die Dreieckssysteme, dann `x[perm] = z`.

### `_cholmod_sample_noise(factor, size)`

Soll einen Gaußschen Rauschterm

```text
eps ~ N(0, A^-1)
```

aus der sparse Cholesky-Zerlegung erzeugen. Bei

```text
L L.T = P A P.T
```

wird im permutierten System

```text
eps_permuted = L.T \ z
eps[perm] = eps_permuted
```

berechnet.

Wichtig zur aktuellen Implementierung:

```python
z = np.ones(size)  # np.random.normal(size=size)
```

Aktuell ist `z` also deterministisch auf Einsen gesetzt. Fuer echtes Sampling aus `N(0, A^-1)` muesste hier normalerweise ein Standardnormalvektor verwendet werden. Die mathematische Absicht ist der Cholesky-Noise-Draw, aber der aktuelle Code erzeugt dadurch keinen zufaelligen Normalterm.

## Gaussian-Mixture-Hilfen

### `_mixture_gaussian_params(z, nobs, mix)`

Diese Funktion wird fuer Link-Funktionen verwendet, bei denen die Poisson-Likelihood durch eine Gauß-Mischung approximiert wird.

Variablen:

- `z`: diskreter Mischungszustand pro Bin.
- `nobs`: beobachtete Zaehldaten pro Bin.
- `mix`: Dictionary mit Mischungsparametern.
- `mu[i]`: Mittelwert der aktiven Gauß-Komponente fuer Bin `i`.
- `s2[i]`: Varianz der aktiven Gauß-Komponente fuer Bin `i`.
- `dinv[i] = 1 / s2[i]`: diagonale Likelihood-Praezision.

Die Funktion liest fuer jeden Bin:

```text
n_i = nobs[i]
component = z[i]
mu[i] = mix["means"][n_i][component]
s2[i] = mix["sigma"][n_i]^2
dinv[i] = 1 / s2[i]
```

Rueckgabe:

```text
mu, dinv
```

### `_sample_f_cond_z_precision(...)`

Zieht `f` bedingt auf Mischungszustaende `z` bei einem Prior in Praezisionsform.

Prior:

```text
f ~ N(mu0, Q^-1)
```

Approximation der Likelihood bedingt auf `z`:

```text
f_i | z_i, n_i approximately N(mu_i, s_i^2)
```

Daraus folgt eine Gaußsche Vollkonditionalverteilung:

```text
A = Q + diag(dinv)
b = Q mu0 + dinv * mu
f | z, n ~ N(A^-1 b, A^-1)
```

Gespeicherte/benutzte Variablen:

- `mu`: komponentenabhaengiger Likelihood-Mittelwert pro Bin.
- `dinv`: komponentenabhaengige Likelihood-Praezision pro Bin.
- `A`: Posterior-Praezision.
- `b`: rechte Seite fuer den Posterior-Mittelwert.
- `mean = A^-1 b`.
- `noise ~ N(0, A^-1)`.
- Rueckgabe: `mean + noise`.

Bei sparse Prior wird `A` sparse gebaut und per CHOLMOD geloest; bei dichtem Prior wird `np.linalg.cholesky(A)` genutzt.

## Klasse `Density`

`Density` ist die Basisklasse fuer das mathematische Modell. Sie kennt Prior, Daten, Likelihood, Posterior-Zielfunktion und allgemeine Sampling-/Optimierungsbausteine. Die konkrete Link-Funktion `field_from_f` kommt aus den Mixins.

### `__init__(prior_mean, prior_covariance, prior_precision, sparse, **kwargs)`

Speichert:

- `kwargs`: Zusatzparameter wie `lam`, `n`, `m`.
- `sparse`: ob sparse Lineare Algebra verwendet wird.
- `prior_mean`, `prior_covariance`, `prior_precision`.

Wenn Kovarianz oder Praezision uebergeben wurde, ruft der Konstruktor `set_prior_Gaussian(...)` auf.

### `set_prior_Gaussian(...)`

Setzt den Gauß-Prior.

Covariance mode:

```text
prior_covariance = Sigma0
mode = COVARIANCE
prior_precision = None
sparse = False
```

Precision mode:

```text
prior_precision = Q
mode = PRECISION
prior_covariance = None
sparse = sparse or issparse(Q)
```

`prior_mean` wird zu einem Vektor der Laenge `N` gemacht. Wenn kein Mittelwert uebergeben wurde, ist `prior_mean = 0`.

### Prior-Anwendungen

Die Methoden

```text
_apply_prior_inverse_covar
_apply_prior_precision_from_covariance
_apply_prior_inverse_precision
_apply_prior_direct_precision
```

beschreiben mathematisch Anwendungen von `Sigma0^-1`, `Q^-1` oder `Q`. Ein Teil der zugehoerigen Lambda-Attribute ist im aktuellen Code auskommentiert; die mathematische Idee ist dennoch:

- In Kovarianzform wird `Q f = Sigma0^-1 f` ueber zwei Dreieckssysteme mit `Lprior` berechnet.
- In Praezisionsform ist `Q` direkt gegeben; dicht optional ueber `Vprior Vprior.T`, sparse direkt als Matrixprodukt.

### `set_data(nobs)`

Speichert die Beobachtungen:

```text
self.nobs = nobs.ravel()
self.ndata = sum(nobs)
```

`nobs` ist ein eindimensionaler Vektor in Scan-Reihenfolge. `ndata` ist die Gesamtzahl aller Ereignisse.

### `Lprior`

Lazy Property fuer

```text
Lprior Lprior.T = prior_covariance = Sigma0
```

Nur im Covariance mode erlaubt.

### `Vprior`

Lazy Property fuer

```text
Vprior Vprior.T = prior_precision = Q
```

Bei sparse Praezision ist `Vprior` kein einfacher Matrixfaktor, sondern das CHOLMOD-Factor-Objekt.

### `get_prior_precision()`

Gibt `Q` zurueck. Im Precision mode ist `Q` schon als `prior_precision` gespeichert.

Im Covariance mode soll `Q = Sigma0^-1` bei Bedarf aus `Lprior` aufgebaut werden. Das kann bei grossen Dimensionen teuer sein, weil eine volle Matrix entsteht.

### Eindimensionale Posterior-Hilfen

`laplace_approximation_one_dimension(m, v2, n)`:

- Baut ein eindimensionales Modell mit Prior `f ~ N(m, v2)`.
- Berechnet den MAP-Schaetzer `pm`.
- Approximiert die Posteriorvarianz mit der inversen negativen Hessian:

```text
pv2 = 1 / H(pm)
```

`posterior_f_one_dimension(f, pm, pv2, n)`:

Nicht normalisierte Posterior-Dichte in `f`:

```text
exp(-(f-pm)^2 / (2 pv2)) * lambda(f)^n * exp(-lambda(f))
```

`posterior_field_one_dimension(field, pm, pv2, n)`:

Posterior in der Feld-/Intensitaetsvariable. Hier kommt die transformierte Gauß-Dichte `density_under_gaussian(field, pm, pv2)` zum Einsatz.

`prior_n_under_gaussian(pm, pv2, n)`:

Approximiert die prior-praediktive Wahrscheinlichkeit fuer Zaehldaten per Gauss-Hermite-Quadratur:

```text
E_f[ Poisson(n | lambda(f)) ]
```

### Abstrakte Link-Methoden

Diese Methoden muessen von einem Mixin definiert werden:

- `field_from_f(f)`: berechnet `lambda`.
- `f_from_field(field)`: inverse Transformation.
- `derivative_log_field_from_f(f)`: Ableitung von `log(lambda(f))`.
- `second_derivative_log_field_from_f(f)`: zweite Ableitung von `log(lambda(f))`.
- `derivative_field_from_f(f)`: Ableitung von `lambda(f)`.
- `second_derivate_field_from_f(f)`: zweite Ableitung von `lambda(f)`.

### Likelihood und Posterior

`loglikelihood(f)` berechnet die Poisson-Log-Likelihood ohne konstante Fakultaetsterme:

```text
log p(n | f) = sum_i n_i log(lambda_i(f_i)) - sum_i lambda_i(f_i)
```

`neg_logposterior(f)` berechnet die negative Log-Posterior-Dichte bis auf Konstanten:

```text
-log p(f | n)
= -log p(n | f) + 1/2 (f-mu0).T Q (f-mu0)
```

Je nach Prior-Modus wird der quadratische Prior-Term berechnet als:

```text
Precision mode:   (f-mu0).T Q (f-mu0)
Covariance mode:  || Lprior^-1 (f-mu0) ||^2
```

`neg_grad_logposterior(f)` berechnet den Gradienten:

```text
negative gradient = -[n_i d/df log(lambda_i) - d/df lambda_i] + Q(f-mu0)
```

`hessian_neg_log_posterior(f)` berechnet die Hessian der negativen Log-Posterior-Dichte:

```text
D + Q
```

mit diagonaler Likelihood-Kruemmung `D`.

Achtung: Im Code steht

```python
D = -np.diag(self.ndata * second_derivative_log_field_from_f(f)
             - second_derivate_field_from_f(f))
```

Mathematisch waere fuer unabhaengige Bins naheliegend, die binweisen `nobs_i` statt der Gesamtzahl `ndata` zu verwenden. Die Dokumentation beschreibt hier die Implementierung, nicht eine Korrektur.

### `first_guess_estimator(f=None, s2=None)`

Berechnet eine Gaußsche Naeherung als Startwert fuer Optimierung oder Sampling.

Interpretation:

- `f`: pixelweise Schaetzung des latenten Feldes.
- `s2`: pixelweise Varianz dieser Schaetzung.
- `dinv = 1/s2`: diagonale Beobachtungspraezision.

Das resultierende lineare Gauß-Modell ist:

```text
f_obs approximately N(f, diag(s2))
f_true ~ N(mu0, Q^-1)
```

Posterior-Mittel:

```text
(Q + diag(dinv))^-1 (Q mu0 + dinv * f_obs)
```

Im Covariance mode wird eine Woodbury-/Whitening-Form mit `Lprior` verwendet, um `Q` nicht explizit bilden zu muessen.

### `max_logposterior_estimator(...)`

Minimiert `neg_logposterior(f)` mit SciPy und gibt den MAP-Schaetzer zurueck:

```text
f_MAP = argmin_f neg_logposterior(f)
```

Startwert ist, wenn nicht anders angegeben, `first_guess_estimator()`.

## `SigmoidMixin` und `PolyaGammaDensity`

`SigmoidMixin` definiert den beschraenkten Link

```text
lambda_i = lam * sigmoid(f_i)
```

`lam` wird in `kwargs['lam']` gespeichert und per Property gelesen.

Ableitungen:

```text
d lambda / df = lam * sigmoid(f) * sigmoid(-f)
d^2 lambda / df^2 = lam * sigmoid(f) * sigmoid(-f) * (sigmoid(-f) - sigmoid(f))
d log(lambda) / df = sigmoid(-f)
d^2 log(lambda) / df^2 = -sigmoid(f) sigmoid(-f)
```

Inverse:

```text
s = field / lam
f = log(s / (1-s))
```

Im Code wird `s` unten auf `0.01` geklippt.

### Polya-Gamma-Gibbs-Sampler `sample_posterior`

Dieser Sampler nutzt eine Datenaugmentation fuer den Sigmoid-Link. Die Idee ist, den Faktor

```text
lambda(f)^n exp(-lambda(f))
= [lam sigmoid(f)]^n exp(-lam sigmoid(f))
```

durch zusaetzliche negative Zaehldaten `k` und Polya-Gamma-Variablen `w` in ein bedingt Gaußsches Modell fuer `f` zu verwandeln.

Pro Iteration:

1. Latente negative Zaehlungen:

```text
rate_neg = lam * sigmoid(-f)
k_i ~ Poisson(rate_neg_i)
```

Im Code:

```python
rate_neg = self.field_from_f(-f)
k = np.random.poisson(rate_neg)
```

2. Polya-Gamma-Gewichte:

```text
b_i = nobs_i + k_i
w_i ~ PG(b_i, f_i)
```

Im Code:

```python
b_counts = (self.nobs + k).astype(int)
w = sample_polya_gamma(b_counts, f)
```

3. Linearterm:

```text
kappa_i = 0.5 * (nobs_i - k_i)
```

4. Gaußsche Vollkonditionalverteilung:

Wenn der Prior `f ~ N(mu0, Q^-1)` ist, dann ist

```text
A = Q + diag(w)
b = Q mu0 + kappa
f | w, k, n ~ N(A^-1 b, A^-1)
```

Im Code werden drei lineare Algebra-Faelle unterschieden.

#### Covariance mode, dicht

Gegeben:

```text
Sigma0 = L L.T
Q = Sigma0^-1
A = Q + diag(w)
```

Der Code arbeitet in whitened coordinates `f = L u`. Dann gilt:

```text
A = L^-T M L^-1
M = I + L.T diag(w) L
```

Gespeicherte Variablen:

- `weighted_L = w[:, None] * Lprior`: entspricht `diag(w) L`.
- `M = I + Lprior.T @ weighted_L`.
- `chol`: Cholesky-Faktor von `M`.
- `rhs = L^-1 mu0 + L.T kappa`.
- `m = L M^-1 rhs`: Posterior-Mittel in originaler `f`-Reihenfolge.
- `eps = L chol^-T z`: Noise mit intendierter Kovarianz `A^-1`.
- `f = m + eps`.

Achtung: Auch hier ist im aktuellen Code `z = np.ones(nbins)` statt `np.random.normal(...)`, dadurch ist der Noise deterministisch.

#### Precision mode, sparse

Gegeben:

```text
Q = prior_precision
A = Q + diag(w)
bvec = Q mu0 + kappa
```

Im Code:

```python
A = (self.prior_precision + sps.diags(w, format="csc")).tocsc()
factor = _sparse_cholesky(A)
m = _cholmod_solve_A(factor, bvec)
eps = _cholmod_sample_noise(factor, nbins)
f = m + eps
```

Hier entsteht die CHOLMOD-Permutation. `A` wird von CHOLMOD aus Fill-in-Gruenden in eine neue Ordnung gebracht. `m` und `eps` werden am Ende durch `x[perm] = z` beziehungsweise `eps[perm] = eps_permuted` wieder in die alte/originale Bin-Reihenfolge zurueckgeschrieben. Das gespeicherte `f` ist also wieder in der normalen Scan-Reihenfolge.

#### Precision mode, dicht

Gegeben:

```text
A = Q + diag(w)
bvec = Q mu0 + kappa
```

Der Code berechnet:

```text
chol chol.T = A
m = A^-1 bvec
eps = chol^-T z
f = m + eps
```

Auch hier ist aktuell `z = np.ones(nbins)` statt eines Zufallsvektors.

### Speicherung der Samples

`sample_posterior` ist ein Generator. Er speichert nicht das gesamte Array `f_samples`; obwohl `f_samples` angelegt wird, wird es nicht befuellt oder zurueckgegeben. Stattdessen:

- Bei akzeptierter Iteration wird `self.last_sample = f` gesetzt.
- Dann wird `yield f` ausgefuehrt.
- `last_sample` dient beim naechsten Aufruf als Startwert, falls kein `initial_f` uebergeben wird.

## `SmoothRampMixin` und `RampDensity`

Der Ramp-Link ist:

```text
lambda = softplus(k f) / k
```

`softplus_k` wird als `self.softplus_k` gespeichert. Fuer `k = 1` ist `lambda = softplus(f)`.

Inverse:

```text
f = log(expm1(k field)) / k
```

Ableitungen laut Code:

```text
d lambda / df = sigmoid(k f)
d^2 lambda / df^2 = sigmoid(f) sigmoid(-f)
d log(lambda) / df = sigmoid(f) / softplus(f)
d^2 log(lambda) / df^2 =
    sigmoid(f) sigmoid(-f) / softplus(f)
    - sigmoid(f)^2 / softplus(f)^2
```

Achtung: Die zweite Ableitung und die Log-Ableitungen verwenden im Code `f` statt `k f`. Fuer `softplus_k != 1` ist das mathematisch nicht konsistent mit `field_from_f`.

### Mixture-Sampler

`RampDensity.sample_posterior` nutzt eine externe Gauß-Mischungsapproximation aus `gibbs_softplus_mixture`.

Gespeicherte Attribute:

- `_mix`: Cache fuer die Mischungsparameter.
- `nmax_mix`: maximale beobachtete Count-Klasse fuer die Mischung.
- `cache_dir`: Ordner fuer gecachte Mischungsdaten.
- `softplus_k`: Skalierungsparameter des Links.

Property `mix` laedt oder baut:

```text
self._mix = gsm.load_or_build_mix(nmax_mix, cache_dir, softplus_k)
```

Sampler-Schritte:

```text
z_i ~ p(z_i | f_i, nobs_i, mix)
f ~ p(f | z, nobs, prior)
```

Im Code:

```python
fz_cache = gsm.prepare_f_cond_z(self)
z = gsm.sample_z_cond_f(f, self.nobs, self.mix)
f = gsm.sample_f_cond_z_cache(z, self, fz_cache)
```

`z` speichert pro Bin den aktiven Mischungsindex. Die bedingte Verteilung von `f` wird danach als Gaußverteilung mit priorbasierter Praezision und diagonaler Likelihood-Praezision geloest.

## `ExponentialMixin` und `ExponentialDensity`

Der Exponential-Link ist:

```text
lambda = exp(f)
```

Im Code wird `eme.safe_exp(f)` verwendet, um numerische Probleme zu reduzieren.

Inverse:

```text
f = log(field)
```

mit Clipping gegen `log(0)`.

Ableitungen:

```text
d lambda / df = exp(f)
d^2 lambda / df^2 = exp(f)
d log(lambda) / df = 1
d^2 log(lambda) / df^2 = 0
```

Der Sampler ist analog zum Ramp-Sampler, nutzt aber Mischungsparameter aus `exp_mix_explink`:

```python
self._mix = eme.load_or_build_exp_mix(self.nmax_mix, self.cache_dir)
```

Auch hier speichert `z` die aktive Gauß-Mischungskomponente pro Bin, und `f` wird bedingt auf `z` aus einer Gaußverteilung gezogen.

## Konkrete Dichteklassen

### `PolyaGammaDensity(SigmoidMixin, Density)`

Kombiniert:

- Gauß-Prior und Posterior-Logik aus `Density`.
- Sigmoid-Link und Polya-Gamma-Sampler aus `SigmoidMixin`.

Wichtig fuer Initialisierung:

```python
PolyaGammaDensity(prior_mean, prior_covariance=..., lam=...)
PolyaGammaDensity(prior_mean, prior_precision=..., sparse=True, lam=...)
```

### `RampDensity(SmoothRampMixin, Density)`

Kombiniert:

- Gauß-Prior aus `Density`.
- Softplus-Link und Gauß-Mischungs-Sampler aus `SmoothRampMixin`.

### `ExponentialDensity(ExponentialMixin, Density)`

Kombiniert:

- Gauß-Prior aus `Density`.
- Exponential-Link und Gauß-Mischungs-Sampler aus `ExponentialMixin`.

## 2D-Mixins und 2D-Klassen

### `Mixin2D`

Speichert keine eigene Matrixstruktur, sondern interpretiert Vektoren der Laenge `N = n*m` als Bilder.

Parameter:

- `n = kwargs['n']`: Anzahl Zeilen.
- `m = kwargs['m']`: Anzahl Spalten.

Methoden:

- `image_to_scanorder(image)`: macht aus einem 2D-Array einen flachen Vektor mit `.ravel()`.
- `scanorder_to_image(linear_image, n=None, m=None)`: formt einen Vektor zurueck nach `(n, m)`.
- `random_catalog_from_nobs(nobs)`: erzeugt zufaellige Ereigniskoordinaten innerhalb der Bins. Fuer jeden Count in Bin `(i, j)` wird ein Punkt `(i + U[0,1), j + U[0,1))` erzeugt.
- `imshow(d, **kwargs)`: zeigt einen Vektor als Bild.

Die Reihenfolge ist die normale NumPy-C-Reihenfolge von `.ravel()`: erst die Spalten innerhalb einer Zeile, dann die naechste Zeile.

### `Density2D`, `PolyaGammaDensity2D`, `RampDensity2D`, `ExponentialDensity2D`

Diese Klassen kombinieren jeweils `Mixin2D` mit der entsprechenden 1D-Dichteklasse. Mathematisch aendert sich nichts am Modell; nur die Interpretation der linearen Indizes als 2D-Bins kommt hinzu.

## Wichtigste Variablen im Ueberblick

```text
N / nbins
    Anzahl Bins, also Dimension von f.

f
    Latentes Gauß-Feld in originaler Scan-Reihenfolge.

field / lambda
    Nichtnegative Poisson-Intensitaet pro Bin.

nobs
    Beobachtete Counts pro Bin, flach gespeichert.

ndata
    Summe aller Counts.

prior_mean / mu0
    Prior-Mittelwert von f.

prior_covariance / Sigma0
    Prior-Kovarianz, wenn Covariance mode genutzt wird.

prior_precision / Q
    Prior-Praezision, wenn Precision mode genutzt wird.

Lprior
    Unterer Cholesky-Faktor von Sigma0, also Sigma0 = Lprior Lprior.T.

Vprior
    Cholesky-Faktor von Q oder CHOLMOD-Factor bei sparse Q.

w
    Polya-Gamma-Gewichte; ergeben diag(w) als diagonale Likelihood-Praezision.

k
    Latente negative Poisson-Counts im Sigmoid-/Polya-Gamma-Sampler.

kappa
    Linearer Term 0.5 * (nobs - k) im Polya-Gamma-Gaußmodell.

A
    Posterior-Praezision fuer f in einer Gibbs-Iteration.

bvec / b
    Rechte Seite fuer Posterior-Mittel: Q mu0 + Likelihood-Term.

m / mean
    Posterior-Mittel A^-1 b.

eps / noise
    Rauschterm mit intendierter Kovarianz A^-1.

z
    Je nach Kontext:
    - Standardnormalvektor fuer Sampling, im aktuellen Code teils durch Einsen ersetzt.
    - Mischungszustand in Ramp-/Exponential-Samplern.

perm
    CHOLMOD-Permutation. Wird als alte/originale Indizes in neuer/permutierter Reihenfolge verwendet:
    b_permuted = b[perm], Rueckabbildung x[perm] = x_permuted.
```


## Quellen zu CHOLMOD und Permutationen

Die CHOLMOD-/Permutationserklaerung basiert auf folgenden Quellen:

1. scikit-sparse 0.5.0, User Guide: "Cholesky Decomposition (`sksparse.cholmod`)"
   URL: https://scikit-sparse.readthedocs.io/en/latest/tutorial/cholmod.html

   Relevante Punkte:

   - Das Modul ist ein Interface zu SuiteSparse CHOLMOD fuer sparse, symmetrisch positiv definite Matrizen.
   - Die Dokumentation beschreibt die Cholesky-Faktorisierung mit fill-reducing permutation als `L L^T = P A P^T`.
   - Im Beispiel wird die permutierte Matrix mit `A[p][:, p]` gebildet. Daraus folgt die in dieser Datei verwendete Vektor-Konvention `v_permuted = v[perm]`.
   - Das Beispiel nennt die Motivation der Permutation: weniger Fill-in im Cholesky-Faktor.

2. scikit-sparse 0.5.0, API Reference: `CholeskyFactor.get_perm()`
   URL: https://scikit-sparse.readthedocs.io/en/latest/reference/generated/sksparse.cholmod.CholeskyFactor.get_perm.html

   Relevante Punkte:

   - `get_perm()` gibt den Permutationsvektor zurueck, der in der Faktorisierung verwendet wurde.
   - Die Dokumentation definiert `P = I[p]`.
   - `P A P^T` ist die Matrix, die faktorisiert wurde.
   - Mit `P = I[p]` ist `p[j]` der alte/originale Index, der in der neuen/permutierten Zeile/Spalte `j` steht.

3. scikit-sparse 0.5.0, API Reference: `CholeskyFactor.solve()`
   URL: https://scikit-sparse.readthedocs.io/en/latest/reference/generated/sksparse.cholmod.CholeskyFactor.solve.html

   Relevante Punkte:

   - `solve(b, system="A")` loest `A x = b`.
   - Die Dokumentation weist darauf hin, dass nur das System `"A"` die Permutation automatisch beruecksichtigt.
   - Loest man dagegen manuell mit den Faktoren `L` und `L.T`, muss man die Permutation selbst anwenden. Genau deshalb nutzt der Code `rhs = b[perm]` und danach `x[perm] = z`.

4. scikit-sparse 0.4.4, stabile aeltere Dokumentation: "Sparse Cholesky decomposition (`sksparse.cholmod`)"
   URL: https://scikit-sparse.readthedocs.io/en/stable/cholmod.html

   Relevante Punkte:

   - Diese aeltere API-Dokumentation passt zu Code, der noch mit `Factor`-Objekten und Methoden wie `factor.L()`, `factor.P()` oder `solve_A()` arbeitet.
   - Sie bestaetigt, dass scikit-sparse CHOLMOD-Faktorisierungen mit fill-reducing permutation bereitstellt und zum Loesen von Systemen `A x = b` verwendet.

5. SuiteSparse / CHOLMOD Projekt
   URL: https://github.com/DrTimothyAldenDavis/SuiteSparse

   Relevanter Punkt:

   - CHOLMOD ist die SuiteSparse-Komponente fuer sparse Cholesky-Faktorisierungen. scikit-sparse ist hier nur das Python-Interface; die eigentliche sparse Cholesky-Logik und die Fill-reducing-Orderings kommen aus CHOLMOD/SuiteSparse.
