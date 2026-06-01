# Documentation for `source/polyagammadensity.py`

This is an English working version of the German documentation
`polyagammadensity_dokumentation.md`. It describes the current code path and
keeps the older helper functions as historical notes because they are useful for
understanding the sparse CHOLMOD permutation logic.

## Basic Helper Functions

### `to_numpy(x, dtype=float)`

Converts scalar and iterable inputs into one-dimensional NumPy arrays. Existing
NumPy arrays are flattened with `ravel()`.

### `sigmoid(f)`, `inv_sigmoid(u)`, `der_sigmoid(f)`

These implement the logistic link and related transforms:

```text
sigmoid(f) = 1 / (1 + exp(-f))
inv_sigmoid(u) = log(u / (1-u))
der_sigmoid(f) = sigmoid(f) sigmoid(-f)
```

The sigmoid implementation is simple and currently not overflow-protected for
very large negative `f`.

### `softplus(t)`, `inv_softplus(t)`

The softplus implementation is numerically stable:

```python
softplus(t) = log1p(exp(-abs(t))) + max(t, 0)
```

It is used by the ramp-like link functions.

### `sample_polya_gamma(b, c)`

Draws samples from `PG(b, c)` using the `polyagamma` package. The parameter
`b` is clipped to at least one because the package requires positive integer
shape parameters.

## Cholesky, Priors, and Permutations

### Two Ways to Store the Gaussian Prior

The latent parameter field `f` has a Gaussian prior. The code supports two
representations:

```text
Covariance mode:
    f ~ N(mu, Sigma)
    Sigma = L L.T

Precision mode:
    f ~ N(mu, Q^-1)
    Q = prior_precision
```

Covariance mode is convenient for dense small problems. Precision mode is the
important representation for sparse spatial priors.

### Dense Covariance Cholesky

If `Sigma = L L.T`, then applying the inverse covariance to a vector can be
done without forming `Sigma^-1`:

```text
L y = v
L.T x = y
x = Sigma^-1 v
```

This is the purpose of `solve_using_chol_factor(L, v)`.

### Sparse Cholesky and Older Helper Names

Sparse Cholesky is computed with:

```python
factor = cholesky(A, lower=True)
```

from `sksparse.cholmod`. Older versions of this code had a wrapper:

```python
def _sparse_cholesky(A):
    from sksparse.cholmod import cholesky
    return cholesky(A, lower=True)
```

That wrapper no longer exists in the current code. The idea remains the same:
`factor` is the CHOLMOD factorization of the sparse positive definite matrix
`A`.

### CHOLMOD Permutation

CHOLMOD does not generally factorize `A` directly. It uses a fill-reducing
permutation:

```text
L L.T = P A P.T
```

The permutation vector `perm` is interpreted as:

```text
perm[j] = original/old index that appears at new CHOLMOD position j
```

Therefore:

```python
v[perm]
```

puts a vector from original order into CHOLMOD order, and:

```python
out[perm] = y
```

maps a vector from CHOLMOD order back into the original bin order.

For arrays with more than one right hand side the code uses:

```python
out[perm, ...] = y
```

The `...` keeps all remaining axes. For a vector this is equivalent to
`out[perm] = y`; for a matrix it permutes rows and leaves columns untouched.

### `_as_cholmod_lower_factor(factor)`

This is a small adapter for different `scikit-sparse` return conventions.
Some local builds return `(L, perm)` directly; other builds return a CHOLMOD
`Factor` object. The adapter normalizes both cases to:

```text
L, perm
```

where `L` is a sparse lower triangular Cholesky factor and `perm` is the
CHOLMOD permutation.

## Current Sparse Cholesky Operators

The current code uses four operator functions:

```python
apply_cholesky_sparse(factor, v)
apply_cholesky_sparse_T(factor, v)
apply_cholesky_sparse_inverse(factor, v)
apply_cholesky_sparse_inverse_T(factor, v)
```

They interpret the CHOLMOD factor as a Cholesky factor in the original variable
order:

```text
C = P.T L
C C.T = A
```

The operators mean:

```text
apply_cholesky_sparse           -> C v
apply_cholesky_sparse_T         -> C.T v
apply_cholesky_sparse_inverse   -> C^-1 v
apply_cholesky_sparse_inverse_T -> C^-T v
```

These functions replace the older helpers `_cholmod_solve_A` and
`_cholmod_sample_noise`.

### Older `_cholmod_solve_A(factor, b)` Logic

The older helper solved:

```text
A x = b
```

by manually applying the permutation:

```text
rhs = b[perm]
L y = rhs
L.T z = y
x[perm] = z
```

The current equivalent is:

```python
tmp = apply_cholesky_sparse_inverse(factor, b)
x = apply_cholesky_sparse_inverse_T(factor, tmp)
```

because this computes:

```text
x = C^-T C^-1 b = A^-1 b
```

### Older `_cholmod_sample_noise(factor, size)` Logic

The older helper drew:

```text
eps ~ N(0, A^-1)
```

with:

```text
z ~ N(0, I)
eps_permuted = L.T \ z
eps[perm] = eps_permuted
```

The current code writes this as:

```python
z = np.random.normal(size=nbins)
eps = apply_cholesky_sparse_inverse_T(factor, z)
```

## Class `Density`

`Density` is the base class for the density estimation model. It stores the
prior, observation counts, and shared likelihood/posterior machinery.

### `set_prior_Gaussian(...)`

This method configures either covariance mode or precision mode.

If `prior_covariance` is given:

```text
mode = COVARIANCE
prior_covariance = dense array
prior_precision = None
```

If `prior_precision` is given:

```text
mode = PRECISION
prior_precision = dense array or sparse CSC matrix
sparse = True if requested or if the matrix is sparse
prior_covariance = None
```

The prior mean is converted to a one-dimensional array and checked against the
matrix dimension.

### `set_data(nobs)`

Stores the observed counts. The vector length must match `nbins`, if the prior
has already been configured.

### `Lprior`

Lazy Cholesky factor for dense covariance mode:

```text
prior_covariance = Lprior Lprior.T
```

### `random_prior_parameters()`

Draws a prior sample in parameter space.

In sparse precision mode, the current code does:

```python
z = np.random.normal(size=self.nbins)
factor = cholesky(self.prior_precision, lower=True)
f = apply_cholesky_sparse_inverse_T(factor, z)
return self.prior_mean + f
```

This gives a draw with covariance `Q^-1`, where `Q = prior_precision`.

### Likelihood and Posterior

For a parameter vector `f`, the link-specific `field_from_f(f)` produces the
Poisson rate field. The log likelihood is:

```text
sum(nobs * log(field)) - sum(field)
```

The negative log posterior adds the Gaussian prior quadratic form.



Several methods still refer to `apply_prior_choleski_covar*` and
`apply_prior_precision`. In the current file, the covariance-apply methods are
commented out and `_apply_prior_precision` has a leading underscore. This looks
like an unfinished refactor rather than a completed API change.

## `SigmoidMixin` and `PolyaGammaDensity`

`SigmoidMixin` implements the bounded logistic Poisson rate:

```text
lambda(f) = lam * sigmoid(f)
```

It also implements the Pólya-Gamma Gibbs sampler.

### `sample_posterior(...)`

The sampler iterates:

1. sample latent negative counts `k`
2. sample Pólya-Gamma variables `w`
3. sample `f` from the Gaussian conditional posterior

The conditional posterior precision is:

```text
A = Q + diag(w)
```

and the right hand side is:

```text
b = Q mu0 + kappa
```

where:

```text
kappa = 0.5 * (nobs - k)
```

### Dense Covariance Mode

The dense covariance path uses the identity:

```text
A = Sigma^-1 + diag(w)
Sigma = L L.T
M = I + L.T diag(w) L
```

The posterior mean and noise are computed through the dense Cholesky factor of
`M`.

### Sparse Precision Mode

The current sparse precision path is:

```python
A = (self.prior_precision + sps.diags(w, format="csc")).tocsc()

factor = cholesky(A, lower=True)
tmp = apply_cholesky_sparse_inverse(factor, b)
m = apply_cholesky_sparse_inverse_T(factor, tmp)

z = np.random.normal(size=nbins)
eps = apply_cholesky_sparse_inverse_T(factor, z)

f = m + eps
```

This is the current replacement for the older:

```python
factor = _sparse_cholesky(A)
m = _cholmod_solve_A(factor, b)
eps = _cholmod_sample_noise(factor, nbins)
f = m + eps
```

The mathematical purpose is unchanged.

### Dense Precision Mode

The dense precision path forms:

```text
A = prior_precision + diag(w)
```

and uses dense triangular solves from a dense Cholesky factor.

## `SmoothRampMixin` and `RampDensity`

`SmoothRampMixin` uses the softplus-like link:

```text
lambda(f) = softplus(k f) / k
```

The posterior sampler uses a Gaussian-mixture approximation implemented in
`gibbs_softplus_mixture.py`. The mixture is built or loaded lazily through
`load_or_build_mix`.

## `ExponentialMixin` and `ExponentialDensity`

`ExponentialMixin` uses the exponential link:

```text
lambda(f) = exp(f)
```

with numerically safer helper functions from `exp_mix_explink.py`. Its sampler
uses the same Gaussian-mixture framework as the ramp density.

## Concrete Density Classes

The concrete classes combine a link mixin with `Density`:

```python
class PolyaGammaDensity(SigmoidMixin, Density)
class RampDensity(SmoothRampMixin, Density)
class ExponentialDensity(ExponentialMixin, Density)
```

The corresponding 2D classes add `Mixin2D`:

```python
class PolyaGammaDensity2D(Mixin2D, PolyaGammaDensity)
class RampDensity2D(Mixin2D, RampDensity)
class ExponentialDensity2D(Mixin2D, ExponentialDensity)
```

## 2D Mixin

`Mixin2D` stores grid dimensions `n` and `m`, converts between image order and
scan order, and provides helper methods for image-like visualization and random
catalog generation from grid counts.

## Important Variables

```text
prior_mean
    Mean vector of the Gaussian prior.

prior_covariance
    Dense covariance matrix Sigma.

prior_precision
    Dense or sparse precision matrix Q.

sparse
    Whether the precision representation is sparse.

mode
    Either COVARIANCE or PRECISION.

Lprior
    Dense Cholesky factor of prior_covariance.

factor
    CHOLMOD factorization of a sparse precision-like matrix.

perm
    CHOLMOD permutation, interpreted as new-position -> original-index.

nobs
    Observed count vector.

w
    Pólya-Gamma variables in the sigmoid sampler.

kappa
    Linear term from the Pólya-Gamma augmented likelihood.
```

## CHOLMOD References

The code follows the standard `scikit-sparse` / CHOLMOD convention:

```text
L L.T = P A P.T
```

and the permutation vector can be understood through the matrix indexing
`A[p][:, p]`. This is why vectors are permuted with `v[perm]` and mapped back
with `out[perm] = ...`.
