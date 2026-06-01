# Sparse Precision Prior in `polyagammadensity.py`

This is an English working version of `sparse_precision_prior.md`. It describes
the current sparse precision implementation and keeps the older helper-function
names as historical notes for understanding the permutation logic.

## Motivation

Dense covariance matrices become expensive for spatial grids. For an `n * m`
grid, a dense covariance has `(n*m)^2` entries and dense Cholesky factorization
quickly becomes too slow and memory-intensive.

Sparse precision matrices are the natural alternative for many spatial priors:

```text
f ~ N(mu, Q^-1)
```

where `Q` is sparse. A typical example is a grid Laplacian or another local
neighborhood operator.

The important computational goal is to avoid building the dense covariance
`Q^-1`.

## Main Code Changes

### Sparse Precision as Prior

The model can now be initialized with:

```python
prior_precision=precision
sparse=True
```

If `prior_precision` is a SciPy sparse matrix, the code stores it as a CSC
matrix:

```python
self.prior_precision = sps.csc_matrix(prior_precision, dtype=float)
self.mode = Density.PRECISION
self.sparse = True
```

### Precision Mode

In precision mode the prior quadratic form is:

```text
(f - mu).T Q (f - mu)
```

This is cheap for sparse `Q` because it only needs sparse matrix-vector
products.

### Sparse Gibbs Step

In the Pólya-Gamma sampler, the conditional posterior precision is:

```text
A = Q + diag(w)
```

where:

```text
Q = prior_precision
w = Polya-Gamma variables
```

In sparse precision mode this is built as:

```python
A = (self.prior_precision + sps.diags(w, format="csc")).tocsc()
```

and then factorized by CHOLMOD:

```python
factor = cholesky(A, lower=True)
```

## CHOLMOD Installation

The sparse Cholesky functionality uses `sksparse.cholmod`, which depends on
SuiteSparse/CHOLMOD.

### Ubuntu

Typical system dependency:

```bash
sudo apt-get install libsuitesparse-dev
```

Then install Python dependencies in the project environment:

```bash
pip install scikit-sparse
```

### macOS

With Homebrew:

```bash
brew install suite-sparse
pip install scikit-sparse
```

Depending on the local environment, include and library paths may have to be
provided explicitly.

## Sparse Grid Prior Example

A sparse grid precision typically connects neighboring grid cells. For example,
a Laplacian-like precision on a rectangular grid has local stencil structure:

```text
center cell connected to its immediate neighbors
```

This leads to very few nonzero entries per row, so sparse storage and sparse
factorization become useful.

In code the central idea is:

```python
precision = grid_precision_laplacian(...)
pgd = PolyaGammaDensity2D(
    prior_mean=pm,
    prior_precision=precision,
    sparse=True,
    lam=lam,
    n=n,
    m=m,
)
```

## Current Sparse Cholesky Path

The current code no longer uses separate `_sparse_cholesky`,
`_cholmod_solve_A`, or `_cholmod_sample_noise` helper functions. Instead it uses
direct CHOLMOD factorization plus four operator functions:

```python
factor = cholesky(A, lower=True)

tmp = apply_cholesky_sparse_inverse(factor, bvec)
m = apply_cholesky_sparse_inverse_T(factor, tmp)

z = np.random.normal(size=nbins)
eps = apply_cholesky_sparse_inverse_T(factor, z)

f = m + eps
```

This computes:

```text
m = A^-1 bvec
eps ~ N(0, A^-1)
```

without explicitly forming `A^-1`.

## CHOLMOD Permutation

CHOLMOD applies a fill-reducing permutation:

```text
L L.T = P A P.T
```

The code interprets the permutation vector as:

```text
perm[j] = original/old index at new CHOLMOD position j
```

Therefore:

```python
v[perm]
```

puts a vector into CHOLMOD order, and:

```python
out[perm] = y
```

maps a CHOLMOD-ordered vector back to the original bin order.

The current code wraps this in:

```python
apply_cholesky_sparse(factor, v)
apply_cholesky_sparse_T(factor, v)
apply_cholesky_sparse_inverse(factor, v)
apply_cholesky_sparse_inverse_T(factor, v)
```

These functions use:

```text
C = P.T L
C C.T = A
```

so:

```text
apply_cholesky_sparse           -> C v
apply_cholesky_sparse_T         -> C.T v
apply_cholesky_sparse_inverse   -> C^-1 v
apply_cholesky_sparse_inverse_T -> C^-T v
```

## Older Helper Functions

The older implementation used explicit helper names. They are no longer present
in the code, but they are useful for understanding the algebra.

### `_sparse_cholesky(A)`

Old helper:

```python
def _sparse_cholesky(A):
    from sksparse.cholmod import cholesky
    return cholesky(A, lower=True)
```

Current equivalent:

```python
factor = cholesky(A, lower=True)
```

### `_cholmod_solve_A(factor, bvec)`

Old purpose:

```text
solve A m = bvec
```

With `L L.T = P A P.T`, the manual implementation was:

```text
rhs = bvec[perm]
L y = rhs
L.T z = y
m[perm] = z
```

Current equivalent:

```python
tmp = apply_cholesky_sparse_inverse(factor, bvec)
m = apply_cholesky_sparse_inverse_T(factor, tmp)
```

### `_cholmod_sample_noise(factor, nbins)`

Old purpose:

```text
draw eps ~ N(0, A^-1)
```

Manual version:

```text
z ~ N(0, I)
eps_permuted = L.T \ z
eps[perm] = eps_permuted
```

Current equivalent:

```python
z = np.random.normal(size=nbins)
eps = apply_cholesky_sparse_inverse_T(factor, z)
```

## Sparse Pólya-Gamma Path

The current sparse Pólya-Gamma step is:

```python
A = (self.prior_precision + sps.diags(w, format="csc")).tocsc()

factor = cholesky(A, lower=True)
tmp = apply_cholesky_sparse_inverse(factor, bvec)
m = apply_cholesky_sparse_inverse_T(factor, tmp)

z = np.random.normal(size=nbins)
eps = apply_cholesky_sparse_inverse_T(factor, z)

f = m + eps
```

The older notation:

```python
factor = _sparse_cholesky(A)
m = _cholmod_solve_A(factor, bvec)
eps = _cholmod_sample_noise(factor, nbins)
f = m + eps
```

describes the same mathematical computation, but the current code uses the more
general `apply_cholesky_sparse*` operators.

## What Stays the Same

The statistical model is unchanged:

```text
Poisson likelihood
Gaussian prior
Pólya-Gamma augmentation
Gaussian conditional update
```

Only the linear algebra changes. Dense covariance mode uses dense Cholesky
factors. Sparse precision mode uses sparse CHOLMOD factors.

## Expected Performance

Sparse precision matrices are useful when the prior has local structure. The
matrix has only a small number of nonzero entries per row, so sparse storage is
much smaller than dense covariance storage.

The performance gain depends on:

```text
grid size
fill-in after CHOLMOD permutation
number of Gibbs iterations
cost of repeated factorizations of A = Q + diag(w)
```

For large grids, sparse precision should scale much better than dense
covariance, but the posterior precision still changes every Gibbs iteration
because `w` changes.

## Practical Checks

For debugging, small fake sparse matrices are important:

1. build a small symmetric positive definite sparse matrix `A`
2. factorize it with CHOLMOD
3. compare sparse operator results with dense NumPy solves
4. check the permutation convention using `v[perm]` and `out[perm] = ...`
5. verify that sampled noise has approximately covariance `A^-1`

These checks are the safest way to confirm that the sparse and dense paths
produce the same results for small controlled examples.
