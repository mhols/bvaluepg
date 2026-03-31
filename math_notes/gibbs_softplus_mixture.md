---
title: Poisson–Softplus Model with Gaussian Mixture Augmentation
author: Cristina Chávez Chong

---

# Model + Derivation

## 1) Model assumptions

We observe count data \(n_i\) and model:

\[
n_i \mid f_i \sim \text{Poisson}(\lambda_i),
\qquad 
\lambda_i := \text{softplus}(f_i),
\]

with

\[
\text{softplus}(f) = \log(1 + e^f).
\]

Prior on the latent field:

\[
f \sim \mathcal N(\mu_0, C).
\]

---

## 2) Likelihood

The likelihood per pixel:

\[
p(n_i \mid f_i)
\propto
\text{softplus}(f_i)^{n_i} \exp(-\text{softplus}(f_i)).
\]

Log-likelihood:

\[
\ell(f \mid n)
=
\sum_i \left(
n_i \log(\text{softplus}(f_i)) - \text{softplus}(f_i)
\right).
\]



---

## 3) Idea: Gaussian mixture approximation

For each count \(n\), we approximate:

\[
p(n \mid f)
\approx
\sum_{k=1}^{K_n}
w_{n,k}\,\mathcal N(f \mid \mu_{n,k}, \sigma_n^2).
\]

Introduce latent variables:

\[
z_i \in \{1,\dots,K_{n_i}\}.
\]

Then:

\[
p(n_i \mid f_i)
\approx
\sum_k w_{n_i,k}\,\mathcal N(f_i \mid \mu_{n_i,k}, \sigma_{n_i}^2).
\]

---

## 4) Augmented model

\[
p(f, z \mid n)
\propto
p(f)\prod_i
\Big[
w_{n_i,z_i}\,
\mathcal N(f_i \mid \mu_{n_i,z_i}, \sigma_{n_i}^2)
\Big].
\]

---

## 5) Gibbs sampler

We alternate:

\[
z \mid f,n
\quad\text{and}\quad
f \mid z,n.
\]

---

# Step 1: \(z_i \mid f_i, n_i\)

For each pixel:

\[
p(z_i = k \mid f_i, n_i)
\propto
w_{n_i,k}\,\mathcal N(f_i \mid \mu_{n_i,k}, \sigma_{n_i}^2).
\]

Explicitly:

\[
p(z_i = k \mid f_i, n_i)
=
\frac{
w_{n_i,k}\,\exp\!\left(-\frac{(f_i-\mu_{n_i,k})^2}{2\sigma_{n_i}^2}\right)
}{
\sum_j
w_{n_i,j}\,\exp\!\left(-\frac{(f_i-\mu_{n_i,j})^2}{2\sigma_{n_i}^2}\right)
}.
\]

**Interpretation:**  
Each pixel selects a local Gaussian component that approximates its likelihood.

---

# Step 2: \(f \mid z, n\)

Given \(z\), the likelihood becomes Gaussian:

\[
f_i \mid z_i, n_i
\sim
\mathcal N(\mu_i, s_i^2),
\]

with

\[
\mu_i := \mu_{n_i,z_i},
\qquad
s_i^2 := \sigma_{n_i}^2.
\]

Define:

\[
\mu = (\mu_1,\dots,\mu_N),
\qquad
D = \mathrm{diag}(s_i^2).
\]

---

## Posterior

\[
p(f \mid z,n)
\propto
\exp\left(
-\frac12 (f-\mu_0)^\top C^{-1}(f-\mu_0)
-\frac12 (f-\mu)^\top D^{-1}(f-\mu)
\right).
\]

---

## Combine terms

\[
\log p(f \mid z,n)
=
-\frac12 f^\top (C^{-1}+D^{-1}) f
+ f^\top (C^{-1}\mu_0 + D^{-1}\mu)
+ c.
\]

Define:

\[
Q := C^{-1}+D^{-1},
\qquad
h := C^{-1}\mu_0 + D^{-1}\mu.
\]

---

## Posterior distribution

\[
f \mid z,n
\sim
\mathcal N(m, Q^{-1}),
\]

with

\[
m = Q^{-1}h.
\]

