# Quadtree – Area, Correlation, and Patch Modeling

## Context


> If quadtree cells vary in size, how should **rate**, **correlation**, and **prior modeling** be adjusted?

Until now, each cell has been treated as a point-like unit. For adaptive grids with cells of vastly different sizes, this is only an approximation.

---

# 1. Previous Discrete Cell Model

For each cell \(i\):

$$
n_i \sim \operatorname{Poisson}(z_i)
$$

with rate

$$
z_i = L(f_i)
$$

or in your current script:

$$
z_i = \lambda \, \sigma(f_i)
$$

where:

- \(f_i\): latent Gaussian value of the cell
- \(L\): link function
- \(\sigma(x)=\frac{1}{1+e^{-x}}\): sigmoid
- \(\lambda\): global scaling

---

# 2. Latent Prior

Discrete:

$$
f \sim \mathcal N(\mu, \Sigma)
$$

Continuous:

$$
f(x) \sim GP(\mu(x), K(x,x'))
$$

where:

- \(\mu(x)\): Mean function
- \(K(x,x')\): Kernel / Covariance function

Example (RBF):

$$
K(x,x') = v^2 \exp\left(-\frac{\|x-x'\|^2}{2\rho^2}\right)
$$

---

# 3. The actual problem with cells of different sizes

A quadtree cell is not a point mass, but an area:

$$
\Omega_i
$$

with cell area:

$$
|\Omega_i|
$$

Then the expected event rate should depend not only on the center, but on the **entire area**.

---

# 4. Clean area model

Let the intensity in space be:

$$
\lambda(x) = L(f(x))
$$

Then the expected count rate of the cell is:

$$
z_i = \int_{\Omega_i} L(f(x)) \, dx
$$

and thus:

$$
n_i \sim \operatorname{Poisson}\left(\int_{\Omega_i} L(f(x))dx\right)
$$

---

# 5. Practical Center Approximation

If the cell is small enough or the field varies smoothly enough:

$$
f(x) \approx f(c_i)
$$

with cell center \(c_i\), then approximately:

$$
z_i \approx |\Omega_i| \, L(f(c_i))
$$

This leads to:

$$
n_i \sim \operatorname{Poisson}(|\Omega_i|L(f(c_i)))
$$

---

# 6. Relation to Code

## Current

```python
rate_i = lam * sigmoid(f_i)
```

## Better area-corrected version

```python
rate_i = area_i * lam * sigmoid(f_i)
```

Mathematically:

$$
z_i = |\Omega_i| \lambda \sigma(f_i)
$$

---

# 7. Why this is important

Without area, the following implicitly holds:

- large cell: same expected counts as small cell
- small cell: same expected counts as large cell

This automatically implies:

$$
\text{Intensity} \propto \frac{1}{|\Omega_i|}
$$

This is often undesirable.

---

# 8. Correlation also requires area

So far in the discrete model:

$$
\Sigma_{ij} = K(c_i,c_j)
$$

i.e., only center-to-center.

A patch-based covariance would be cleaner:

$$
\operatorname{Cov}(i,j)=
\frac{1}{|\Omega_i||\Omega_j|}
\int_{\Omega_i}\int_{\Omega_j} K(x,x') \, dx \, dx'
$$

Then the following factors are taken into account:

- Size of both cells
- Shape of both cells
- Distance between the surfaces
- Overlap / proximity

---

# 9. When is the center-to-center approximation sufficient?

When the correlation length \(\rho\) is large compared to the cell size:

$$
\rho \gg \text{cell diameter}
$$

Then the field varies little within the cell.

Then:

$$
f(x) \approx f(c_i)
$$

is very good.

---

# 10. When does it become poor?

When:

- large cells exist
- small \(\rho\)
- strong local hotspots
- strongly nonlinear link function

Then the integral model is significantly better.



> With adaptive quadtree patches, both **rate** and **correlation** must be considered on an area-based basis.

---

# 12. Next mathematical step


> Gaussian approximation of the inverse link function using Taylor expansion

This means:

We seek an approximation to bring the complicated model

$$
z_i = \int_{\Omega_i} L(f(x))dx
$$

back into an approximately Gaussian form.

To do this, we consider:

$$
\tilde f_i = L^{-1}(z_i / |\Omega_i|)
$$

and linearize around the prior mean \(\mu\).

---

# 13. Short-term testing

Comparison of two models:

### Model A (current)

$$
z_i = \lambda \sigma(f_i)
$$

### Model B (with area)

$$
z_i = |\Omega_i| \lambda \sigma(f_i)
$$

Compare using identical prior plots.

---

# 14. Gaussian approximation via inverse link function (sigmoid)


## Goal

From the area-based model

$$
z_i = \lambda \int_{\Omega_i} \sigma(f(x))\,dx
$$

an approximate Gaussian cell variable is to be constructed.

---

## Step 1: Area-weighted mean of the transformed intensity

Define

$$
\bar s_i := \frac{z_i}{\lambda |\Omega_i|}
= \frac{1}{|\Omega_i|}\int_{\Omega_i}\sigma(f(x))dx.
$$

This is the mean sigmoid value in the cell.

$$
\bar s_i \in (0,1).
$$

---

## Step 2: Apply the inverse link function

For the sigmoid function, the inverse link function is the logit function:

$$
\sigma^{-1}(u)=\operatorname{logit}(u)=\log\left(\frac{u}{1-u}\right).
$$

Define:

$$
\tilde f_i := \operatorname{logit}(\bar s_i).
$$

Thus:

$$
\tilde f_i = \operatorname{logit}\left(\frac{z_i}{\lambda |\Omega_i|}\right).
$$

---

## Step 3: Taylor approximation of the sigmoid function

Linearization around the mean \(\mu(x)\):

$$
\sigma(f(x)) \approx \sigma(\mu(x)) + \sigma'(\mu(x))(f(x)-\mu(x)).
$$

For the sigmoid function, we have:

$$
\sigma'(u)=\sigma(u)(1-\sigma(u)).
$$

---

## Step 4: Integrate over the cell

Then

$$
\bar s_i \approx a_i + \eta_i
$$

where

$$
a_i := \frac{1}{|\Omega_i|}\int_{\Omega_i}\sigma(\mu(x))dx
$$

and

$$
\eta_i := \frac{1}{|\Omega_i|}\int_{\Omega_i}\sigma'(\mu(x))(f(x)-\mu(x))dx.
$$

Since \(\eta_i\) is a linear functional of a Gaussian process, the following holds approximately:

$$
\eta_i \sim \mathcal N(0,\tau_i^2)
$$

and thus:

$$
\bar s_i \approx \mathcal N(a_i,\tau_i^2).
$$

---

## Step 5: Second Taylor Approximation (Logit)

Now:

$$
\tilde f_i = \operatorname{logit}(\bar s_i)
$$

Linearization around \(a_i\):

$$
\tilde f_i \approx \operatorname{logit}(a_i) + \operatorname{logit}'(a_i)(\bar s_i-a_i).
$$

For the logit function, we have:

$$
\operatorname{logit}'(u)=\frac{1}{u(1-u)}.
$$

Thus:

$$
\tilde f_i \approx \operatorname{logit}(a_i)+\frac{1}{a_i(1-a_i)}(\bar s_i-a_i).
$$

---

## Step 6: Result

Thus, \(\tilde f_i\) is also approximately Gaussian:

$$
\tilde f_i \approx \mathcal N(m_i,v_i)
$$

where

$$
m_i \approx \operatorname{logit}(a_i)
$$

and

$$
v_i \approx \left(\frac{1}{a_i(1-a_i)}\right)^2 \tau_i^2.
$$

---

## Step 7: Special Case of Constant Mean

If \(\mu(x)=\mu_0\):

$$
a_i \approx \sigma(\mu_0)=a.
$$

Then the covariance simplifies to:

$$
\operatorname{Cov}(\tilde f_i,\tilde f_j)
\approx
\frac{1}{|\Omega_i||\Omega_j|}
\int_{\Omega_i}\int_{\Omega_j}K(x,x')dx dx'.
$$

This means:

> The effective Gaussian covariance of the cell values is the area-weighted kernel covariance.

---

## Interpretation

This transforms the complex nonlinearity model back into an approximate Gaussian vector:

$$
\tilde f \approx \mathcal N(\tilde\mu, \tilde\Sigma)
$$

and this is precisely what could be used for quadtree cells.
