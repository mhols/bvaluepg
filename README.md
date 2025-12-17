# bvaluepg
work on bvalue estimation with polya gamma

# Bayesian Simulation & Inference – Poisson Case

*(Structured notes, originally from handwritten draft on Dec 17)*

---

## 1. Goal & Context

Simulation and inference for a **Poisson model** applied to discrete or spatial data  
(e.g. pixels, intensities, count processes).

Main focus:
- Density simulation
- Latent parameter field \( f \)
- Bayesian inference instead of purely frequentist estimation

---

## 2. Model Assumptions

### 2.1 Observation Model (Poisson)

- Poisson case (counts / intensities)
- Log-link for the intensity

\[
\log N = a - b M
\]

- Initial focus on **the parameter \( a \) only**
- Simplified setting for first simulations

---

## 3. Data & Structure

### 3.1 Handling the Data

- Data should **not be fixed**, but **randomly generated**
- The model class should **produce random output**
- Use of a **checkerboard pattern** as synthetic structure
- Resolution: approximately **1000 pixels**

---

## 4. Prior Model

### 4.1 Gaussian Prior

- Prior distribution for the latent field \( f \)
- Multivariate normal distribution

\[
f \sim \mathcal N(\mu, \Sigma)
\]

**Variants:**
- White noise prior
- Structured covariance via **Cholesky decomposition**

**Prior parameters:**
- Mean
- Variance / covariance

➡️ These define the **underlying latent \( f \)-values**.

---

## 5. Sampling & Inference

### 5.1 Sampling Strategy

- Sampling of the underlying parameters
- Optional grouping of pixels into **chunks / blocks**
- **Gibbs sampling** as the main inference method

### 5.2 Pólya–Gamma Augmentation

- Use of **Pólya–Gamma augmentation** to transform
  - non-Gaussian likelihoods
  - into conditionally Gaussian updates
- Reduction to **1D Gaussian updates** per component

---

## 6. Estimators

### 6.1 Point Estimators

- **MLE (Maximum Likelihood Estimator)**
  - data-driven, no prior
- **MAP (Maximum A Posteriori)**
  - combines likelihood and prior
  - target estimator for the latent field \( f \)

➡️ Comparison of MLE vs. MAP for analysis and validation

---

## 7. Code Structure

### 7.1 Architecture

A **single working class** implementing the entire model:

#### Responsibilities of the Class

- Data generation
- Prior definition  
  - mean  
  - covariance
- Random sampling from the prior
- Gibbs sampling procedure
- Computation of MLE and MAP estimates

### 7.2 Suggested Structure

```text
ModelClass
├── constructor
│   └── initialize data and prior
├── destructor
│   └── cleanup / logging
├── generate_data()
├── sample_prior()
├── gibbs_step()
├── estimate_mle()
└── estimate_map()
```
## 8. Summary

- Clear Bayesian workflow:
  1. Define prior
  2. Simulate data
  3. Specify Poisson likelihood
  4. Apply Pólya–Gamma augmentation
  5. Run Gibbs sampling
  6. Compare MAP and MLE
- Emphasis on **simulation and understanding**, not only estimation
