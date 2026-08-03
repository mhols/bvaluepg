# BvaluePG

Work on spatial Poisson rate estimation with Pólya-Gamma augmentation.

The current project state is centered around a discretized spatial count model:

```text
counts per bin -> latent field f -> rate lambda = LAM * sigmoid(f)
```

The main practical goal is to build reproducible catalog pipelines, run the PG
sampler on synthetic and real earthquake data, and compare it to simpler
approximations such as MAP/Laplace.

## Current Workflow

### 1. Italy catalog preprocessing

Current linear workflow:

```text
load INGV/HORUS-style catalog
-> filter by magnitude/time
-> project lon/lat to x/y in km
-> NND declustering
-> rotate
-> cut rectangular window
-> assign square bins
-> write counts, events, bins, metadata
```

Main script:

```text
data/preprocess_nnd_rot_cut_bin.py
```

Current default output prefix:

```text
data/preprocess_nnd_rot_cut_bin_Mc_2.5_eta_-4.60_dkm_20
```

Important outputs:

```text
*_events.csv   event-level catalog with declustering, cut and bin columns
*_bins.csv     bin centers, bin ids and counts
*_counts.npz   count grid plus x/y edges
*_meta.json    preprocessing parameters and event-count summaries
```

The older `data/binning_catalog.py` is still useful as a minimal reference for
the rectangular binning logic, but the reproducible current workflow should go
through `preprocess_nnd_rot_cut_bin.py`.

### 2. Synthetic catalogs

Synthetic catalogs are generated as full INGV-like event catalogs, so they can
run through the same preprocessing logic as real data.

Main script:

```text
data/make_synthetic_catalog.py
```

It creates patterns such as `block`, `bars`, and `checkerboard`, samples Poisson
counts, converts counts into event coordinates inside bins, and writes:

```text
data/synthetic/*_events.csv
data/synthetic/*_truth.npz
data/synthetic/*_meta.json
```

The truth file contains `lambda_true`, `f_true`, and `counts`, which makes RMSE
comparisons possible in synthetic experiments.

### 3. Sofiane SPDE-ETAS data

Sofiane's synthetic SPDE-ETAS files are stored in:

```text
data/sofiane_spde_etas/
```

Adapter script:

```text
data/preprocess_sofiane_spde_etas.py
```

This converts Sofiane's numeric files to pipe-separated BvaluePG-compatible
catalogs. The original relative magnitudes are preserved in
`sofiane_mag_relative`; an offset is added to create a `Magnitude` column that
passes our usual magnitude filter.

See also:

```text
data/sofiane_spde_etas/README.md
```

## Core Source Modules

### `source/polyagammadensity.py`

Main inference code.

Important pieces:

- `PolyaGammaDensity2D`: 2D model wrapper for image/grid-shaped fields.
- `field_from_f(f)`: maps latent field values to rates via
  `lambda = LAM * sigmoid(f)`.
- `first_guess_estimator()`: builds an initial latent field estimate from the
  observed counts.
- `max_logposterior_estimator(...)`: MAP estimate for the latent field.
- `sample_posterior(...)`: PG Gibbs sampler.
- `scanorder_to_image(...)` / `image_to_scanorder(...)`: conversion between
  2D grids and row-major vectors.

The current sparse PG runs use a sparse prior precision and therefore enter the
`case PRECISION sparse` branch in the sampler.

### `source/covariance_kernels.py`

Prior precision and covariance helpers.

Most relevant for current experiments:

```text
precision_matern(n, m, rho, v2, boundary="symmetric")
```

This builds a sparse Matern-style precision from a discrete Laplacian. The
current scripts usually use this as the Gaussian prior precision for `f`.

### `source/syntheticdata.py`

Older synthetic pattern and experiment helpers.

Currently useful mainly for reusable patterns such as:

```text
checkerboard(...)
```

The newer full-catalog workflow lives in `data/make_synthetic_catalog.py`.

## Main Experiments

### Italy PG experiment

```text
experiments/exp_italy_preprocess_nnd_rot_cut_pg.py
```

Runs the PG sampler on the current Italy preprocessing output.

Shows:

- observed declustered count grid,
- count histogram,
- PG posterior mean rate,
- PG posterior rate standard deviation,
- a single `f` sample,
- histogram of `f` samples for one selected bin.

This is the current place to continue with map overlays, e.g. by transforming
Italy borders into `x_rot_km/y_rot_km` or by back-transforming PG grid centers to
lon/lat.

### Synthetic PG vs. Laplace

```text
experiments/exp_synthetic_catalog_pg_laplace.py
```

Uses the synthetic truth file from `data/synthetic`, fits MAP, draws Laplace
samples around the MAP, runs the PG sampler, and compares everything against
`lambda_true`.

Main diagnostics:

- raw counts vs. truth,
- MAP rate vs. truth,
- Laplace posterior mean/sd,
- PG posterior mean/sd,
- RMSE values,
- coverage diagnostics,
- `f`-scale comparison.

### Sofiane SPDE-ETAS PG experiment

```text
experiments/exp_sofiane_spde_etas_pg.py
```

Runs our PG model on Sofiane's synthetic catalogs after preprocessing.

Important comparison:

```text
all events vs. known background events
```

Sofiane's `sofiane_parent_id == 0` identifies background events. This lets us
check how much triggering inflates the estimated background rate when all events
are used instead of only known background events.

## Useful Commands

From the repository root:

```bash
/Users/toni/Documents/CodeProjects/VirtualEnvs/PolyaGamma/bin/python data/preprocess_nnd_rot_cut_bin.py
/Users/toni/Documents/CodeProjects/VirtualEnvs/PolyaGamma/bin/python data/make_synthetic_catalog.py
/Users/toni/Documents/CodeProjects/VirtualEnvs/PolyaGamma/bin/python data/preprocess_sofiane_spde_etas.py
```

Experiments:

```bash
/Users/toni/Documents/CodeProjects/VirtualEnvs/PolyaGamma/bin/python experiments/exp_italy_preprocess_nnd_rot_cut_pg.py
/Users/toni/Documents/CodeProjects/VirtualEnvs/PolyaGamma/bin/python experiments/exp_synthetic_catalog_pg_laplace.py
/Users/toni/Documents/CodeProjects/VirtualEnvs/PolyaGamma/bin/python experiments/exp_sofiane_spde_etas_pg.py
```

For non-interactive checks:

```bash
MPLBACKEND=Agg PYTHONPYCACHEPREFIX=/private/tmp/bvaluepg_pycache /Users/toni/Documents/CodeProjects/VirtualEnvs/PolyaGamma/bin/python experiments/exp_italy_preprocess_nnd_rot_cut_pg.py
```

## Current vs. Legacy

Current main workflow:

```text
data/preprocess_nnd_rot_cut_bin.py
-> experiments/exp_italy_preprocess_nnd_rot_cut_pg.py
```

Older or diagnostic scripts remain useful for comparison, but should not be
treated as the main pipeline unless explicitly needed:

```text
data/binning_catalog.py
data/preprocess_italy_nnd_rotate_cut_grid.py
experiments/exp_italy_nnd_pipeline_pg.py
```

## Notes

- Grid vectors use row-major scan order: `counts.ravel(order="C")`.
- Internal 2D arrays are shaped as `(ny, nx)`.
- The current rectangular grid keeps zero-count cells.
- For real Italy data, `lambda_true` is unknown, so RMSE against truth is only
  meaningful for synthetic experiments.
