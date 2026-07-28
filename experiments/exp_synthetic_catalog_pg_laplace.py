"""PG-Experiment auf dem synthetischen Katalog mit Laplace-Vergleich.

Figures:
- Wahrheit, beobachtete Counts, MAP, Laplace-Mittel und PG-Mittel
- Unsicherheit: Laplace-SD, PG-SD und Fehlerbilder gegen lambda_true

Das Skript laedt die Truth-Datei aus data/synthetic, arbeitet auf den
synthetischen Counts und kennt dadurch lambda_true/f_true fuer RMSE-Vergleiche.

in progress:

das Bild sieht mir passt nicht. nochmal pruefen

Bilder asu picture_for_paper vergleichen
dann schick machen und einbauen
warum sieht pg mean rate so komisch aus? checken, ob das an der visualisierung liegt oder an den samples
das muss an den samples liegen
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import covariance_kernels as ck
import polyagammadensity as pgd


# ============================================================
# PARAMETERS TO CHANGE
# ============================================================

TRUTH_FILE = REPO_ROOT / "data" / "synthetic" / "synthetic_catalog_truth.npz"

LAM = 12.0
RHO = 8.0
PRIOR_VARIANCE = 1.0
BOUNDARY = "symmetric"

MAP_NITER = 500

N_LAPLACE_SAMPLES = 200

PG_N_ITER = 800
PG_BURN_IN = 30
PG_THIN = 5

RANDOM_SEED = 0


# Some scikit-sparse versions do not accept lower=True. polyagammadensity.py
# calls cholesky(A, lower=True), so we patch only inside this experiment.
_original_cholmod_cholesky = pgd.cholesky


def cholmod_cholesky_compat(A, *args, **kwargs):
    kwargs.pop("lower", None)
    return _original_cholmod_cholesky(A, *args, **kwargs)


pgd.cholesky = cholmod_cholesky_compat


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def load_truth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(TRUTH_FILE) as data:
        lambda_true = data["lambda_true"].astype(float)
        f_true = data["f_true"].astype(float)
        counts = data["counts"].astype(int)
    return lambda_true, f_true, counts


def make_model(counts: np.ndarray, lambda_true: np.ndarray) -> pgd.PolyaGammaDensity2D:
    ny, nx = counts.shape
    mean_intensity = float(lambda_true.mean())
    prior_probability = float(np.clip(mean_intensity / LAM, 1e-6, 1.0 - 1e-6))
    prior_mean_scalar = float(pgd.inv_sigmoid(prior_probability))
    prior_precision = ck.precision_matern(
        n=ny,
        m=nx,
        rho=RHO,
        v2=PRIOR_VARIANCE,
        boundary=BOUNDARY,
    )

    model = pgd.PolyaGammaDensity2D(
        prior_mean=np.full(ny * nx, prior_mean_scalar),
        prior_precision=prior_precision,
        sparse=True,
        lam=LAM,
        n=ny,
        m=nx,
    )
    model.set_data(counts.ravel(order="C"))
    return model


def fit_map(model: pgd.PolyaGammaDensity2D) -> tuple[np.ndarray, np.ndarray]:
    f0 = model.first_guess_estimator()
    f_map = model.max_logposterior_estimator(
        f0=f0,
        method="TNC",
        niter=MAP_NITER,
    )
    lambda_map = model.field_from_f(f_map)
    return f_map, lambda_map


def laplace_precision_at_map(model: pgd.PolyaGammaDensity2D, f_map: np.ndarray) -> sps.csc_matrix:
    nobs = model.nobs
    d2_lambda = model.second_derivate_field_from_f(f_map)
    d2_log_lambda = model.second_derivative_log_field_from_f(f_map)
    diag_lik = d2_lambda - nobs * d2_log_lambda
    diag_lik = np.maximum(diag_lik, 1e-8)
    return (model.prior_precision + sps.diags(diag_lik, format="csc")).tocsc()


def sample_laplace(model: pgd.PolyaGammaDensity2D, f_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED + 10_000)
    precision = laplace_precision_at_map(model, f_map)
    factor = pgd.cholesky(precision)

    samples_f = np.empty((N_LAPLACE_SAMPLES, model.nbins), dtype=float)
    for sample_id in range(N_LAPLACE_SAMPLES):
        z = rng.normal(size=model.nbins)
        eps = pgd.Density.apply_cholesky_sparse_inverse_T(factor, z)
        samples_f[sample_id] = f_map + eps

    samples_lambda = model.field_from_f(samples_f)
    return samples_f, samples_lambda


def sample_pg(model: pgd.PolyaGammaDensity2D, f_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples_f = []
    for f_sample in model.sample_posterior(
        n_iter=PG_N_ITER,
        burn_in=PG_BURN_IN,
        thin=PG_THIN,
        initial_f=f_map,
        random_seed=RANDOM_SEED + 20_000,
    ):
        samples_f.append(f_sample.copy())

    samples_f = np.asarray(samples_f)
    if samples_f.size == 0:
        raise ValueError("No PG samples retained; adjust PG_N_ITER, PG_BURN_IN, PG_THIN.")
    samples_lambda = model.field_from_f(samples_f)
    return samples_f, samples_lambda


def summarize(samples: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": samples.mean(axis=0),
        "sd": samples.std(axis=0),
        "q025": np.quantile(samples, 0.025, axis=0),
        "q975": np.quantile(samples, 0.975, axis=0),
    }


def image(model: pgd.PolyaGammaDensity2D, values: np.ndarray) -> np.ndarray:
    return model.scanorder_to_image(values)


def plot_comparison(
    model: pgd.PolyaGammaDensity2D,
    lambda_true: np.ndarray,
    counts: np.ndarray,
    lambda_map: np.ndarray,
    laplace: dict[str, np.ndarray],
    pg: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    panels = (
        (axes[0, 0], lambda_true, "lambda_true", 0.0, LAM),
        (axes[0, 1], counts, "observed counts", 0.0, LAM),
        (axes[0, 2], image(model, lambda_map), "MAP rate", 0.0, LAM),
        (axes[0, 3], image(model, laplace["mean"]), "Laplace mean rate", 0.0, LAM),
        (axes[1, 0], image(model, pg["mean"]), "PG mean rate", 0.0, LAM),
        (axes[1, 1], image(model, laplace["sd"]), "Laplace rate SD", None, None),
        (axes[1, 2], image(model, pg["sd"]), "PG rate SD", None, None),
        (axes[1, 3], image(model, pg["mean"]) - lambda_true, "PG mean - truth", None, None),
    )

    for ax, values, title, vmin, vmax in panels:
        im = ax.imshow(values, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    error_panels = (
        (axes[0], image(model, lambda_map) - lambda_true, "MAP - truth"),
        (axes[1], image(model, laplace["mean"]) - lambda_true, "Laplace mean - truth"),
        (axes[2], image(model, pg["mean"]) - lambda_true, "PG mean - truth"),
    )
    max_abs = max(float(np.max(np.abs(values))) for _, values, _ in error_panels)
    for ax, values, title in error_panels:
        im = ax.imshow(values, origin="lower", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.show()


def plot_f_comparison(
    model: pgd.PolyaGammaDensity2D,
    f_true: np.ndarray,
    f_map: np.ndarray,
    laplace_f: dict[str, np.ndarray],
    pg_f: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = (
        (axes[0, 0], f_true, "f_true"),
        (axes[0, 1], image(model, f_map), "MAP f"),
        (axes[0, 2], image(model, laplace_f["mean"]), "Laplace mean f"),
        (axes[1, 0], image(model, pg_f["mean"]), "PG mean f"),
        (axes[1, 1], image(model, laplace_f["sd"]), "Laplace f SD"),
        (axes[1, 2], image(model, pg_f["sd"]), "PG f SD"),
    )

    for ax, values, title in panels:
        im = ax.imshow(values, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.show()


def print_metrics(
    lambda_true: np.ndarray,
    counts: np.ndarray,
    lambda_map: np.ndarray,
    laplace: dict[str, np.ndarray],
    pg: dict[str, np.ndarray],
) -> None:
    lambda_true_flat = lambda_true.ravel(order="C")
    counts_flat = counts.ravel(order="C")

    laplace_covered = (lambda_true_flat >= laplace["q025"]) & (lambda_true_flat <= laplace["q975"])
    pg_covered = (lambda_true_flat >= pg["q025"]) & (lambda_true_flat <= pg["q975"])

    print("Synthetic PG/Laplace experiment")
    print(f"grid: {counts.shape[1]} x {counts.shape[0]}")
    print(f"events: {int(counts.sum())}")
    print(f"lam: {LAM:g}; rho: {RHO:g}; prior variance: {PRIOR_VARIANCE:g}; boundary: {BOUNDARY}")
    print(f"laplace samples: {N_LAPLACE_SAMPLES}")
    print(f"PG retained samples: {(PG_N_ITER - PG_BURN_IN) // PG_THIN}")
    print("")
    print(f"RMSE raw counts vs truth: {rmse(counts_flat, lambda_true_flat):.4f}")
    print(f"RMSE MAP vs truth: {rmse(lambda_map, lambda_true_flat):.4f}")
    print(f"RMSE Laplace mean vs truth: {rmse(laplace['mean'], lambda_true_flat):.4f}")
    print(f"RMSE PG mean vs truth: {rmse(pg['mean'], lambda_true_flat):.4f}")
    print(f"Laplace 95% coverage: {float(laplace_covered.mean()):.3f}")
    print(f"PG 95% coverage: {float(pg_covered.mean()):.3f}")
    print(f"mean Laplace SD: {float(laplace['sd'].mean()):.4f}")
    print(f"mean PG SD: {float(pg['sd'].mean()):.4f}")


def main() -> None:
    lambda_true, f_true, counts = load_truth()

    model = make_model(counts, lambda_true)
    f_map, lambda_map = fit_map(model)

    laplace_f_samples, laplace_lambda = sample_laplace(model, f_map)
    pg_f_samples, pg_lambda = sample_pg(model, f_map)

    laplace = summarize(laplace_lambda)
    pg = summarize(pg_lambda)
    laplace_f = summarize(laplace_f_samples)
    pg_f = summarize(pg_f_samples)

    print_metrics(lambda_true, counts, lambda_map, laplace, pg)
    plot_comparison(model, lambda_true, counts, lambda_map, laplace, pg)
    plot_f_comparison(model, f_true, f_map, laplace_f, pg_f)


if __name__ == "__main__":
    main()
