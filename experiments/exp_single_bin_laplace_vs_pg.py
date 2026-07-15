from __future__ import annotations

"""
Single-bin comparison of the exact posterior?, Laplace approximation, and
Pólya-Gamma Gibbs samples for the sigmoid Poisson model.

1-dim
"""

from pathlib import Path
import contextlib
import io
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "source"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import matplotlib.pyplot as plt
import numpy as np

from polyagammadensity import PolyaGammaDensity, sigmoid


LAM = 20.0
PRIOR_MEAN = -2.0
PRIOR_VARIANCE = 4.0
COUNTS = [0, 5, 10, 100]

F_MIN = -9.0
F_MAX = 7.0
N_GRID = 3000

N_ITER = 12_000
BURN_IN = 2_000
THIN = 5
RANDOM_SEED = 123

### versuchen wirs mal gleich mit ner normalisierten Dichte
def normalize_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    area = np.trapezoid(y, x)
    if not np.isfinite(area) or area <= 0:
        return y
    return y / area


def normal_density(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    sd = np.sqrt(variance)
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (np.sqrt(2 * np.pi) * sd)


def make_model(nobs: int) -> PolyaGammaDensity:
    model = PolyaGammaDensity(
        prior_mean=np.array([PRIOR_MEAN]),
        prior_covariance=np.array([[PRIOR_VARIANCE]]),
        lam=LAM,
    )
    model.set_data(np.array([nobs]))
    return model


def draw_pg_samples(model: PolyaGammaDensity, initial_f: float, seed: int) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        samples = [
            sample.copy()
            for sample in model.sample_posterior(
                n_iter=N_ITER,
                burn_in=BURN_IN,
                thin=THIN,
                initial_f=np.array([initial_f]),
                random_seed=seed,
            )
        ]
    return np.asarray(samples).ravel()


def laplace_approximation(model: PolyaGammaDensity, nobs: int) -> tuple[float, float]:
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.laplace_approximation_one_dimension(
            PRIOR_MEAN,
            PRIOR_VARIANCE,
            nobs,
        )


def plot_f_space() -> None:
    f_grid = np.linspace(F_MIN, F_MAX, N_GRID)
    fig, axes = plt.subplots(1, len(COUNTS), figsize=(14, 4.2), sharey=True)

    for panel, (ax, nobs) in enumerate(zip(axes, COUNTS)):
        model = make_model(nobs)

        exact = model.posterior_f_one_dimension(f_grid, PRIOR_MEAN, PRIOR_VARIANCE, nobs)
        exact = normalize_density(f_grid, exact)

        laplace_mean, laplace_variance = laplace_approximation(model, nobs)
        laplace = normal_density(f_grid, laplace_mean, laplace_variance)

        samples = draw_pg_samples(model, laplace_mean, RANDOM_SEED + panel)

        ax.hist(
            samples,
            bins=70,
            density=True,
            color="C0",
            alpha=0.25,
            label="PG samples",
        )
        ax.plot(f_grid, exact, color="black", lw=2.5, label="exact posterior")
        ax.plot(f_grid, laplace, color="C3", lw=2.2, ls="--", label="Laplace")
        ax.axvline(laplace_mean, color="C3", lw=1.2, alpha=0.7)
        ax.set_title(f"observed count n = {nobs}")
        ax.set_xlabel("latent field f")
        ax.grid(alpha=0.25)

        if panel == 0:
            ax.set_ylabel("density")
            ax.legend()

    fig.suptitle(
        f"Single-bin posterior: lambda={LAM:g}, prior N({PRIOR_MEAN:g}, {PRIOR_VARIANCE:g})",
        y=1.03,
    )
    fig.tight_layout()


def plot_rate_space() -> None:
    f_grid = np.linspace(F_MIN, F_MAX, N_GRID)
    rate_grid = LAM * sigmoid(f_grid)

    fig, axes = plt.subplots(1, len(COUNTS), figsize=(14, 4.2), sharey=True)

    for panel, (ax, nobs) in enumerate(zip(axes, COUNTS)):
        model = make_model(nobs)

        exact_f = model.posterior_f_one_dimension(f_grid, PRIOR_MEAN, PRIOR_VARIANCE, nobs)
        exact_f = normalize_density(f_grid, exact_f)

        # Change of variables: p(rate) = p(f) / |d rate / d f|.
        rate_derivative = LAM * sigmoid(f_grid) * sigmoid(-f_grid)
        exact_rate = normalize_density(rate_grid, exact_f / np.maximum(rate_derivative, 1e-12))

        laplace_mean, laplace_variance = laplace_approximation(model, nobs)
        laplace_f = normal_density(f_grid, laplace_mean, laplace_variance)
        laplace_rate = normalize_density(rate_grid, laplace_f / np.maximum(rate_derivative, 1e-12))

        samples_f = draw_pg_samples(model, laplace_mean, RANDOM_SEED + 100 + panel)
        samples_rate = LAM * sigmoid(samples_f)

        ax.hist(
            samples_rate,
            bins=70,
            density=True,
            color="C0",
            alpha=0.25,
            label="PG samples",
        )
        ax.plot(rate_grid, exact_rate, color="black", lw=2.5, label="exact posterior")
        ax.plot(rate_grid, laplace_rate, color="C3", lw=2.2, ls="--", label="Laplace")
        ax.axvline(LAM * sigmoid(laplace_mean), color="C3", lw=1.2, alpha=0.7)
        ax.set_title(f"observed count n = {nobs}")
        ax.set_xlabel("rate lambda * sigmoid(f)")
        ax.grid(alpha=0.25)

        if panel == 0:
            ax.set_ylabel("density")
            ax.legend()

    fig.suptitle("Same comparison on the Poisson-rate scale", y=1.03)
    fig.tight_layout()


def plot_sample_counts() -> None:
    fig, axes = plt.subplots(1, len(COUNTS), figsize=(14, 4.2), sharey=True)

    for panel, (ax, nobs) in enumerate(zip(axes, COUNTS)):
        model = make_model(nobs)
        laplace_mean, _ = laplace_approximation(model, nobs)
        samples = draw_pg_samples(model, laplace_mean, RANDOM_SEED + 200 + panel)

        ax.hist(
            samples,
            bins=70,
            density=False,
            color="C0",
            alpha=0.55,
            edgecolor="white",
        )
        ax.set_title(f"observed count n = {nobs}")
        ax.set_xlabel("latent field f")
        ax.grid(alpha=0.25)

        if panel == 0:
            ax.set_ylabel("number of PG samples")

    fig.suptitle("Same PG samples as raw histogram counts", y=1.03)
    fig.tight_layout()


def main() -> None:
    plot_f_space()
    plot_rate_space()
    plot_sample_counts()
    plt.show()


if __name__ == "__main__":
    main()


# Ueberlegung
# schwarze Kurve ist hier nur deshalb "exakt", weil wir nur ein einziges
# Bin anschauen. In 1D kann man die Dichte einfach auf einem feinen f-Gitter
# auswerten:
#
#     Posterior = Likelihood * Prior
#
# und danach numerisch normieren. Im echten raeumlichen Modell geht das nicht,
# weil f dann tausende gekoppelte Komponenten hat. Genau dafuer brauchen wir
# den Polya-Gamma-Sampler.
#
# Was man sieht:
# - Die PG-Samples liegen gut auf der schwarzen Kurve. Der Sampler trifft also
#   in diesem Test die richtige Zielverteilung.
# - Laplace ist in dieser Einstellung auf der f-Skala gar nicht so schlecht.
# - Auf der Rate-Skala sieht man besser, dass die Verteilung schief und durch
#   lambda * sigmoid(f) nach oben begrenzt ist.
# - Der Fall n=0 ist am spannendsten: keine Events druecken die Rate stark nach
#   unten, und dort sieht man die Asymmetrie am deutlichsten.
# - Die zusaetzliche Count-Histogramm-Figur zeigt nur die rohen Sample-Anzahlen.
#   Das ist hilfreich zum Bauchgefuehl, aber fuer den Vergleich mit Dichtekurven
#   ist die normierte Dichteansicht die richtige Skala.
#
# Fuer eine staerkere Paper-Figur sollten wir Parameter waehlen, bei denen
# Laplace sichtbarer danebenliegt.
