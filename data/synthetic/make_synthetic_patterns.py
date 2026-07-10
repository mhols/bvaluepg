"""Simple synthetic intensity patterns for BvaluePG.

Two synthetic images, each measuring 64 × 64, are created

1. A block of high intensity in the top left-hand corner.
2. Three horizontal bars, with the middle bar being the brightest.

"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import polyagammadensity as pgd
import covariance_kernels as ck


# --- Parameters -----------------------------------------------------------

NY = 64
NX = 64
SEED = 20260707

# Upper limit of the sigmoid model for the expected number per cell.
LAM_MAX = 12.0

# Pattern 1: upper-left block
PATTERN_1_BACKGROUND = 0.8
PATTERN_1_HOT = 8.0
PATTERN_1_ROWS = slice(0, 26)
PATTERN_1_COLS = slice(0, 26)

# Pattern 2: three horizontal bars, middle bar stronger
PATTERN_2_BACKGROUND = 0.6
PATTERN_2_BAR_TOP_ADD = 4.0
PATTERN_2_BAR_MIDDLE_ADD = 8.0
PATTERN_2_BAR_BOTTOM_ADD = 4.0
PATTERN_2_BAR_WIDTH = 5
PATTERN_2_COL_START = 14
PATTERN_2_COL_END = 50
PATTERN_2_BAR_CENTERS_Y = (24, 32, 40)

# Sparse Polyagamma experiment parameters
PRIOR_RHO = 8.0
PRIOR_V2 = 1.0
PRIOR_BOUNDARY = "symmetric"
N_ITER = 30
BURN_IN = 10
THIN = 2

# --- Pattern construction -------------------------------------------------

def field_to_sigmoid_latent(lambda_true: np.ndarray, lam_max: float) -> np.ndarray:
    """Ordne lambda_i zu den latenten Werten f_i, (wobei lambda_i = lam_max sigmoid(f_i) gilt)"""
    probability = np.asarray(lambda_true, dtype=float) / float(lam_max)
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return pgd.inv_sigmoid(probability)


def make_pattern_1_block() -> np.ndarray:
    """Block with high intensity in the top left corner"""
    lambda_true = PATTERN_1_BACKGROUND * np.ones((NY, NX), dtype=float)
    lambda_true[PATTERN_1_ROWS, PATTERN_1_COLS] = PATTERN_1_HOT
    return lambda_true


def make_pattern_2_bars() -> np.ndarray:
    """Three horizontal bars, with the middle bar being the brightest"""
    lambda_true = PATTERN_2_BACKGROUND * np.ones((NY, NX), dtype=float)
    cols = slice(PATTERN_2_COL_START, PATTERN_2_COL_END)
    additions = (
        PATTERN_2_BAR_TOP_ADD,
        PATTERN_2_BAR_MIDDLE_ADD,
        PATTERN_2_BAR_BOTTOM_ADD,
    )

    for center, addition in zip(PATTERN_2_BAR_CENTERS_Y, additions):
        half_width = PATTERN_2_BAR_WIDTH // 2
        rows = slice(center - half_width, center + half_width + 1)
        lambda_true[rows, cols] += addition

    return lambda_true


def sample_counts(lambda_true: np.ndarray, seed: int) -> np.ndarray:
    """Draw independent Poisson counts from the image"""
    rng = np.random.default_rng(seed)
    return rng.poisson(lambda_true).astype(int)


def validate_pattern(name: str, lambda_true: np.ndarray, counts: np.ndarray, f_true: np.ndarray) -> None:
    """ Shape Fehler warum?
    Asserts bauen
    """
    assert lambda_true.shape == (NY, NX), f"{name}: wrong lambda_true shape"
    assert counts.shape == (NY, NX), f"{name}: wrong counts shape"
    assert f_true.shape == (NY, NX), f"{name}: wrong f_true shape"
    assert np.issubdtype(counts.dtype, np.integer), f"{name}: counts must be integer"
    assert np.all(counts >= 0), f"{name}: counts must be non-negative"
    assert np.all(lambda_true > 0), f"{name}: lambda_true must be positive"
    assert np.all(lambda_true < LAM_MAX), f"{name}: lambda_true must be below LAM_MAX"
    assert np.isfinite(f_true).all(), f"{name}: f_true contains non-finite values"
    assert lambda_true.ravel(order="C").shape == (NY * NX,), f"{name}: bad scan-order length"


# --- Sparse Polya-Gamma experiment ---------------------------------------

def make_sparse_polya_gamma_model(counts: np.ndarray) -> pgd.PolyaGammaDensity2D:
    """sparse-prior PolyaGammaDensity2D model"""
    baseline = float(np.mean(counts))
    baseline = np.clip(baseline, 1e-6, LAM_MAX - 1e-6)
    prior_mean = pgd.inv_sigmoid((baseline / LAM_MAX) * np.ones(NY * NX))
    prior_precision = ck.precision_matern(
        n=NY,
        m=NX,
        rho=PRIOR_RHO,
        v2=PRIOR_V2,
        boundary=PRIOR_BOUNDARY,
    )

    model = pgd.PolyaGammaDensity2D(
        prior_mean=prior_mean,
        prior_precision=prior_precision,
        sparse=True,
        n=NY,
        m=NX,
        lam=LAM_MAX,
    )
    model.set_data(counts)
    return model


def sample_sparse_polya_gamma(
    name: str,
    counts: np.ndarray,
    initial_f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run sparse Polya-Gamma posterior sample and return mean/variance rates."""
    model = make_sparse_polya_gamma_model(counts)
    samples = []

    print(
        f"{name}: sparse PG sampling "
        f"(n_iter={N_ITER}, burn_in={BURN_IN}, thin={THIN}, "
        f"rho={PRIOR_RHO}, v2={PRIOR_V2}, boundary={PRIOR_BOUNDARY})"
    )
    for f_sample in model.sample_posterior(
        n_iter=N_ITER,
        burn_in=BURN_IN,
        thin=THIN,
        initial_f=initial_f.ravel(order="C"),
        random_seed=SEED,
    ):
        samples.append(f_sample)

    if not samples:
        raise RuntimeError(f"{name}: no posterior samples were returned")

    field_samples = np.array([model.field_from_f(sample) for sample in samples])
    field_mean = np.mean(field_samples, axis=0)
    field_var = np.var(field_samples, axis=0)

    return (
        model.scanorder_to_image(field_mean),
        model.scanorder_to_image(field_var),
    )


# --- Plotting -------------------------------------------------------------

def plot_patterns(patterns: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    """Show lambda_true, counts, and f_true for each pattern."""
    nrows = len(patterns)
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(11, 4 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.array([axes])

    for row, (name, lambda_true, counts, f_true) in enumerate(patterns):
        panels = (
            ("true expected counts lambda", lambda_true),
            ("observed counts", counts),
            ("latent f_true", f_true),
        )
        for col, (title, image) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(image, origin="upper", cmap="viridis")
            ax.set_title(f"{name}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


def plot_sparse_results(
    results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Show true intensity, counts, posterior mean, and posterior variance."""
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(14, 4 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.array([axes])

    for row, (name, lambda_true, counts, posterior_mean, posterior_var) in enumerate(results):
        panels = (
            ("true expected counts lambda", lambda_true),
            ("observed counts", counts),
            ("posterior mean lambda", posterior_mean),
            ("posterior variance lambda", posterior_var),
        )
        for col, (title, image) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(image, origin="upper", cmap="viridis")
            ax.set_title(f"{name}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


def main() -> None:
    pattern_specs = (
        ("pattern_1_upper_left_block", make_pattern_1_block(), SEED),
        ("pattern_2_three_bars", make_pattern_2_bars(), SEED),
    )

    patterns = []
    sparse_results = []
    for name, lambda_true, seed in pattern_specs:
        counts = sample_counts(lambda_true, seed=seed)
        f_true = field_to_sigmoid_latent(lambda_true, lam_max=LAM_MAX)
        validate_pattern(name, lambda_true, counts, f_true)
        print(
            f"{name}: total counts={int(counts.sum())}, "
            f"lambda range=({lambda_true.min():.2f}, {lambda_true.max():.2f})"
        )

        patterns.append((name, lambda_true, counts, f_true))

        posterior_mean, posterior_var = sample_sparse_polya_gamma(name, counts, initial_f=f_true)
        sparse_results.append((name, lambda_true, counts, posterior_mean, posterior_var))

    plot_patterns(patterns)
    plot_sparse_results(sparse_results)


if __name__ == "__main__":
    main()


'''
name als string eingebaut, damit plot title macht. wird in plot_patterns und plot_sparse_results verwendet, festgelegt in pattern_specs
todo restl comments in englisch
'''