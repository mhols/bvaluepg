from pathlib import Path
import math
import sys

import numpy as np
import pytest
import scipy.sparse as sps
import scipy.sparse.linalg as sparse_linalg


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from exp_validate_precision_matern_5pt import (
    PrecisionCorrelationExperiment,
    alpha_from_rho,
    first_efolding_crossing,
)
import covariance_kernels as ck


def test_alpha_formula():
    rho = 1.7
    gamma = math.exp(-1.0 / rho)
    expected = gamma / (1.0 - gamma) ** 2

    assert alpha_from_rho(rho) == pytest.approx(expected)


def test_operator_construction():
    experiment = PrecisionCorrelationExperiment(ny=3, nx=4)
    L, A, P0 = experiment.build_operators(
        rho=1.2, boundary="symmetric"
    )

    expected_A = sps.eye(12, format="csc") + alpha_from_rho(1.2) * L

    assert np.allclose(A.toarray(), expected_A.toarray())
    assert np.allclose(P0.toarray(), (A.T @ A).toarray())
    assert not np.allclose(P0.toarray(), A.toarray())


def test_first_efolding_crossing():
    distances = np.array([0.0, 1.0, 2.0])
    correlations = np.array([1.0, 0.5, 0.2])
    target = math.exp(-1.0)
    expected = 1.0 + (target - 0.5) / (0.2 - 0.5)

    assert first_efolding_crossing(
        distances, correlations
    ) == pytest.approx(expected)


def test_missing_crossing_returns_nan():
    result = first_efolding_crossing(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 0.8, 0.6]),
    )

    assert np.isnan(result)


def test_precision_matern_sets_center_variance_to_v2(monkeypatch):
    monkeypatch.setattr(ck.plt, "figure", lambda: None)
    monkeypatch.setattr(ck.plt, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(ck.plt, "show", lambda: None)

    ny, nx = 3, 4
    v2 = 2.5
    precision = ck.precision_matern(
        ny,
        nx,
        rho=1.7,
        v2=v2,
        boundary="symmetric",
    )

    center = (ny // 2) * nx + nx // 2
    delta = np.zeros(ny * nx)
    delta[center] = 1.0
    covariance_column = sparse_linalg.spsolve(precision, delta)

    assert covariance_column[center] == pytest.approx(v2)


def test_precision_matern_respects_boundary_argument():
    ny, nx = 4, 5

    precision_zero = ck.precision_matern(
        ny,
        nx,
        rho=1.7,
        v2=2.5,
        boundary="zero",
    )
    precision_symmetric = ck.precision_matern(
        ny,
        nx,
        rho=1.7,
        v2=2.5,
        boundary="symmetric",
    )

    assert (precision_zero - precision_symmetric).nnz > 0
    assert not np.allclose(
        precision_zero.toarray(),
        precision_symmetric.toarray(),
    )


@pytest.mark.parametrize("boundary", ["zero", "symmetric"])
def test_precision_matern_is_symmetric_positive_definite_on_rectangular_grid(
    boundary,
):
    ny, nx = 3, 4
    precision = ck.precision_matern(
        ny,
        nx,
        rho=1.7,
        v2=2.5,
        boundary=boundary,
    )
    precision_dense = precision.toarray()

    assert precision.shape == (ny * nx, ny * nx)
    assert np.allclose(precision_dense, precision_dense.T)
    assert np.linalg.eigvalsh(precision_dense).min() > 0.0


def _show_boundary_precision_plots():
    """Manual visual check for the 5-point precision boundary effect.

    This helper is intentionally not a pytest test. Run this file directly to
    see the plots:

        python tests/test_precision_matern_5pt_validation.py
    """
    ny, nx = 20, 25
    rho = 3.0
    v2 = 1.0
    def compute_results(delta_index):
        delta = np.zeros(ny * nx)
        delta[delta_index] = 1.0

        results = {}
        for boundary in ["zero", "symmetric"]:
            precision = ck.precision_matern(
                ny,
                nx,
                rho=rho,
                v2=v2,
                boundary=boundary,
            )
            precision_impulse = precision @ delta
            covariance_column = sparse_linalg.spsolve(precision, delta)
            results[boundary] = {
                "precision_impulse": precision_impulse.reshape(ny, nx),
                "covariance_column": covariance_column.reshape(ny, nx),
            }
        return results

    def plot_boundary_comparison(results, title, profile_row):
        diff_precision = (
            results["symmetric"]["precision_impulse"]
            - results["zero"]["precision_impulse"]
        )
        diff_covariance = (
            results["symmetric"]["covariance_column"]
            - results["zero"]["covariance_column"]
        )

        fig, axes = ck.plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)
        fig.suptitle(title)

        rows = [
            ("zero", results["zero"]),
            ("symmetric", results["symmetric"]),
            ("symmetric - zero", {
                "precision_impulse": diff_precision,
                "covariance_column": diff_covariance,
            }),
        ]

        for row, (label, data) in enumerate(rows):
            im0 = axes[row, 0].imshow(data["precision_impulse"])
            axes[row, 0].set_title(f"{label}: P @ delta")
            ck.plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

            im1 = axes[row, 1].imshow(data["covariance_column"])
            axes[row, 1].set_title(f"{label}: covariance column")
            ck.plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

            axes[row, 2].plot(data["covariance_column"][profile_row, :])
            axes[row, 2].set_title(f"{label}: row profile")
            axes[row, 2].grid(True)

            for col in range(3):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

    center = (ny // 2) * nx + nx // 2
    near_corner = 1 * nx + 1

    center_results = compute_results(center)
    near_corner_results = compute_results(near_corner)

    plot_boundary_comparison(
        center_results,
        f"center impulse, grid={ny}x{nx}, rho={rho}, v2={v2}",
        profile_row=ny // 2,
    )
    plot_boundary_comparison(
        near_corner_results,
        f"near-corner impulse at row=1, col=1, grid={ny}x{nx}, rho={rho}, v2={v2}",
        profile_row=1,
    )

    ck.plt.show()


if __name__ == "__main__":
    _show_boundary_precision_plots()
