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
    # precision_matern() still contains diagnostic plots. Suppress only their
    # interactive display; the production calculation remains unchanged.
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
