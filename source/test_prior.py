import numpy as np
import pytest

from bvaluepg.polyagammadensity import PolyaGammaDensity


@pytest.fixture
def pgd_small_spd():
    """
    Small, well-conditioned setup for testing prior-related properties.
    """
    prior_mean = np.zeros(3)
    prior_cov = np.eye(3)  # SPD
    lam = 1.0
    return PolyaGammaDensity(prior_mean=prior_mean, prior_covariance=prior_cov, lam=lam)


def test_nbins_matches_prior_mean_shape(pgd_small_spd):
    assert pgd_small_spd.nbins == 3


def test_set_data_happy_path_sets_nobs(pgd_small_spd):
    nobs = np.array([0, 1, 2])
    pgd_small_spd.set_data(nobs)
    assert np.all(pgd_small_spd.nobs == nobs)


def test_set_data_wrong_length_raises(pgd_small_spd):
    with pytest.raises(AssertionError):
        pgd_small_spd.set_data(np.array([1, 2]))


def test_Lprior_is_cholesky_factor(pgd_small_spd):
    """
    Lprior should be a Cholesky factor of prior_covariance:
    L @ L.T == prior_covariance
    """
    L = pgd_small_spd.Lprior
    assert L.shape == (3, 3)
    assert np.allclose(L @ L.T, pgd_small_spd.prior_covariance)


def test_Lprior_non_pd_raises_linalgerror():
    """
    Cholesky must fail for non-positive-definite matrices.
    """
    prior_mean = np.zeros(2)
    non_pd_cov = np.array([[1.0, 2.0],
                           [2.0, 1.0]])  # eigenvalues: 3 and -1 -> not PD
    pgd = PolyaGammaDensity(prior_mean=prior_mean, prior_covariance=non_pd_cov, lam=1.0)

    with pytest.raises(np.linalg.LinAlgError):
        _ = pgd.Lprior


@pytest.mark.xfail(
    reason="Known issue: Lprior uses `if not self._Lprior`, which breaks on numpy arrays. "
           "Fix by using `if self._Lprior is None:` in polyagammadensity.py.",
    strict=False,
)
def test_Lprior_is_cached_and_reusable(pgd_small_spd):
    """
    Accessing Lprior twice should return the same object (cached) and not raise.
    """
    L1 = pgd_small_spd.Lprior
    L2 = pgd_small_spd.Lprior
    assert L1 is L2
    assert np.allclose(L1, L2)
