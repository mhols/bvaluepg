

import numpy as np
import pytest

from bvaluepg.polyagammadensity import PolyaGammaDensity


@pytest.fixture
def pgd_small_spd():
    """Small, well-conditioned setup for random/prior sampling tests."""
    prior_mean = np.zeros(3)
    prior_cov = np.eye(3)  # SPD
    lam = 2.5
    return PolyaGammaDensity(prior_mean=prior_mean, prior_covariance=prior_cov, lam=lam)


def test_random_prior_prameters_shape_and_type(pgd_small_spd):
    np.random.seed(0)
    f = pgd_small_spd.random_prior_prameters()

    assert isinstance(f, np.ndarray)
    assert f.shape == (pgd_small_spd.nbins,)


def test_random_prior_prameters_reproducible_with_seed(pgd_small_spd):
    np.random.seed(123)
    f1 = pgd_small_spd.random_prior_prameters()

    np.random.seed(123)
    f2 = pgd_small_spd.random_prior_prameters()

    assert np.allclose(f1, f2)


def test_random_prior_field_bounds_and_shape(pgd_small_spd):
    np.random.seed(0)
    pf = pgd_small_spd.random_prior_field()

    assert isinstance(pf, np.ndarray)
    assert pf.shape == (pgd_small_spd.nbins,)
    assert np.all(np.isfinite(pf))
    assert np.all(pf >= 0.0)
    assert np.all(pf <= pgd_small_spd.lam)


def test_random_prior_field_lam_zero_is_all_zero():
    prior_mean = np.zeros(4)
    prior_cov = np.eye(4)
    pgd = PolyaGammaDensity(prior_mean=prior_mean, prior_covariance=prior_cov, lam=0.0)

    np.random.seed(0)
    pf = pgd.random_prior_field()

    assert np.allclose(pf, 0.0)


def test_random_prior_returns_nonnegative_integers_and_correct_length(pgd_small_spd):
    np.random.seed(0)
    counts = pgd_small_spd.random_prior()

    assert isinstance(counts, list)
    assert len(counts) == pgd_small_spd.nbins
    assert all(isinstance(c, (int, np.integer)) for c in counts)
    assert all(c >= 0 for c in counts)


def test_random_prior_reproducible_with_seed(pgd_small_spd):
    np.random.seed(999)
    c1 = pgd_small_spd.random_prior()

    np.random.seed(999)
    c2 = pgd_small_spd.random_prior()

    assert c1 == c2