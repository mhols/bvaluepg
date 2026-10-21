import numpy as np
import scipy.sparse as sps
from sksparse.cholmod import CholeskyFactor, cholesky

from polyagammapoisson import polyagammadensity as pgd


def small_precision():
    line = sps.diags([-np.ones(2), 2 * np.ones(3), -np.ones(2)], [-1, 0, 1])
    return (sps.kron(sps.eye(3), line) + sps.kron(line, sps.eye(3))).tocsc()


def test_reused_factor_mean_and_noise_covariance():
    prior = small_precision()
    factor = CholeskyFactor(prior, lower=True)
    b = np.linspace(-1, 2, 9)

    for weights in (np.zeros(9), np.linspace(0.1, 2, 9), np.full(9, 0.5)):
        A = prior + sps.diags(weights, format="csc")
        factor.factorize(A)
        lower = pgd._as_cholmod_lower_factor(factor)
        L, perm = lower
        assert not np.array_equal(perm, np.arange(9))
        np.testing.assert_allclose((L @ L.T).toarray(), A.toarray()[perm][:, perm], atol=1e-12)

        tmp = pgd.Density.apply_cholesky_sparse_inverse(lower, b)
        mean = pgd.Density.apply_cholesky_sparse_inverse_T(lower, tmp)
        np.testing.assert_allclose(mean, np.linalg.solve(A.toarray(), b), atol=1e-12)

        # Transforming the identity gives the exact noise map, without MC error.
        noise_map = pgd.Density.apply_cholesky_sparse_inverse_T(lower, np.eye(9))
        np.testing.assert_allclose(noise_map @ noise_map.T, np.linalg.inv(A.toarray()), atol=1e-12)

        old_factor = cholesky(A, lower=True)
        old_noise_map = pgd.Density.apply_cholesky_sparse_inverse_T(old_factor, np.eye(9))
        # Different orderings can give different noise maps for the same covariance.
        np.testing.assert_allclose(noise_map @ noise_map.T, old_noise_map @ old_noise_map.T, atol=1e-12)
        old_tmp = pgd.Density.apply_cholesky_sparse_inverse(old_factor, b)
        old_mean = pgd.Density.apply_cholesky_sparse_inverse_T(old_factor, old_tmp)
        np.testing.assert_allclose(mean, old_mean, atol=1e-12)


def test_sampler_constructs_one_factor_per_chain(monkeypatch):
    factors = []

    def track_factor(*args, **kwargs):
        factor = CholeskyFactor(*args, **kwargs)
        factors.append(factor)
        return factor

    monkeypatch.setattr(pgd, "CholeskyFactor", track_factor)
    model = pgd.PolyaGammaDensity2D(n=3, m=3, lam=5)
    model.set_prior_Gaussian(prior_mean=np.zeros(9), prior_precision=small_precision(), sparse=True)
    model.set_nobs(np.ones(9, dtype=int))
    samples = list(model.sample_posterior(n_iter=4, random_seed=0, sample_lam=True, return_lam=True))
    assert len(factors) == 1
    assert len(samples) == 4
    for f, lam in samples:
        assert f.shape == (9,)
        assert np.all(np.isfinite(f))
        assert np.isfinite(lam) and lam > 0
