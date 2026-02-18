import numpy as np
import matplotlib.pyplot as plt
from polyagammadensity import PolyaGammaDensity, sigmoid
import syntheticdata as sd
import polyagamma as pg
import scipy.linalg as la


n, m = 20, 20
lam = 10

pgd = PolyaGammaDensity(
    prior_mean=np.zeros(n*m),
    prior_covariance=sd.spatial_covariance_gaussian(n, m, 3, 1),
    lam=lam
)

# generate data
f_true = pgd.random_prior_parameters()
events = pgd.random_events_from_f(f_true)
pgd.set_data(events)

# init
f = np.zeros(n*m)

# gibbs
n_iter = 20
samples_f = []

for t in range(n_iter):

    # (1) sample k | f   (aux counts)
    # k_i ~ Poisson(lam * sigmoid(-f_i))
    field_neg = lam * sigmoid(-f)
    k = np.random.poisson(field_neg)

    # (2) sample omega | f, n+k  (Polya-Gamma)
    # omega_i ~ PG(b_i, f_i) with b_i = n_i + k_i
    b = events + k

    # omega = np.array([pg.pgdraw(bi, fi) for bi, fi in zip(b, f)])
    # pgdraw laueft nicht richtig, daher:
    # nutze random_polyagamma(h, z)
    omega = pg.random_polyagamma(b.astype(float), f.astype(float))

    # Schnelle Pruefung (Omega sollte positiv sein)
    if t % 10 == 0:
        print(f"t={t:3d}  omega: mean={omega.mean():.3f}, min={omega.min():.3e}, max={omega.max():.3f}")

    # (3) sample f | omega, b  (Gaussian Update)
    # Dies ist der entscheidende Schritt.
    # mean beinhaltet kappa = (n - k)/2 (je nach Formulierung)
    kappa = 0.5 * (events - k)

    L = pgd.Lprior  # Cholesky of prior covariance: Sigma = L L^T
    Sigma_inv = la.cho_solve((L, True), np.eye(n*m))
    Sigma_inv_mu = la.cho_solve((L, True), pgd.prior_mean)

    Q = Sigma_inv + np.diag(omega)
    rhs = kappa.astype(float) + Sigma_inv_mu.astype(float)

    LQ = la.cholesky(Q, lower=True)
    mean_f = la.cho_solve((LQ, True), rhs)

    z = np.random.normal(size=n*m)
    tmp = la.solve_triangular(LQ, z, lower=True)
    f = mean_f + la.solve_triangular(LQ.T, tmp, lower=False)

    # speichere einige samples
    if t > 50 and t % 5 == 0:
        samples_f.append(f.copy())

print("done")


https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html