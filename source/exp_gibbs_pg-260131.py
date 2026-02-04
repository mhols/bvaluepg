"""

Der Kernidee besteht aus drei Schritten pro Iteration:
1. **Sample für latente „negative“ Zählungen** k aus einer Poisson-Verteilung
   mit der Rate lam * sigmoid(-f).  Zusammen mit den
   beobachteten Zählungen nobs bilden diese die Gesamtzahl b = nobs + k,
   die die Pólya-Gamma-Verteilung bestimmt.

2. **Sample für Poolya-Gamma-Variablen** w unter Berücksichtigung der aktuellen Werte für f und
   b.

3. **Ziehen des latenten Felde** f aus seiner bedingten Gaußschen
   Verteilung.  Die (posterior() Präzisionsmatrix ist
   Sigma0^{-1} + diag(w), wobei Sigma0 die vorherige Kovarianz ist.
   Der posteriore mu löst
   (Sigma0^{-1} + diag(w)) * m = kappa + Sigma0^{-1} * mu0, wobei
   kappa = 0,5*(nobs - k).  Wir nehmen eine Sample von f, indem wir dieses
   lineare System lösen und dann Gaußsches Rauschen hinzufügen, das mit dem
   Cholesky-Faktor der Präzisionsmatrix skaliert ist.

"""

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

from polyagamma import random_polyagamma
from polyagammadensity import PolyaGammaDensity
import syntheticdata as sd


def sample_polya_gamma(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ziehen aus PG(b, c) unter Verwendung des Pakets „polyagamma“.

    """
    b = np.asarray(b, dtype=int)

    # Ensure positivity (polyagamma requires h >= 1)
    b = np.clip(b, 1, None)

    c = np.asarray(c, dtype=float)

    return random_polyagamma(h=b, z=c, method="saddle")


def gibbs_sampler(
    pgd: PolyaGammaDensity,
    n_iter: int,
    burn_in: int = 0,
    thin: int = 1,
    initial_f: np.ndarray | None = None,
    random_seed: int | None = None,
):
    """
    Parameters
    ----------
    pgd : PolyaGammaDensity
        Eine Instanz der Datenklasse, 
        die vorherige Kovarianz und die beobachteten Zählungen „nobs“ enthält.
    n_iter : int Gesamtzahl der Gibbs-Iterationen (einschließlich Burn-in).
    burn_in : int, optional Anzahl der anfänglichen Iterationen, die verworfen werden sollen.  Der Standardwert ist 0.    
    thin : int, optional
        Verdünnungsintervall für die Speicherung von Proben.  Nur jede „dünne“
        Probe nach dem Einbrennen wird aufbewahrt.  Der Standardwert ist 1.
    initial_f : ndarray, optional
        Anfangszustand für das latente Feld.  Bei „None“ wird der Sampler
        mit dem vorherigen Mittelwert initialisiert.
    random_seed : int, optional
        Falls angegeben, wird der NumPy-Zufallsgenerator für Reproduzierbarkeit festgelegt.

    Returns
    -------
    f_samples : ndarray, shape (n_keep, nbins)
       Array der abgetasteten latenten Felder.  ``n_keep`` entspricht
        ``floor((n_iter - burn_in) / thin)``.  Jede Zeile enthält eine
        Ziehung des latenten Feldes.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    nbins = pgd.nbins
    # Precompute the prior precision matrix and its product with the prior mean
    Sigma0 = pgd.prior_covariance
    mu0 = pgd.prior_mean
    # Precompute the inverse once (symmetric, positive definite)
    Sigma0_inv = np.linalg.inv(Sigma0)
    Sigma0_inv_mu0 = Sigma0_inv @ mu0

    # Initialiesieren
    if initial_f is None:
        f = mu0.copy()
    else:
        f = np.asarray(initial_f, dtype=float).copy()
        if f.shape != mu0.shape:
            raise ValueError("initial_f must have shape matching prior_mean")

    # Samples speichern
    n_keep = max(0, (n_iter - burn_in) // thin)
    f_samples = np.zeros((n_keep, nbins))

    # Gibbs loop
    sample_idx = 0
    for it in range(n_iter):
        # --- Step 1: sample k gegenben f ---
        # Die Rate für die latenten „negativen“ Zählungen ist lam * sigmoid(-f)
        rate_neg = pgd.field_from_f(-f)
        k = np.random.poisson(rate_neg)

        # --- Step 2: sample w gegeben f und counts ---
        b_counts = (pgd.nobs + k).astype(int)  # Gültigen PG-Formparameter sicherstellen
        w = sample_polya_gamma(b_counts, f)

        # --- Step 3: sample f gegeben w, k ---
        # Kappa berechnen
        kappa = 0.5 * (pgd.nobs - k)
        # Posterior-Präzision und Mittelwert
        A = Sigma0_inv + np.diag(w)
        # Rechte Seite: Sigma0_inv * mu0 + kappa
        bvec = Sigma0_inv_mu0 + kappa
        # Löse A m = bvec für m (posteriorer Mittelwert)
        # Cholesky-Faktorisierung für numerische Stabilität.
        chol = np.linalg.cholesky(A)
        # Löse den Mittelwert unter Verwendung der Cholesky-Faktoren.
        # Löse zunächst L y = bvec.
        y = spla.solve_triangular(chol, bvec, lower=True)
        # Löse L^T m = y
        m = spla.solve_triangular(chol.T, y, lower=False)

        # Ziehe eine zufällig aus N(0, A^{-1})
        z = np.random.normal(size=nbins)
        # Löse L^T x = z für x, sodass x ~ N(0, A^{-1})
        eps = spla.solve_triangular(chol.T, z, lower=False)
        f = m + eps

        # Speichern, wenn burn_in abgeschlossen ist und bei Ausdünnungsintervall
        if it >= burn_in and ((it - burn_in) % thin == 0):
            f_samples[sample_idx] = f
            sample_idx += 1

    return f_samples


def main():
    # --- setup ---
    n, m = 20, 20
    lam = 10
    pgd = PolyaGammaDensity(
        prior_mean=np.zeros(n * m),
        prior_covariance=sd.spatial_covariance_gaussian(n, m, rho=3, v2=1),
        lam=lam,
    )

    # Generieren Sie eine zufällige Grundwahrheit und Beobachtungen.
    f_true = pgd.random_prior_parameters()
    events = pgd.random_events_from_f(f_true)
    pgd.set_data(events)

    # Run Gibbs sampler
    n_iter = 200  
    burn_in = 100
    thin = 1
    samples = gibbs_sampler(
        pgd,
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        initial_f=None,
        random_seed=0,
    )

    # Berechnen Sie den posterioren Mittelwert des latenten Feldes.
    f_est = samples.mean(axis=0)
    field_est = pgd.field_from_f(f_est)

    # Wahre und geschätzte Felder visualisieren
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("True field (λσ(f_true))")
    plt.imshow(sd.scanorder_to_image(pgd.field_from_f(f_true), n, m).T)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title("Observed events")
    plt.imshow(sd.scanorder_to_image(events, n, m).T)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title("Gibbs estimate of field")
    plt.imshow(sd.scanorder_to_image(field_est, n, m).T)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    print("Gibbs sampling done.")


if __name__ == "__main__":
    main()


https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html