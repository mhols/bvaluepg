"""
Daten von https://www.usgs.gov/programs/earthquake-hazards

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
from polyagammadensity import PolyaGammaDensity, inv_sigmoid
import syntheticdata as sd

from pathlib import Path

# exp_gibbs_pg.py liegt in repo/source/
HERE = Path(__file__).resolve().parent          # .../repo/source
REPO_ROOT = HERE.parent                         # .../repo
DATA_DIR = REPO_ROOT / "data"

counts_path = DATA_DIR / "eq_counts_30x30.npy"


def sample_polya_gamma(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ziehen aus PG(b, c) unter Verwendung des Pakets „polyagamma“.

    """
    b = np.asarray(b, dtype=int)

    # Ensure positivity (polyagamma requires h >= 1)
    b = np.clip(b, 1, None)

    c = np.asarray(c, dtype=float)

    return random_polyagamma(h=b, z=c, method="saddle")

# # Idee Sigma0_inv_dot(v) mit L =pdg.Lprior
def sigma0_inv_dot(v, L):
    """Berechne Sigma0^{-1} @ v unter Verwendung der Cholesky-Zerlegung L von Sigma0."""
    # Rechne L @ y = v
    y = spla.solve_triangular(L, v, lower=True, trans=False)
    # Rechne L.T @ x = y
    x = spla.solve_triangular(L, y, lower=True, trans=True)
    return x


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
    # Sigma0_inv = np.linalg.inv(Sigma0)
    # L = pgd._Lprior
    L = np.asarray(pgd.Lprior, dtype=float)
    I = np.eye(pgd.nbins)

    X = spla.solve_triangular(L, I, lower=True)         # X = L^{-1}
    Sigma0_inv = spla.solve_triangular(L.T, X, lower=False)  # (L^T)^{-1} @ L^{-1}



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
        y = spla.solve_triangular(chol, bvec, lower=True, trans=False)
        # Löse L^T m = y
        # m = spla.solve_triangular(chol.T, y, lower=False)
        m = spla.solve_triangular(chol, y, lower=True, trans=True)

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

    # brauch ich zum debuggen, damit ich von außerhalb der Funktion auf die Variablen zugreifen kann
    # wandern spaeter ins Returnn der Funktion main() oder in eine neue Funktion, die die Ergebnisse zurückgibt
    # global samples, f_est, field_est, events, f_true

    # --- setup ---
    n, m = 30, 30
    lam = 10
    # pgd = PolyaGammaDensity(
    #     prior_mean=np.zeros(n * m),
    #     prior_covariance=sd.spatial_covariance_gaussian(n, m, rho=3, v2=1),
    #     lam=lam,
    # )

    # --- Load earthquake grid counts (nobs) ---
    counts = np.load(counts_path)                    # shape (30, 30)
    n, m = counts.shape                              # sollte (30,30) sein
    nobs = counts.ravel(order="C").astype(int)       # Länge = 900

    # --- Choose lam so that 0 <= rate <= lam is plausible ---
    # (lam ist die Obergrenze meiner Poisson-Rate pro Zelle)
    lam = max(1, int(np.max(nobs) + 2 * np.sqrt(np.max(nobs))) + 1)

    # --- Build model ---
    pgd = PolyaGammaDensity(
        prior_mean=inv_sigmoid(5 * np.ones(n * m) / lam),
        prior_covariance=sd.spatial_covariance_gaussian(n, m, rho=3, v2=5/lam),
        lam=lam,
    )

    pgd.set_data(nobs)

    # Run Gibbs sampler
    n_iter = 200  
    burn_in = 100
    thin = 10
    samples = gibbs_sampler(
        pgd,
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        initial_f=None,
        random_seed=0,
    )

    # Berechne posterioren Mittelwert des latenten Feldes.
    f_est = samples.mean(axis=0)
    field_est = pgd.field_from_f(f_est)

    

    # Observed counts vs estimated intensity field visualisieren
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Observed counts (nobs)")
    plt.imshow(counts.T)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Estimated rate field (lam * sigmoid(f))")
    plt.imshow(sd.scanorder_to_image(field_est, n, m).T)
    plt.colorbar()

        # --- Plot stored Gibbs samples (rate fields) ---
    for i, sample in enumerate(samples):
        plt.figure(figsize=(4, 4))
        plt.title(f"Gibbs sample {i}")
        sample_field = pgd.field_from_f(sample)
        plt.imshow(sd.scanorder_to_image(sample_field, n, m).T)
        plt.colorbar()

    plt.tight_layout()

    plt.figure()
    ff = np.linspace(0, lam, 100000)[1:-1]
    plt.plot( ff, pgd.density_under_gaussian(ff, pgd.prior_mean[0], pgd.prior_covariance[0,0]))

    plt.figure()
    plt.hist( pgd.nobs )
    plt.show()

    print("Gibbs sampling done.")

    return samples, f_est, field_est, counts, (n, m), lam

if __name__ == "__main__":
    main()