"""
Gibbs-Sampling mit Pólya-Gamma-Augmentation auf einem adaptiv erzeugten QuadTree-Grid.

Diese Variante orientiert sich an exp_gibbs_pg_quadtree.py,
erzeugt den QuadTree aber direkt aus den Rohdaten über die Klasse QuadTree,
statt ein zuvor gespeichertes Grid aus CSV/NPY zu laden.

Aktueller Stand:
- Rohdaten: Erdbebenpunkte aus GeoJSON oder CSV
- Grid: wird on-the-fly per QuadTree-Klasse erzeugt
- Beobachtungen: Counts pro QuadTree-Zelle
- Prior: Gaußsche Kovarianz über die Zellzentren
- Visualisierung: Rechtecke statt imshow



Waere es statistisch sauberer, die Zellfläche als Exposure einzubauen oder, also
    rate_i = area_i * lam * sigmoid(f_i).?
"""

from pathlib import Path

# from mini_variable_explorer import explore

import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import PowerNorm

from polyagamma import random_polyagamma
from polyagammadensity import PolyaGammaDensity, inv_sigmoid
from quadtree import QuadTree


HERE = Path(__file__).resolve().parent          # .../bvaluepg/source
REPO_ROOT = HERE.parent                         # .../bvaluepg
DATA_DIR = REPO_ROOT / "data"

JSON_FILE = DATA_DIR / "earthquakes_3point5_cl_2010-2020.json"
CSV_FILE = DATA_DIR / "earthquakes_3point5_cl_2010-2020.csv"

# QuadTree-Parameter
NMAX = 25
MAX_DEPTH = 12


def load_earthquake_data() -> gpd.GeoDataFrame:
    """Lade Erdbebendaten aus GeoJSON oder CSV als GeoDataFrame."""
    if JSON_FILE.exists():
        gdf = gpd.read_file(JSON_FILE)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        return gdf

    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        lat_col = "latitude"
        lon_col = "longitude"
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )
        return gdf

    raise FileNotFoundError(
        "Keine Erdbebendaten gefunden. Erwartet wird eine der Dateien:\n"
        f"- {JSON_FILE}\n"
        f"- {CSV_FILE}"
    )


def sample_polya_gamma(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ziehe Samples aus PG(b, c) mit dem polyagamma-Paket."""
    b = np.asarray(b, dtype=int)
    b = np.clip(b, 1, None)
    c = np.asarray(c, dtype=float)
    return random_polyagamma(h=b, z=c, method="saddle")



def gaussian_covariance_from_coords(
    x: np.ndarray,
    y: np.ndarray,
    rho: float,
    v2: float,
    jitter: float = 1e-8,
) -> np.ndarray:
    """
    Baue eine Gauß-Kovarianzmatrix aus 2D-Koordinaten auf.

    C_ij = v2 * exp(-||s_i - s_j||^2 / (2 * rho^2))
    """
    coords = np.column_stack([x, y])
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    cov = v2 * np.exp(-dist2 / (2.0 * rho**2))
    cov += jitter * np.eye(len(x))
    return cov



def gibbs_sampler(
    pgd: PolyaGammaDensity,
    n_iter: int,
    burn_in: int = 0,
    thin: int = 1,
    initial_f: np.ndarray | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """Gibbs-Sampler für das latente Feld auf dem QuadTree-Grid."""
    if random_seed is not None:
        np.random.seed(random_seed)

    nbins = pgd.nbins
    mu0 = pgd.prior_mean
    L = np.asarray(pgd.Lprior, dtype=float)
    I = np.eye(nbins)

    # Sigma0^{-1} aus der Cholesky-Zerlegung der Prior-Kovarianz
    X = spla.solve_triangular(L, I, lower=True)
    Sigma0_inv = spla.solve_triangular(L.T, X, lower=False)
    Sigma0_inv_mu0 = Sigma0_inv @ mu0

    if initial_f is None:
        f = mu0.copy()
    else:
        f = np.asarray(initial_f, dtype=float).copy()
        if f.shape != mu0.shape:
            raise ValueError("initial_f must have shape matching prior_mean")

    n_keep = max(0, (n_iter - burn_in) // thin)
    f_samples = np.zeros((n_keep, nbins))

    sample_idx = 0
    for it in range(n_iter):
        # 1) latente negative Counts
        rate_neg = pgd.field_from_f(-f)
        k = np.random.poisson(rate_neg)

        # 2) Pólya-Gamma-Variablen
        b_counts = (pgd.nobs + k).astype(int)
        w = sample_polya_gamma(b_counts, f)

        # 3) f | w, k
        kappa = 0.5 * (pgd.nobs - k)
        A = Sigma0_inv + np.diag(w)
        bvec = Sigma0_inv_mu0 + kappa

        chol = np.linalg.cholesky(A)
        ytmp = spla.solve_triangular(chol, bvec, lower=True, trans=False)
        m = spla.solve_triangular(chol, ytmp, lower=True, trans=True)

        z = np.random.normal(size=nbins)
        eps = spla.solve_triangular(chol.T, z, lower=False)
        f = m + eps

        if it >= burn_in and ((it - burn_in) % thin == 0):
            f_samples[sample_idx] = f
            sample_idx += 1

    return f_samples



def plot_quadtree_values(
    grid_df: pd.DataFrame,
    values: np.ndarray,
    title: str,
    cmap: str = "viridis",
    power_gamma: float | None = None,
) -> None:
    """Plotte Werte auf dem adaptiven Grid als gefärbte Rechtecke."""
    values = np.asarray(values, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    if power_gamma is not None:
        norm = PowerNorm(gamma=power_gamma, vmin=np.min(values), vmax=np.max(values))
    else:
        norm = plt.Normalize(vmin=np.min(values), vmax=np.max(values))

    cmap_obj = plt.cm.get_cmap(cmap)

    for i, row in grid_df.iterrows():
        color = cmap_obj(norm(values[i]))
        rect = Rectangle(
            (row["xmin"], row["ymin"]),
            row["xmax"] - row["xmin"],
            row["ymax"] - row["ymin"],
            facecolor=color,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.9,
        )
        ax.add_patch(rect)

    minx = grid_df["xmin"].min()
    maxx = grid_df["xmax"].max()
    miny = grid_df["ymin"].min()
    maxy = grid_df["ymax"].max()
    pad_x = 0.05 * (maxx - minx)
    pad_y = 0.05 * (maxy - miny)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax)
    plt.tight_layout()



def build_quadtree_grid(gdf: gpd.GeoDataFrame, countmax: int, maxdepth: int) -> tuple[QuadTree, pd.DataFrame]:
    """Erzeuge aus den Punktdaten einen QuadTree und gib zusätzlich ein Grid-DataFrame zurück."""
    xs = gdf.geometry.x.to_numpy(dtype=float)
    ys = gdf.geometry.y.to_numpy(dtype=float)

    minx, miny, maxx, maxy = gdf.total_bounds
    qt = QuadTree(minx, maxx, miny, maxy, xs, ys, countmax=countmax, maxdepth=maxdepth)

    grid_df = pd.DataFrame(qt.lop).copy()
    grid_df["x_center"] = 0.5 * (grid_df["xmin"] + grid_df["xmax"])
    grid_df["y_center"] = 0.5 * (grid_df["ymin"] + grid_df["ymax"])

    return qt, grid_df



def main():
    global gdf, qt, grid_df, samples, f_est, field_est, nobs, lam, area

    gdf = load_earthquake_data()
    print(f"Loaded {len(gdf)} earthquake events")
    print(f"CRS: {gdf.crs}")

    qt, grid_df = build_quadtree_grid(gdf, countmax=NMAX, maxdepth=MAX_DEPTH)

    # Counts / Zentren / Zellflächen laden
    nobs = grid_df["count"].to_numpy().astype(int)
    x_center = grid_df["x_center"].to_numpy(dtype=float)
    y_center = grid_df["y_center"].to_numpy(dtype=float)
    area = (
        (grid_df["xmax"].to_numpy(dtype=float) - grid_df["xmin"].to_numpy(dtype=float))
        * (grid_df["ymax"].to_numpy(dtype=float) - grid_df["ymin"].to_numpy(dtype=float))
    )

    nbins = len(nobs)
    print(f"Constructed {nbins} quadtree cells")
    print(f"Total observed events: {nobs.sum()}")
    print(f"Min/Max counts per cell: {nobs.min()} / {nobs.max()}")
    print(f"Min/Max cell area: {area.min():.6f} / {area.max():.6f}")

    # lam als grobe obere Schranke der Rate pro Zelle
    lam = max(1, int(np.max(nobs) + 2 * np.sqrt(np.max(nobs))) + 1)
    print(f"Using lam = {lam}")

    # Prior-Kovarianz über Zellzentren
    # rho ist hier in denselben Koordinaten wie lon/lat angegeben.
    rho = 1.5
    v2 = 5.0 / lam
    Sigma0 = gaussian_covariance_from_coords(x_center, y_center, rho=rho, v2=v2)

    # Prior-Mittelwert so wählen, dass lam * sigmoid(mu0) ungefähr bei 5 liegt.
    baseline_rate = np.clip(5.0, 1e-6, lam - 1e-6)
    prior_mean = inv_sigmoid((baseline_rate / lam) * np.ones(nbins))

    pgd = PolyaGammaDensity(
        prior_mean=prior_mean,
        prior_covariance=Sigma0,
        lam=lam,
    )
    pgd.set_data(nobs)

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

    f_est = samples.mean(axis=0)
    field_est = pgd.field_from_f(f_est)

    plot_quadtree_values(
        grid_df,
        nobs,
        title="Observed counts on quadtree grid",
        cmap="viridis",
        power_gamma=0.6,
    )

    plot_quadtree_values(
        grid_df,
        field_est,
        title="Estimated rate field on quadtree grid",
        cmap="viridis",
        power_gamma=0.6,
    )

    # Posterior-Standardabweichung des latenten Feldes
    f_sd = samples.std(axis=0)
    plot_quadtree_values(
        grid_df,
        f_sd,
        title="Posterior sd of latent field f",
        cmap="magma",
        power_gamma=None,
    )

    plt.figure()
    plt.hist(nobs, bins=30)
    plt.title("Histogram of observed counts per quadtree cell")
    plt.xlabel("count")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.show()

    print("Gibbs sampling on quadtree grid done.")

    return samples, f_est, field_est, nobs, grid_df, lam, area, qt


if __name__ == "__main__":
    main()
    # explore(locals())
