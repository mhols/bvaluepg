from __future__ import annotations

"""
Diagnostic prior-parameter pass for the declustered Italy catalogue.

It is intentionally linear and exploratory. It does not save plots or data.
"""

from pathlib import Path
import json
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "source"
DATA_DIR = PROJECT_ROOT / "data"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polyagammadensity import PolyaGammaDensity, inv_sigmoid


CATALOG_CSV = DATA_DIR / "italy_ingv_rotated_rect_events_declustered_Mc_2.5_eta_-4.60.csv"
META_JSON = DATA_DIR / "italy_ingv_rotated_rect_meta.json"

NX = 100
NY = 200
LAMBDA_SAFETY_FACTOR = 1.35
LAMBDA_MIN_HEADROOM = 2.0
PRIOR_SD_CANDIDATES = [0.75, 1.0, 1.5, 2.0]


def logit(p: float) -> float:
    return float(inv_sigmoid(float(np.clip(p, 1e-9, 1.0 - 1e-9))))


def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "kept" in df.columns:
        df = df[df["kept"].astype(bool)].copy()
    required = {"x_rot_km", "y_rot_km", "mag"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df.dropna(subset=["x_rot_km", "y_rot_km"]).copy()


def load_bounds(df: pd.DataFrame) -> tuple[float, float, float, float]:
    if META_JSON.exists():
        with META_JSON.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        bounds = meta.get("rectangle_bounds", meta)
        keys = ["x_min_km", "x_max_km", "y_min_km", "y_max_km"]
        if all(key in bounds for key in keys):
            return tuple(float(bounds[key]) for key in keys)

    pad_x = 0.02 * (df["x_rot_km"].max() - df["x_rot_km"].min())
    pad_y = 0.02 * (df["y_rot_km"].max() - df["y_rot_km"].min())
    return (
        float(df["x_rot_km"].min() - pad_x),
        float(df["x_rot_km"].max() + pad_x),
        float(df["y_rot_km"].min() - pad_y),
        float(df["y_rot_km"].max() + pad_y),
    )


def make_count_grid(
    df: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = bounds
    counts, x_edges, y_edges = np.histogram2d(
        df["x_rot_km"].to_numpy(),
        df["y_rot_km"].to_numpy(),
        bins=[nx, ny],
        range=[[x_min, x_max], [y_min, y_max]],
    )
    return counts.T.astype(int), x_edges, y_edges


def suggest_prior_parameters(counts: np.ndarray) -> dict[str, float]:
    flat = counts.ravel()
    nonzero = flat[flat > 0]
    p99 = float(np.percentile(flat, 99.0))
    p995 = float(np.percentile(flat, 99.5))
    max_count = int(flat.max())
    mean_count = float(flat.mean())

    lam = max(
        max_count + LAMBDA_MIN_HEADROOM,
        LAMBDA_SAFETY_FACTOR * max(p995, 1.0),
        1.0,
    )
    lam = float(math.ceil(lam))

    prior_mean = logit(mean_count / lam)
    return {
        "lambda": lam,
        "prior_mean": prior_mean,
        "prior_variance_1": 1.0,
        "mean_count": mean_count,
        "max_count": float(max_count),
        "p99_count": p99,
        "p995_count": p995,
        "nonzero_mean": float(nonzero.mean()) if len(nonzero) else 0.0,
        "nonzero_fraction": float(len(nonzero) / len(flat)),
    }


def prior_predictive_pmf(lam: float, prior_mean: float, prior_sd: float, n_values: np.ndarray) -> np.ndarray:
    model = PolyaGammaDensity(
        prior_mean=np.array([prior_mean]),
        prior_covariance=np.array([[prior_sd**2]]),
        lam=lam,
    )
    return model.prior_n_under_gaussian(prior_mean, prior_sd**2, n_values)


def plot_event_and_count_grid(
    df: pd.DataFrame,
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    sc = ax.scatter(
        df["x_rot_km"],
        df["y_rot_km"],
        c=df["mag"],
        s=4,
        cmap="viridis",
        linewidths=0,
        alpha=0.75,
    )
    ax.set_title("Declustered Italy catalogue")
    ax.set_xlabel("rotated x [km]")
    ax.set_ylabel("rotated y [km]")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="magnitude")

    ax = axes[1]
    image = ax.imshow(
        counts,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        interpolation="nearest",
        cmap="magma",
        aspect="auto",
    )
    ax.set_title(f"Count grid ({counts.shape[1]} x {counts.shape[0]})")
    ax.set_xlabel("rotated x [km]")
    ax.set_ylabel("rotated y [km]")
    fig.colorbar(image, ax=ax, label="events per cell")


def plot_count_histograms(counts: np.ndarray, suggestions: dict[str, float]) -> None:
    flat = counts.ravel()
    nonzero = flat[flat > 0]
    max_count = int(flat.max())
    bins = np.arange(-0.5, max_count + 1.5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    ax.hist(flat, bins=bins, color="0.35", edgecolor="white")
    ax.axvline(suggestions["mean_count"], color="C0", lw=2, label="mean")
    ax.axvline(suggestions["p99_count"], color="C1", lw=2, label="p99")
    ax.axvline(suggestions["p995_count"], color="C3", lw=2, label="p99.5")
    ax.set_title("All grid counts, including zeros")
    ax.set_xlabel("count per cell")
    ax.set_ylabel("number of cells")
    ax.legend()

    ax = axes[1]
    if len(nonzero):
        ax.hist(nonzero, bins=np.arange(0.5, max_count + 1.5, 1.0), color="0.45", edgecolor="white")
    ax.axvline(suggestions["nonzero_mean"], color="C0", lw=2, label="nonzero mean")
    ax.set_title("Nonzero grid counts")
    ax.set_xlabel("count per nonzero cell")
    ax.set_ylabel("number of cells")
    ax.legend()


def plot_prior_predictive(counts: np.ndarray, suggestions: dict[str, float]) -> None:
    observed = counts.ravel()
    max_observed = int(observed.max())
    max_plot = max(max_observed + 3, int(suggestions["lambda"]) + 2)
    bins = np.arange(-0.5, max_plot + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.hist(
        observed,
        bins=bins,
        density=True,
        alpha=0.35,
        color="black",
        label="observed grid counts",
    )

    for prior_sd in PRIOR_SD_CANDIDATES:
        n_values = np.arange(0, max_plot + 1, dtype=int)
        pmf = prior_predictive_pmf(suggestions["lambda"], suggestions["prior_mean"], prior_sd, n_values)
        ax.step(
            n_values,
            pmf,
            where="mid",
            lw=2,
            label=f"prior predictive sd={prior_sd:g}",
        )

    ax.axvline(suggestions["lambda"], color="C3", ls="--", lw=2, label="lambda upper scale")
    ax.set_xlim(-0.5, max_plot + 0.5)
    ax.set_title("Prior predictive count check")
    ax.set_xlabel("count per cell")
    ax.set_ylabel("density")
    ax.legend()


def print_summary(df: pd.DataFrame, counts: np.ndarray, suggestions: dict[str, float]) -> None:
    flat = counts.ravel()
    print("\nDeclustered Italy count-grid diagnostic")
    print("=" * 45)
    print(f"catalogue: {CATALOG_CSV}")
    print(f"events used: {len(df)}")
    print(f"grid shape: {counts.shape[1]} x {counts.shape[0]} = {counts.size} cells")
    print(f"total grid count: {int(flat.sum())}")
    print(f"zero cells: {int((flat == 0).sum())} ({(flat == 0).mean():.1%})")
    print(f"nonzero cells: {int((flat > 0).sum())} ({suggestions['nonzero_fraction']:.1%})")
    print(f"mean count: {suggestions['mean_count']:.3f}")
    print(f"nonzero mean count: {suggestions['nonzero_mean']:.3f}")
    print(f"p99 count: {suggestions['p99_count']:.3f}")
    print(f"p99.5 count: {suggestions['p995_count']:.3f}")
    print(f"max count: {int(suggestions['max_count'])}")
    print("\nFirst prior-parameter suggestion for lambda * sigmoid(f):")
    print(f"lambda = {suggestions['lambda']:.0f}")
    print(f"prior_mean = logit(mean_count / lambda) = {suggestions['prior_mean']:.3f}")
    print("try prior_sd values:", ", ".join(str(v) for v in PRIOR_SD_CANDIDATES))
    print("Interpretation: lambda is a per-cell upper intensity scale, not a global event count.")


def pg_samples(model: PolyaGammaDensity, nobs: int, random_seed: int) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = [
            model.draw_pg_samples_one_dimension(
                PRIOR_MEAN,
                PRIOR_VARIANCE,
                nobs,
                random_seed=random_seed + i,
            )
            for i in range(3)
        ]
    return np.asarray(samples).ravel()

def draw_pg_samples(model: PolyaGammaDensity, laplace_mean: float, random_seed: int) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = [
            model.draw_pg_samples_one_dimension(
                laplace_mean,
                PRIOR_VARIANCE,
                nobs=1,
                random_seed=random_seed + i,
            )
            for i in range(3)
        ]
    return np.asarray(samples).ravel()
 

def main() -> None:
    df = load_catalog(CATALOG_CSV)
    bounds = load_bounds(df)
    counts, x_edges, y_edges = make_count_grid(df, bounds, NX, NY)
    suggestions = suggest_prior_parameters(counts)

    print_summary(df, counts, suggestions)
    plot_event_and_count_grid(df, counts, x_edges, y_edges)
    plot_count_histograms(counts, suggestions)
    plot_prior_predictive(counts, suggestions)
    plt.show()





if __name__ == "__main__":
    main()
