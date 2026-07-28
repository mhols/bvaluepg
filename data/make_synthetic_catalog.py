"""Erzeuge einen vollstaendigen synthetischen Erdbeben-Katalog.

Ausgaben:
- synthetic_catalog_events.csv: INGV-aehnliche Pipe-CSV fuer unsere Pipeline.
- synthetic_catalog_truth.npz: lambda_true, f_true, counts und Zellraster.
- synthetic_catalog_meta.json: Parameter und kurze Summen.

Der Katalog orientiert sich am Format von italy_ingv_m2point5_2015-2026.txt:
Pipe-Separator, INGV-Spaltennamen und lon/lat/depth/magnitude/time.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPO_ROOT / "source"
SYNTHETIC_DIR = SCRIPT_DIR / "synthetic"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
import polyagammadensity as pgd
import syntheticdata as sd


# ============================================================
# PARAMETERS TO CHANGE
# ============================================================

OUTPUT_PREFIX = SYNTHETIC_DIR / "2026-07-28_bars_synthetic_catalog"
PATTERN = "bars"  # "block", "bars", "checkerboard"

SEED = 20260728
NY = 64
NX = 64

# Italy-like lon/lat window. The values are deliberately simple, not a map cut.
LON_MIN = 6.0
LON_MAX = 19.0
LAT_MIN = 36.0
LAT_MAX = 47.5

# Upper limit of the sigmoid model for the expected number per cell.
LAM_MAX = 12.0

# Pattern 1: upper-left block.
PATTERN_1_BACKGROUND = 0.8
PATTERN_1_HOT = 8.0
PATTERN_1_ROWS = slice(0, 26)
PATTERN_1_COLS = slice(0, 26)

# Pattern 2: three horizontal bars, with the middle bar strongest.
PATTERN_2_BACKGROUND = 0.6
PATTERN_2_BAR_TOP_ADD = 4.0
PATTERN_2_BAR_MIDDLE_ADD = 8.0
PATTERN_2_BAR_BOTTOM_ADD = 4.0
PATTERN_2_BAR_WIDTH = 5
PATTERN_2_COL_START = 14
PATTERN_2_COL_END = 50
PATTERN_2_BAR_CENTERS_Y = (24, 32, 40)

START_TIME = "2015-01-01T00:00:00"
T_END_YEARS = 10.0

MIN_MAGNITUDE = 2.5
B_VALUE = 1.0
MAX_MAGNITUDE = 7.5

DEPTH_MIN_KM = 2.0
DEPTH_MAX_KM = 18.0

OUTPUT_SEPARATOR = "|"
SHOW_PLOTS = True


def field_to_sigmoid_latent(lambda_true: np.ndarray, lam_max: float) -> np.ndarray:
    """Ordne lambda_i zu f_i, wobei lambda_i = lam_max * sigmoid(f_i) gilt. haben wir so auch in 
    polyagammadensity.py implementiert. zum testen statt import
    """
    probability = np.asarray(lambda_true, dtype=float) / float(lam_max)
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return pgd.inv_sigmoid(probability)


def make_pattern_1_block() -> np.ndarray:
    lambda_true = PATTERN_1_BACKGROUND * np.ones((NY, NX), dtype=float)
    lambda_true[PATTERN_1_ROWS, PATTERN_1_COLS] = PATTERN_1_HOT
    return lambda_true


def make_pattern_2_bars() -> np.ndarray:
    lambda_true = PATTERN_2_BACKGROUND * np.ones((NY, NX), dtype=float)
    cols = slice(PATTERN_2_COL_START, PATTERN_2_COL_END)
    additions = (
        PATTERN_2_BAR_TOP_ADD,
        PATTERN_2_BAR_MIDDLE_ADD,
        PATTERN_2_BAR_BOTTOM_ADD,
    )

    for center, addition in zip(PATTERN_2_BAR_CENTERS_Y, additions):
        half_width = PATTERN_2_BAR_WIDTH // 2
        rows = slice(center - half_width, center + half_width + 1)
        lambda_true[rows, cols] += addition
    return lambda_true


def make_pattern_3_() -> np.ndarray:
    pass


def sample_counts(lambda_true: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.poisson(lambda_true).astype(int)


def random_catalog_from_nobs(nobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Erzeuge zufaellige Ereigniskoordinaten innerhalb der besetzten Bins.

    Copy-paste der Logik aus polyagammadensity.py::Mixin2D.random_catalog_from_nobs,
    nur ohne Klassenkontext. Rueckgabe ist wie dort: erst Zeilenkoordinate,
    dann Spaltenkoordinate.
    """
    nobs = np.asarray(nobs, dtype=int)
    if nobs.ndim != 2:
        raise ValueError("nobs must be a two-dimensional count image.")

    n, m = nobs.shape
    x = []
    y = []

    for i in range(n):
        for j in range(m):
            for k in range(nobs[i, j]):
                x.append(i + np.random.uniform())
                y.append(j + np.random.uniform())
    return np.array(x), np.array(y)


def build_true_intensity() -> np.ndarray:
    if PATTERN == "block":
        lambda_true = make_pattern_1_block()
    elif PATTERN == "bars":
        lambda_true = make_pattern_2_bars()
    elif PATTERN == "checkerboard":
        lambda_true = sd.checkerboard(nn=8, ncheck=8, a=0.8, b=6.5)
    else:
        raise ValueError("PATTERN must be 'block', 'bars', or 'checkerboard'.")

    if lambda_true.shape != (NY, NX):
        raise ValueError(f"Expected lambda_true shape {(NY, NX)}, got {lambda_true.shape}.")
    return lambda_true.astype(float)


def counts_to_catalog_points(counts: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    np.random.seed(SEED)
    row_float, col_float = random_catalog_from_nobs(counts)

    bin_iy = np.floor(row_float).astype(int)
    bin_ix = np.floor(col_float).astype(int)

    x_unit = np.clip(col_float / NX, 0.0, np.nextafter(1.0, 0.0))
    y_unit = np.clip(row_float / NY, 0.0, np.nextafter(1.0, 0.0))

    lon = LON_MIN + x_unit * (LON_MAX - LON_MIN)
    lat = LAT_MIN + y_unit * (LAT_MAX - LAT_MIN)

    t_years = np.sort(rng.uniform(0.0, T_END_YEARS, size=len(lon)))
    start = pd.Timestamp(START_TIME, tz="UTC")
    datetimes = start + pd.to_timedelta(t_years * 365.25, unit="D")

    beta = B_VALUE * np.log(10.0)
    mag = MIN_MAGNITUDE + rng.exponential(scale=1.0 / beta, size=len(lon))
    mag = np.minimum(mag, MAX_MAGNITUDE)
    depth = rng.uniform(DEPTH_MIN_KM, DEPTH_MAX_KM, size=len(lon))

    return pd.DataFrame(
        {
            "event_id": np.arange(1, len(lon) + 1, dtype=int),
            "t_years": t_years,
            "datetime": datetimes,
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "mag": mag,
            "bin_ix": bin_ix,
            "bin_iy": bin_iy,
            "lambda_true": np.nan,
        }
    )


def add_f_truth(events: pd.DataFrame, f_true: np.ndarray) -> pd.DataFrame:
    result = events.copy()
    result["f_true"] = f_true[result["bin_iy"].to_numpy(int), result["bin_ix"].to_numpy(int)]
    return result


def write_outputs(events: pd.DataFrame, lambda_true: np.ndarray, f_true: np.ndarray, counts: np.ndarray) -> None:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    ingv = pd.DataFrame(
        {
            "#EventID": events["event_id"],
            "Time": events["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "Latitude": events["lat"],
            "Longitude": events["lon"],
            "Depth/Km": events["depth"],
            "Author": "SYNTHETIC",
            "Catalog": "BvaluePG",
            "Contributor": "Toni",
            "ContributorID": "",
            "MagType": "Mw",
            "Magnitude": events["mag"],
            "MagAuthor": "synthetic",
            "EventLocationName": "synthetic unit-square catalog",
            "EventType": "earthquake",
            "t_years": events["t_years"],
            "bin_ix": events["bin_ix"],
            "bin_iy": events["bin_iy"],
            "lambda_true": events["lambda_true"],
            "f_true": events["f_true"],
        }
    )
    ingv.to_csv(OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_events.csv"), sep=OUTPUT_SEPARATOR, index=False)

    lon_edges = np.linspace(LON_MIN, LON_MAX, NX + 1)
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, NY + 1)
    np.savez_compressed(
        OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_truth.npz"),
        lambda_true=lambda_true,
        f_true=f_true,
        counts=counts,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        pattern=PATTERN,
    )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pattern": PATTERN,
        "seed": SEED,
        "ny": NY,
        "nx": NX,
        "domain": {"lon_min": LON_MIN, "lon_max": LON_MAX, "lat_min": LAT_MIN, "lat_max": LAT_MAX},
        "time": {"start": START_TIME, "t_end_years": T_END_YEARS},
        "magnitude": {"min_magnitude": MIN_MAGNITUDE, "b_value": B_VALUE, "max_magnitude": MAX_MAGNITUDE},
        "depth": {"min_km": DEPTH_MIN_KM, "max_km": DEPTH_MAX_KM},
        "n_events": int(len(events)),
        "count_sum": int(counts.sum()),
        "lambda_min": float(lambda_true.min()),
        "lambda_max": float(lambda_true.max()),
        "lambda_mean": float(lambda_true.mean()),
        "files": {
            "pipeline_events": str(OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_events.csv")),
            "truth": str(OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_truth.npz")),
        },
    }
    with OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_meta.json").open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)


def show_plots(lambda_true: np.ndarray, f_true: np.ndarray, counts: np.ndarray, events: pd.DataFrame) -> None:
    if not SHOW_PLOTS:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], lambda_true, "true rate lambda"),
        (axes[0, 1], counts, "sampled counts"),
        (axes[1, 0], f_true, "true latent field f"),
    )
    for ax, image, title in panels:
        im = ax.imshow(image, origin="lower", extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX))
        ax.set_title(title)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, fraction=0.046)

    axes[1, 1].scatter(events["lon"], events["lat"], s=2, alpha=0.45)
    axes[1, 1].set_title("synthetic event catalogue")
    axes[1, 1].set_xlabel("lon")
    axes[1, 1].set_ylabel("lat")
    axes[1, 1].set_aspect("equal", adjustable="box")
    plt.show()


def main() -> None:
    rng = np.random.default_rng(SEED)
    lambda_true = build_true_intensity()
    f_true = field_to_sigmoid_latent(lambda_true, lam_max=LAM_MAX)
    counts = sample_counts(lambda_true, seed=SEED)
    events = counts_to_catalog_points(counts, rng)
    events["lambda_true"] = lambda_true[events["bin_iy"].to_numpy(int), events["bin_ix"].to_numpy(int)]
    events = add_f_truth(events, f_true)
    events = events.sort_values("t_years").reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1, dtype=int)

    write_outputs(events, lambda_true, f_true, counts)
    print(f"events: {len(events)}")
    print(f"count sum: {int(counts.sum())}")
    print(f"pipeline csv: {OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + '_events.csv')}")
    print(f"truth npz: {OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + '_truth.npz')}")

    show_plots(lambda_true, f_true, counts, events)


if __name__ == "__main__":
    main()
