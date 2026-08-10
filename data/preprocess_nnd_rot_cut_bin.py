from __future__ import annotations

"""Linearer preprocessing pipeline for Italy catalogues.

Workflow:
1. load raw INGV/HORUS-style catalogue
2. add time fields for EqCat/NND
3. run NND declustering on the generous catalogue
4. project lon/lat to local x/y kilometres using the simple binning formula
5. rotate x/y coordinates
6. cut to a rectangular window
7. assign square grid bins and count kept/background events

todo morgen:
meta daten und plots schreiben
synthetic catalogues generieren
    ganzen Katalog fuer die pipeline generieren oder nur background events generieren :/
    vielleicht gleich mit dummy zeit und so
    1. Erzeuge ein wahres Feld, (Block, Balken oder Checkerboard)
    2. Ziehe daraus Poisson-Counts
    3. Wandle Counts in zufällige Eventpunkte pro Bin um
    4. Skaliere diese Punkte auf ein kuenstilches x_proj_km/y_proj_km-Gebiet
    5. Rechne daraus passende lon/lat zurück oder besser direkt synthetische lon/lat setzen
    6. Ergänze Dummy-Zeit, Tiefe, Magnitude (und optional lambda_true, f_true, bin_ix, bin_iy)

ok das wird zuviel fuer hier, besser in eigenem skript. ich muss nur aufpassen, dass die Spaltennamen und Formate passen
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import clustering
from EqCat import EqCat
from create_horus_mat_for_clust import add_time_fields


# ============================================================
# PARAMETERS TO CHANGE
# ============================================================

INPUT_FILE = SCRIPT_DIR / "italy_ingv_m2point5_2015-2026.txt"
OUTPUT_PREFIX = SCRIPT_DIR / "preprocess_nnd_rot_cut_bin_Mc_2.5_eta_-4.60_dkm_20"

# INPUT_FILE = SCRIPT_DIR / "synthetic/2026-07-28_bars_synthetic_catalog_events.csv"
# OUTPUT_PREFIX = SCRIPT_DIR / "synthetic/preprocess_nnd_rot_cut_bin_Mc_2.5_eta_-4.60_dkm_20"

MIN_MAGNITUDE = 2.5
MAX_MAGNITUDE = None
YEAR_MIN = 2015.0
YEAR_MAX = 2026.5

NND_D = 1.6
NND_B = 1.0
ETA_THRESHOLD_LOG10 = -4.6
RANDOM_SEED = 0

EarthRadius = 6371.0
# None means: estimate gamma from the selected catalogue. In the old projection
# used before binning_catalog.py, gamma is the reference latitude and longitude
# is measured relative to Greenwich.
GAMMA_DEG = None

ROTATION_DEGREES = -45.0

# Bounds in rotated kilometres. Set all to None to use automatic quantile bounds.
X_MIN_KM = None
X_MAX_KM = None
Y_MIN_KM = None
Y_MAX_KM = None
AUTO_RECTANGLE_QUANTILE_LOW = 0.01
AUTO_RECTANGLE_QUANTILE_HIGH = 0.99
AUTO_RECTANGLE_PADDING_KM = 40.0

# Square bin size in rotated kilometres.
dkm = 2.0

OUTPUT_SEPARATOR = "|"


def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, sep="|", skiprows=0)
    df.columns = [str(column).strip() for column in df.columns]
    df = df.rename(
        columns={
            "#EventID": "event_id",
            "EventID": "event_id",
            "Time": "datetime",
            "Latitude": "lat",
            "Longitude": "lon",
            "Depth/Km": "depth",
            "Magnitude": "mag",
        }
    )
    required = ["datetime", "lat", "lon", "depth", "mag"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; available columns: {df.columns.tolist()}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for column in ["lat", "lon", "depth", "mag"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "event_id" in df.columns:
        df["event_id_num"] = pd.to_numeric(df["event_id"], errors="coerce")
    else:
        df["event_id"] = np.arange(1, len(df) + 1)
        df["event_id_num"] = df["event_id"].astype(float)

    df = df.dropna(subset=["datetime", "lat", "lon", "mag"]).copy()
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["second"] = df["datetime"].dt.second + df["datetime"].dt.microsecond / 1_000_000
    df = add_time_fields(df)
    return df.sort_values("datetime").reset_index(drop=True)


def create_synthetic_catalog() -> pd.DataFrame:
    """Placeholder for later synthetic background-only catalogues.

    Intended output columns:
    datetime, lat, lon, depth, mag, event_id, and optionally f_true/lambda_true.
    The returned dataframe can then be passed through the same pipeline below.

synthetic catalogues generieren
    ganzen Katalog fuer die pipeline generieren oder nur background events generieren :/
    vielleicht gleich mit dummy zeit und so
    1. Erzeuge ein wahres Feld, (Block, Balken oder Checkerboard)
    2. Ziehe daraus Poisson-Counts
    3. Wandle Counts in zufällige Eventpunkte pro Bin um
    4. Skaliere diese Punkte auf ein kuenstilches x_proj_km/y_proj_km-Gebiet
    5. Rechne daraus passende lon/lat zurück oder besser direkt synthetische lon/lat setzen
    6. Ergänze Dummy-Zeit, Tiefe, Magnitude (und optional lambda_true, f_true, bin_ix, bin_iy)

ok das wird zuviel fuer hier, besser in eigenem skript. ich muss nur aufpassen, dass die Spaltennamen und Formate passen

    """
    raise NotImplementedError("Synthetic catalogue generation will be added later.")


def filter_catalog(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if MIN_MAGNITUDE is not None:
        result = result[result["mag"] >= float(MIN_MAGNITUDE)].copy()
    if MAX_MAGNITUDE is not None:
        result = result[result["mag"] <= float(MAX_MAGNITUDE)].copy()
    if YEAR_MIN is not None:
        result = result[result["decimal_year"] >= float(YEAR_MIN)].copy()
    if YEAR_MAX is not None:
        result = result[result["decimal_year"] <= float(YEAR_MAX)].copy()
    return result.sort_values("datetime").reset_index(drop=True)


def make_eqcat(df: pd.DataFrame) -> EqCat:
    eqcat = EqCat()
    eqcat.data = {
        "N": df["N"].to_numpy(float),
        "Time": df["decimal_year"].to_numpy(float),
        "Mag": df["mag"].to_numpy(float),
        "Lat": df["lat"].to_numpy(float),
        "Lon": df["lon"].to_numpy(float),
        "Depth": df["depth"].fillna(0.0).to_numpy(float),
        "X": df["x_proj_km"].to_numpy(float),
        "Y": df["y_proj_km"].to_numpy(float),
    }
    return eqcat


def run_nnd_declustering(eqcat: EqCat) -> dict[str, np.ndarray]:
    dpar = {"D": NND_D, "b": NND_B, "Mc": MIN_MAGNITUDE}
    np.random.seed(RANDOM_SEED)
    eqcat.data["Z"] = eqcat.data["Depth"]
    return clustering.NND_eta(eqcat, dpar, correct_co_located=True, verbose=False)


def add_nnd_status(df: pd.DataFrame, nnd: dict[str, np.ndarray]) -> pd.DataFrame:
    result = df.copy()
    result["nnd_parent_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["nnd_eta"] = np.nan
    result["nnd_log10_eta"] = np.nan

    child_to_row = pd.Series(result.index.to_numpy(), index=result["N"].astype(float)).to_dict()
    for child, parent, eta in zip(nnd["aEqID_c"], nnd["aEqID_p"], nnd["aNND"]):
        row = child_to_row.get(float(child))
        if row is None:
            continue
        result.at[row, "nnd_parent_id"] = int(parent)
        result.at[row, "nnd_eta"] = float(eta)
        result.at[row, "nnd_log10_eta"] = float(np.log10(eta))

    result["nnd_is_triggered"] = result["nnd_log10_eta"].lt(ETA_THRESHOLD_LOG10).fillna(False)
    result["decluster_kept"] = ~result["nnd_is_triggered"]
    return result


def project_lonlat_to_xy_km(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    result = df.copy()
    gamma = float(result["lat"].mean()) if GAMMA_DEG is None else float(GAMMA_DEG)

    result["x_proj_km"] = EarthRadius * np.radians(result["lon"].to_numpy(float)) * np.cos(np.radians(gamma))
    result["y_proj_km"] = EarthRadius * np.radians(result["lat"].to_numpy(float) - gamma)
    meta = {"EarthRadius": EarthRadius, "gamma_deg": gamma, "longitude_reference": "Greenwich"}
    return result, meta


def rotate_xy(x: np.ndarray, y: np.ndarray, angle_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return cos_a * x - sin_a * y, sin_a * x + cos_a * y


def add_rotated_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["x_rot_km"], result["y_rot_km"] = rotate_xy(
        result["x_proj_km"].to_numpy(float),
        result["y_proj_km"].to_numpy(float),
        ROTATION_DEGREES,
    )
    return result


def rectangle_bounds(df: pd.DataFrame) -> tuple[float, float, float, float]:
    manual_bounds = (X_MIN_KM, X_MAX_KM, Y_MIN_KM, Y_MAX_KM)
    if all(value is not None for value in manual_bounds):
        return tuple(float(value) for value in manual_bounds)
    if any(value is not None for value in manual_bounds):
        raise ValueError("Set all rectangle bounds or set all to None for automatic bounds.")

    reference = df[df["decluster_kept"]] if "decluster_kept" in df.columns else df
    x = reference["x_rot_km"].to_numpy(float)
    y = reference["y_rot_km"].to_numpy(float)
    q_low = AUTO_RECTANGLE_QUANTILE_LOW
    q_high = AUTO_RECTANGLE_QUANTILE_HIGH
    pad = AUTO_RECTANGLE_PADDING_KM
    return (
        float(np.quantile(x, q_low) - pad),
        float(np.quantile(x, q_high) + pad),
        float(np.quantile(y, q_low) - pad),
        float(np.quantile(y, q_high) + pad),
    )


def add_cut_status(df: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    x_min, x_max, y_min, y_max = bounds
    result = df.copy()
    result["inside_final_cut"] = (
        result["x_rot_km"].between(x_min, x_max, inclusive="both")
        & result["y_rot_km"].between(y_min, y_max, inclusive="both")
    )
    return result


def assign_grid_bins(
    df: pd.DataFrame,
    bounds: tuple[float, float, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    result = df.copy()
    inside = result["inside_final_cut"]
    used = inside & result["decluster_kept"]
    dd = result[used].copy()
    if dd.empty:
        raise ValueError("No kept events inside final cut; cannot build bins.")

    nbinx = int(np.ceil((dd["x_rot_km"].max() - dd["x_rot_km"].min()) / dkm))
    nbiny = int(np.ceil((dd["y_rot_km"].max() - dd["y_rot_km"].min()) / dkm))
    x_edges = dd["x_rot_km"].min() + np.arange(nbinx + 1) * dkm
    y_edges = dd["y_rot_km"].min() + np.arange(nbiny + 1) * dkm

    i = np.digitize(result["x_rot_km"], x_edges) - 1
    j = np.digitize(result["y_rot_km"], y_edges) - 1
    in_grid = inside & (i >= 0) & (i < nbinx) & (j >= 0) & (j < nbiny)
    ix = pd.Series(i, index=result.index).where(in_grid, -1).astype("Int64")
    iy = pd.Series(j, index=result.index).where(in_grid, -1).astype("Int64")
    global_id = (iy * nbinx + ix).where(in_grid, -1).astype("Int64")

    result["grid_ix"] = ix
    result["grid_iy"] = iy
    result["global_bin_id"] = global_id
    result["local_bin_id"] = global_id

    i_used = i[used.to_numpy()]
    j_used = j[used.to_numpy()]
    res = []
    for ii in range(nbinx):
        for jj in range(nbiny):
            I = (i_used == ii) & (j_used == jj)
            global_bin_id = jj * nbinx + ii
            res.append(
                {
                    "global_bin_id": global_bin_id,
                    "local_bin_id": global_bin_id,
                    "local_to_global": global_bin_id,
                    "global_to_local": global_bin_id,
                    "bin_ix": ii,
                    "bin_iy": jj,
                    "n_events": int(np.sum(I)),
                    "x_center_rot_km": dd["x_rot_km"].min() + (ii + 0.5) * dkm,
                    "y_center_rot_km": dd["y_rot_km"].min() + (jj + 0.5) * dkm,
                    "inside_mask": True,
                }
            )

    bins = pd.DataFrame(res)
    counts = np.array([r["n_events"] for r in res]).reshape((nbinx, nbiny), order="C").T
    return result, bins, counts, x_edges, y_edges


def write_outputs(
    events: pd.DataFrame,
    bins: pd.DataFrame,
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    bounds: tuple[float, float, float, float],
    projection_meta: dict[str, float],
) -> None:
    prefix = OUTPUT_PREFIX
    event_path = prefix.with_name(prefix.name + "_events.csv")
    bins_path = prefix.with_name(prefix.name + "_bins.csv")
    counts_path = prefix.with_name(prefix.name + "_counts.npz")
    meta_path = prefix.with_name(prefix.name + "_meta.json")

    events.to_csv(event_path, index=False, sep=OUTPUT_SEPARATOR)
    bins.to_csv(bins_path, index=False, sep=OUTPUT_SEPARATOR)
    np.savez_compressed(counts_path, counts=counts, x_edges=x_edges, y_edges=y_edges)

    used = events["inside_final_cut"] & events["decluster_kept"]
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_order": [
            "load",
            "magnitude_time_filter",
            "manual_lonlat_to_xy_km",
            "NND_decluster",
            "rotate",
            "cut",
            "regular_square_grid",
            "counts",
        ],
        "source": {"path": str(INPUT_FILE.resolve())},
        "outputs": {
            "events": str(event_path),
            "bins": str(bins_path),
            "counts": str(counts_path),
            "meta": str(meta_path),
        },
        "selection": {
            "min_magnitude": MIN_MAGNITUDE,
            "max_magnitude": MAX_MAGNITUDE,
            "year_min": YEAR_MIN,
            "year_max": YEAR_MAX,
        },
        "nnd": {
            "implementation": "data/src/clustering.py::NND_eta",
            "D": NND_D,
            "b": NND_B,
            "eta_threshold_log10": ETA_THRESHOLD_LOG10,
            "correct_co_located": True,
            "random_seed": RANDOM_SEED,
        },
        "projection": {
            "formula": "x = EarthRadius*radians(lon)*cos(radians(gamma)); y = EarthRadius*radians(lat-gamma)",
            **projection_meta,
        },
        "rotation": {"rotation_degrees": ROTATION_DEGREES},
        "grid": {
            "shape_ny_nx": list(counts.shape),
            "scan_order": "C row-major; global_bin_id = iy * nx + ix",
            "bounds_rotated_km": {
                "x_min": bounds[0],
                "x_max_requested": bounds[1],
                "x_max_grid": float(x_edges[-1]),
                "y_min": bounds[2],
                "y_max_requested": bounds[3],
                "y_max_grid": float(y_edges[-1]),
            },
            "cell_size_km": dkm,
            "cell_area_km2": dkm**2,
            "inside_mask": "all rectangular grid cells true",
            "zero_cells_retained": True,
        },
        "counts_definition": "events with inside_final_cut=True and decluster_kept=True",
        "event_counts": {
            "after_source_filters": int(len(events)),
            "nnd_triggered": int(events["nnd_is_triggered"].sum()),
            "decluster_kept": int(events["decluster_kept"].sum()),
            "inside_final_cut": int(events["inside_final_cut"].sum()),
            "used_in_counts": int(used.sum()),
            "count_grid_sum": int(counts.sum()),
        },
    }
    with meta_path.open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)

    print(json.dumps(meta["event_counts"], indent=2))
    print(f"events: {event_path}")
    print(f"bins:   {bins_path}")
    print(f"counts: {counts_path}")
    print(f"meta:   {meta_path}")


def show_plots(
    events: pd.DataFrame,
    bins: pd.DataFrame,
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> None:
    kept = events["decluster_kept"]
    triggered = ~kept
    inside = events["inside_final_cut"]
    used = inside & kept

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].scatter(events["lon"], events["lat"], s=3, color="0.35", alpha=0.35)
    axes[0].set_title("Original catalogue")
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(events["x_proj_km"], events["y_proj_km"], s=3, color="0.35", alpha=0.35)
    axes[1].set_title("Projected catalogue")
    axes[1].set_xlabel("x_proj_km")
    axes[1].set_ylabel("y_proj_km")
    axes[1].set_aspect("equal", adjustable="box")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    eta = events["nnd_log10_eta"].dropna().to_numpy()
    if len(eta):
        axes[0].hist(eta, bins=np.arange(np.floor(eta.min()), np.ceil(eta.max()) + 0.2, 0.2), color="0.45")
    axes[0].axvline(ETA_THRESHOLD_LOG10, color="C3", linestyle="--", linewidth=2)
    axes[0].set_title("NND distribution")
    axes[0].set_xlabel("log10(eta)")
    axes[0].set_ylabel("events")

    axes[1].scatter(
        events.loc[triggered, "x_proj_km"],
        events.loc[triggered, "y_proj_km"],
        s=3,
        color="0.65",
        alpha=0.35,
        label="triggered",
    )
    axes[1].scatter(
        events.loc[kept, "x_proj_km"],
        events.loc[kept, "y_proj_km"],
        s=4,
        color="C0",
        alpha=0.55,
        label="kept",
    )
    axes[1].set_title("NND kept/triggered")
    axes[1].set_xlabel("x_proj_km")
    axes[1].set_ylabel("y_proj_km")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(markerscale=3)

    x_min, x_max, y_min, y_max = bounds
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.scatter(events.loc[~inside, "x_rot_km"], events.loc[~inside, "y_rot_km"], s=3, color="0.7", alpha=0.25, label="outside cut")
    ax.scatter(events.loc[inside & triggered, "x_rot_km"], events.loc[inside & triggered, "y_rot_km"], s=3, color="0.45", alpha=0.25, label="inside triggered")
    ax.scatter(events.loc[used, "x_rot_km"], events.loc[used, "y_rot_km"], s=5, color="C0", alpha=0.6, label="inside kept")
    ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="C3", linewidth=2)
    ax.set_title("Rotated catalogue and cut")
    ax.set_xlabel("x_rot_km")
    ax.set_ylabel("y_rot_km")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(markerscale=3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    artist = axes[0].imshow(
        counts,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[0].set_title("Binned kept counts")
    axes[0].set_xlabel("x_rot_km")
    axes[0].set_ylabel("y_rot_km")
    fig.colorbar(artist, ax=axes[0], label="events per bin")

    max_count = int(counts.max())
    axes[1].hist(counts.ravel(), bins=np.arange(-0.5, max_count + 1.5, 1), color="0.45", edgecolor="white")
    axes[1].set_yscale("log")
    axes[1].set_title("Count histogram")
    axes[1].set_xlabel("events per bin")
    axes[1].set_ylabel("number of bins")

    plt.show()


def main() -> None:
    df = load_catalog(INPUT_FILE)
    df = filter_catalog(df)
    df, projection_meta = project_lonlat_to_xy_km(df)
    eqcat = make_eqcat(df)
    nnd = run_nnd_declustering(eqcat)
    df = add_nnd_status(df, nnd)
    df = add_rotated_coordinates(df)
    bounds = rectangle_bounds(df)
    df = add_cut_status(df, bounds)
    events, bins, counts, x_edges, y_edges = assign_grid_bins(df, bounds)
    write_outputs(events, bins, counts, x_edges, y_edges, bounds, projection_meta)
    show_plots(events, bins, counts, x_edges, y_edges, bounds)


if __name__ == "__main__":
    main()
