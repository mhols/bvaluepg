from __future__ import annotations

"""INGV Italy preprocessing: load -> NND -> rotate -> cut -> grid.

Only pipeline data are written. Plotting and posterior sampling live in the
separate experiment script ``experiments/exp_italy_nnd_pipeline_pg.py``.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import clustering
from EqCat import EqCat
from create_horus_mat_for_clust import add_time_fields
from italy_rotated_workflow import (
    DEFAULT_ROTATION_DEGREES,
    RECT_X_MAX_KM,
    RECT_X_MIN_KM,
    RECT_Y_MAX_KM,
    RECT_Y_MIN_KM,
    italy_local_crs,
    load_ingv_txt,
    lonlat_to_rotated_km,
    principal_axis_rotation_degrees,
)


# Pipeline parameters. Change values here before running the script.
INPUT_FILE = SCRIPT_DIR / "italy_ingv_m2point5_2015-2026.txt"
OUTPUT_PREFIX = SCRIPT_DIR / "italy_nnd_rotate_cut_grid_Mc_2.5_eta_-4.60"

MIN_MAGNITUDE = 2.5
YEAR_MIN = 2015.0
YEAR_MAX = 2026.5

NND_D = 1.6
NND_B = 1.0
ETA_THRESHOLD_LOG10 = -4.6
RANDOM_SEED = 0

ROTATION_DEGREES = DEFAULT_ROTATION_DEGREES
X_MIN_KM = RECT_X_MIN_KM
X_MAX_KM = RECT_X_MAX_KM
Y_MIN_KM = RECT_Y_MIN_KM
Y_MAX_KM = RECT_Y_MAX_KM

NX = 100
NY = 200

# Set to an integer only for a quick chronological smoke test.
MAX_EVENTS = None


def prepare_catalog(path: Path, magnitude_min: float, year_min: float, year_max: float) -> pd.DataFrame:
    gdf = load_ingv_txt(path)
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df = df.rename(columns={"time": "datetime", "latitude": "lat", "longitude": "lon"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for column in ("lat", "lon", "depth", "mag"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["event_id_num"] = pd.to_numeric(df.get("event_id"), errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lon", "mag"]).copy()
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["second"] = df["datetime"].dt.second + df["datetime"].dt.microsecond / 1_000_000
    df = add_time_fields(df)
    df = df[
        (df["mag"] >= magnitude_min)
        & (df["decimal_year"] >= year_min)
        & (df["decimal_year"] <= year_max)
    ].copy()
    return df.sort_values("datetime").reset_index(drop=True)


def run_nnd(df: pd.DataFrame, d_value: float, b_value: float, random_seed: int) -> dict[str, np.ndarray]:
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
    np.random.seed(random_seed)
    return clustering.NND_eta(
        eqcat,
        {"D": d_value, "b": b_value},
        correct_co_located=True,
        verbose=False,
    )


def add_nnd_status(df: pd.DataFrame, nnd: dict[str, np.ndarray], threshold: float) -> pd.DataFrame:
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
    result["nnd_is_triggered"] = result["nnd_log10_eta"].lt(threshold).fillna(False)
    result["decluster_kept"] = ~result["nnd_is_triggered"]
    return result


def assign_grid(
    df: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = bounds
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    result = df.copy()
    inside = (
        result["x_rot_km"].between(x_min, x_max, inclusive="both")
        & result["y_rot_km"].between(y_min, y_max, inclusive="both")
    )
    ix = np.floor((result["x_rot_km"] - x_min) / (x_max - x_min) * nx).astype("Int64")
    iy = np.floor((result["y_rot_km"] - y_min) / (y_max - y_min) * ny).astype("Int64")
    ix = ix.clip(0, nx - 1).where(inside, -1)
    iy = iy.clip(0, ny - 1).where(inside, -1)
    global_id = (iy * nx + ix).where(inside, -1).astype("Int64")
    result["inside_final_cut"] = inside
    result["grid_ix"] = ix
    result["grid_iy"] = iy
    result["global_bin_id"] = global_id
    result["local_bin_id"] = global_id

    global_ids = np.arange(nx * ny, dtype=int)
    grid_iy, grid_ix = np.divmod(global_ids, nx)
    geometry = pd.DataFrame(
        {
            "global_bin_id": global_ids,
            "local_bin_id": global_ids,
            "local_to_global": global_ids,
            "global_to_local": global_ids,
            "iy": grid_iy,
            "ix": grid_ix,
            "x_center_rot_km": 0.5 * (x_edges[grid_ix] + x_edges[grid_ix + 1]),
            "y_center_rot_km": 0.5 * (y_edges[grid_iy] + y_edges[grid_iy + 1]),
            "inside_mask": True,
        }
    )
    used = result["inside_final_cut"] & result["decluster_kept"]
    counts = np.bincount(result.loc[used, "global_bin_id"].astype(int), minlength=nx * ny)
    return result, geometry, counts.reshape(ny, nx), x_edges, y_edges


def main() -> None:
    if NX < 1 or NY < 1:
        raise ValueError("nx and ny must be positive")
    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)

    df = prepare_catalog(INPUT_FILE, MIN_MAGNITUDE, YEAR_MIN, YEAR_MAX)

    if MAX_EVENTS is not None:
        df = df.iloc[:MAX_EVENTS].copy().reset_index(drop=True)
    local_crs = italy_local_crs(
        __import__("geopandas").GeoDataFrame(
            df, geometry=__import__("geopandas").points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
        )
    )
    transformer = __import__("pyproj").Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    x_m, y_m = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    
    df["x_proj_km"] = np.asarray(x_m) / 1000.0
    df["y_proj_km"] = np.asarray(y_m) / 1000.0

    nnd = run_nnd(df, NND_D, NND_B, RANDOM_SEED)
    df = add_nnd_status(df, nnd, ETA_THRESHOLD_LOG10)

    rotation = ROTATION_DEGREES
    if rotation is None:
        rotation = principal_axis_rotation_degrees(np.asarray(x_m), np.asarray(y_m))
    df["x_rot_km"], df["y_rot_km"] = lonlat_to_rotated_km(
        df["lon"].to_numpy(), df["lat"].to_numpy(), local_crs, rotation
    )
    bounds = (X_MIN_KM, X_MAX_KM, Y_MIN_KM, Y_MAX_KM)
    events, geometry, counts, x_edges, y_edges = assign_grid(df, bounds, NX, NY)

    prefix = OUTPUT_PREFIX
    event_path = prefix.with_name(prefix.name + "_events.csv")
    geometry_path = prefix.with_name(prefix.name + "_bins.csv")
    counts_path = prefix.with_name(prefix.name + "_counts.npz")
    meta_path = prefix.with_name(prefix.name + "_meta.json")
    events = events.rename(columns={"lat": "latitude", "lon": "longitude", "datetime": "time"})
    events.to_csv(event_path, index=False)
    geometry.to_csv(geometry_path, index=False)
    np.savez_compressed(counts_path, counts=counts, x_edges=x_edges, y_edges=y_edges)

    used = events["inside_final_cut"] & events["decluster_kept"]
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_order": ["load", "magnitude_time_filter", "NND_decluster", "project_rotate", "cut", "regular_grid", "counts"],
        "source": {"path": str(INPUT_FILE.resolve())},
        "outputs": {"events": str(event_path), "bins": str(geometry_path), "counts": str(counts_path), "meta": str(meta_path)},
        "selection": {"min_magnitude": MIN_MAGNITUDE, "year_min": YEAR_MIN, "year_max": YEAR_MAX},
        "nnd": {"implementation": "data/src/clustering.py::NND_eta", "D": NND_D, "b": NND_B, "eta_threshold_log10": ETA_THRESHOLD_LOG10, "correct_co_located": True, "random_seed": RANDOM_SEED},
        "projection": {"crs_wkt": local_crs.to_wkt(), "projected_units_saved": "km", "rotation_degrees": rotation},
        "grid": {"shape_ny_nx": [NY, NX], "scan_order": "C row-major; global_bin_id = iy * nx + ix", "bounds_rotated_km": {"x_min": bounds[0], "x_max": bounds[1], "y_min": bounds[2], "y_max": bounds[3]}, "cell_size_km": [(bounds[1] - bounds[0]) / NX, (bounds[3] - bounds[2]) / NY], "cell_area_km2": (bounds[1] - bounds[0]) / NX * (bounds[3] - bounds[2]) / NY, "inside_mask": "all cells true for the first rectangular-grid pipeline", "zero_cells_retained": True},
        "counts_definition": "events with inside_final_cut=True and decluster_kept=True",
        "event_counts": {"after_source_filters": len(events), "nnd_triggered": int(events["nnd_is_triggered"].sum()), "inside_final_cut": int(events["inside_final_cut"].sum()), "used_in_counts": int(used.sum()), "count_grid_sum": int(counts.sum())},
    }
    with meta_path.open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)

    print(json.dumps(meta["event_counts"], indent=2))
    print(f"events:   {event_path}")
    print(f"bins:     {geometry_path}")
    print(f"counts:   {counts_path}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
