import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bvaluepg_mplconfig")

import geodatasets
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PLOTS_DIR = REPO_ROOT / "plots"
DATA_FILE = SCRIPT_DIR / "italy_ingv_m2point5_2015-2026.txt"

OUTPUT_EVENTS = SCRIPT_DIR / "italy_ingv_rotated_rect_events.csv"
OUTPUT_META = SCRIPT_DIR / "italy_ingv_rotated_rect_meta.json"
OUTPUT_PLOT = PLOTS_DIR / "italy_ingv_rotated_rect_cut.png"

# Set to a number such as 67.0 to fix the angle in the script.
# Keep as None to compute the angle automatically from the point cloud.
DEFAULT_ROTATION_DEGREES = None

# Rectangle bounds in rotated kilometers. Keep all four as None to use an
# automatic quantile box; set all four numbers to define your own cut.
RECT_X_MIN_KM = None
RECT_X_MAX_KM = None
RECT_Y_MIN_KM = None
RECT_Y_MAX_KM = None

AUTO_RECTANGLE_QUANTILE_LOW = 0.01
AUTO_RECTANGLE_QUANTILE_HIGH = 0.99
AUTO_RECTANGLE_PADDING_KM = 25.0


def load_ingv_txt(path: Path) -> gpd.GeoDataFrame:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().lstrip("#").split("|")
        df = pd.read_csv(f, sep="|", names=header)

    df = df.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Magnitude": "mag",
            "Depth/Km": "depth",
            "Time": "time",
            "EventID": "event_id",
            "EventLocationName": "place",
            "EventType": "event_type",
        }
    )
    for col in ["latitude", "longitude", "mag", "depth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"])
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )


def italy_local_crs(gdf: gpd.GeoDataFrame) -> CRS:
    lon_0 = float(gdf.geometry.x.mean())
    lat_0 = float(gdf.geometry.y.mean())
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs"
    )


def principal_axis_rotation_degrees(x_m: np.ndarray, y_m: np.ndarray) -> float:
    coords = np.column_stack([x_m, y_m])
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return -angle


def rotate_xy(x: np.ndarray, y: np.ndarray, angle_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return cos_a * x - sin_a * y, sin_a * x + cos_a * y


def lonlat_to_rotated_km(
    lon: np.ndarray,
    lat: np.ndarray,
    local_crs: CRS,
    rotation_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    x_m, y_m = transformer.transform(lon, lat)
    x_rot_m, y_rot_m = rotate_xy(np.asarray(x_m), np.asarray(y_m), rotation_degrees)
    return x_rot_m / 1000.0, y_rot_m / 1000.0


def rotated_km_to_lonlat(
    x_rot_km: np.ndarray,
    y_rot_km: np.ndarray,
    local_crs: CRS,
    rotation_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_rot_m = np.asarray(x_rot_km) * 1000.0
    y_rot_m = np.asarray(y_rot_km) * 1000.0
    x_m, y_m = rotate_xy(x_rot_m, y_rot_m, -rotation_degrees)
    transformer = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x_m, y_m)
    return np.asarray(lon), np.asarray(lat)


def rectangle_bounds_from_args(
    args: argparse.Namespace,
    x_rot_km: np.ndarray,
    y_rot_km: np.ndarray,
) -> dict[str, float]:
    values = {
        "x_min_km": args.xmin_km,
        "x_max_km": args.xmax_km,
        "y_min_km": args.ymin_km,
        "y_max_km": args.ymax_km,
    }
    if all(value is not None for value in values.values()):
        return {key: float(value) for key, value in values.items()}
    if any(value is not None for value in values.values()):
        raise ValueError("Set all rectangle bounds or leave all bounds empty.")

    q_low = AUTO_RECTANGLE_QUANTILE_LOW
    q_high = AUTO_RECTANGLE_QUANTILE_HIGH
    pad = AUTO_RECTANGLE_PADDING_KM
    return {
        "x_min_km": float(np.quantile(x_rot_km, q_low) - pad),
        "x_max_km": float(np.quantile(x_rot_km, q_high) + pad),
        "y_min_km": float(np.quantile(y_rot_km, q_low) - pad),
        "y_max_km": float(np.quantile(y_rot_km, q_high) + pad),
    }


def load_land_for_plot(gdf: gpd.GeoDataFrame, pad_degrees: float = 5.0) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = gdf.total_bounds
    land = gpd.read_file(geodatasets.get_path("naturalearth land"))
    if land.crs is None:
        land = land.set_crs("EPSG:4326", allow_override=True)
    clipping_box = gpd.GeoDataFrame(
        geometry=[box(minx - pad_degrees, miny - pad_degrees, maxx + pad_degrees, maxy + pad_degrees)],
        crs="EPSG:4326",
    )
    return gpd.clip(land, clipping_box)


def set_plot_bounds(
    ax,
    x_values: pd.Series,
    y_values: pd.Series,
    bounds: dict[str, float],
    pad_fraction: float = 0.08,
) -> None:
    minx = min(float(x_values.min()), bounds["x_min_km"])
    maxx = max(float(x_values.max()), bounds["x_max_km"])
    miny = min(float(y_values.min()), bounds["y_min_km"])
    maxy = max(float(y_values.max()), bounds["y_max_km"])
    pad = max(maxx - minx, maxy - miny) * pad_fraction
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)


def plot_cut(
    gdf: gpd.GeoDataFrame,
    local_crs: CRS,
    rotation_degrees: float,
    bounds: dict[str, float],
    inside_mask: pd.Series,
    output_path: Path,
) -> None:
    land = load_land_for_plot(gdf).to_crs(local_crs)
    land_x = []
    for geom in land.geometry:
        rotated = rotate_geometry_to_rotated_km(geom, rotation_degrees)
        land_x.append(rotated)
    land_rotated = gpd.GeoDataFrame(geometry=land_x, crs=None)

    plot_gdf = gpd.GeoDataFrame(
        gdf.drop(columns="geometry"),
        geometry=gpd.points_from_xy(gdf["x_rot_km"], gdf["y_rot_km"]),
        crs=None,
    )
    selected = plot_gdf[inside_mask]
    outside = plot_gdf[~inside_mask]

    rect = Polygon(
        [
            (bounds["x_min_km"], bounds["y_min_km"]),
            (bounds["x_max_km"], bounds["y_min_km"]),
            (bounds["x_max_km"], bounds["y_max_km"]),
            (bounds["x_min_km"], bounds["y_max_km"]),
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor("#eaf3f8")
    land_rotated.plot(ax=ax, color="#eeeeea", edgecolor="#888888", linewidth=0.35)
    outside.plot(ax=ax, markersize=5, color="#777777", alpha=0.22, linewidth=0)
    selected.plot(ax=ax, markersize=8, color="#b4161b", alpha=0.55, linewidth=0)
    gpd.GeoSeries([rect]).boundary.plot(ax=ax, color="#111111", linewidth=1.3)
    set_plot_bounds(ax, plot_gdf["x_rot_km"], plot_gdf["y_rot_km"], bounds)
    ax.set_aspect("equal")
    ax.set_xlabel("rotated x (km)")
    ax.set_ylabel("rotated y (km)")
    ax.set_title(f"Rotated rectangle cut, angle {rotation_degrees:.1f} degrees")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def rotate_geometry_to_rotated_km(geom, rotation_degrees: float):
    from shapely import affinity

    rotated = affinity.rotate(geom, rotation_degrees, origin=(0, 0), use_radians=False)
    return affinity.scale(rotated, xfact=0.001, yfact=0.001, origin=(0, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut INGV Italy events with an axis-aligned rectangle in rotated coordinates."
    )
    parser.add_argument("--rotation", type=float, default=DEFAULT_ROTATION_DEGREES)
    parser.add_argument("--xmin-km", type=float, default=RECT_X_MIN_KM)
    parser.add_argument("--xmax-km", type=float, default=RECT_X_MAX_KM)
    parser.add_argument("--ymin-km", type=float, default=RECT_Y_MIN_KM)
    parser.add_argument("--ymax-km", type=float, default=RECT_Y_MAX_KM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PLOTS_DIR.mkdir(exist_ok=True)

    gdf = load_ingv_txt(DATA_FILE)
    local_crs = italy_local_crs(gdf)
    projected = gdf.to_crs(local_crs)
    x_m = projected.geometry.x.to_numpy()
    y_m = projected.geometry.y.to_numpy()

    rotation_degrees = args.rotation
    if rotation_degrees is None:
        rotation_degrees = principal_axis_rotation_degrees(x_m, y_m)

    x_rot_km, y_rot_km = lonlat_to_rotated_km(
        gdf["longitude"].to_numpy(),
        gdf["latitude"].to_numpy(),
        local_crs,
        rotation_degrees,
    )
    gdf["x_rot_km"] = x_rot_km
    gdf["y_rot_km"] = y_rot_km

    lon_back, lat_back = rotated_km_to_lonlat(x_rot_km, y_rot_km, local_crs, rotation_degrees)
    gdf["longitude_from_rotated"] = lon_back
    gdf["latitude_from_rotated"] = lat_back

    bounds = rectangle_bounds_from_args(args, x_rot_km, y_rot_km)
    inside_mask = (
        (gdf["x_rot_km"] >= bounds["x_min_km"])
        & (gdf["x_rot_km"] <= bounds["x_max_km"])
        & (gdf["y_rot_km"] >= bounds["y_min_km"])
        & (gdf["y_rot_km"] <= bounds["y_max_km"])
    )

    selected = gdf.loc[inside_mask].drop(columns="geometry")
    selected.to_csv(OUTPUT_EVENTS, index=False)

    meta = {
        "source_file": str(DATA_FILE),
        "output_events": str(OUTPUT_EVENTS),
        "rotation_degrees": float(rotation_degrees),
        "local_crs_wkt": local_crs.to_wkt(),
        "rectangle_bounds_km": bounds,
        "n_events_total": int(len(gdf)),
        "n_events_selected": int(inside_mask.sum()),
        "forward_transform": "EPSG:4326 lon/lat -> local AEQD meters -> rotate -> km",
        "inverse_transform": "rotated km -> inverse rotation -> local AEQD meters -> EPSG:4326 lon/lat",
        "roundtrip_max_abs_lon_error": float(np.max(np.abs(gdf["longitude"] - lon_back))),
        "roundtrip_max_abs_lat_error": float(np.max(np.abs(gdf["latitude"] - lat_back))),
    }
    with OUTPUT_META.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    plot_cut(gdf, local_crs, rotation_degrees, bounds, inside_mask, OUTPUT_PLOT)

    print(f"Loaded events: {len(gdf)}")
    print(f"Selected events: {int(inside_mask.sum())}")
    print(f"Applied rotation: {rotation_degrees:.2f} degrees")
    print(f"Rectangle bounds km: {bounds}")
    print(f"Roundtrip max lon error: {meta['roundtrip_max_abs_lon_error']:.3e}")
    print(f"Roundtrip max lat error: {meta['roundtrip_max_abs_lat_error']:.3e}")
    print(f"Saved events: {OUTPUT_EVENTS}")
    print(f"Saved metadata: {OUTPUT_META}")
    print(f"Saved plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
