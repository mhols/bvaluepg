import os
import argparse
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bvaluepg_mplconfig")

import geodatasets
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import affinity

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PLOTS_DIR = REPO_ROOT / "plots"
DATA_FILE = SCRIPT_DIR / "italy_ingv_m2point5_2015-2026.txt"

NORMAL_PLOT = PLOTS_DIR / "italy_ingv_points_map.png"
ROTATED_PLOT = PLOTS_DIR / "italy_ingv_points_rotated_map.png"

# Set to a number such as 60.0 to fix the angle in the script.
# Keep as None to compute the angle automatically from the point cloud.
DEFAULT_ROTATION_DEGREES = -55.0


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


def marker_sizes(gdf: gpd.GeoDataFrame) -> pd.Series:
    mag = pd.to_numeric(gdf["mag"], errors="coerce")
    scaled = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
    return 4 + 28 * scaled.fillna(0)


def italy_local_crs(gdf: gpd.GeoDataFrame) -> CRS:
    lon_0 = float(gdf.geometry.x.mean())
    lat_0 = float(gdf.geometry.y.mean())
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs"
    )


def principal_axis_rotation_degrees(gdf_projected: gpd.GeoDataFrame) -> float:
    coords = np.column_stack([gdf_projected.geometry.x, gdf_projected.geometry.y])
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return -angle


def rotate_geometries(gdf: gpd.GeoDataFrame, angle_degrees: float) -> gpd.GeoDataFrame:
    rotated = gdf.copy()
    rotated["geometry"] = rotated.geometry.apply(
        lambda geom: affinity.rotate(geom, angle_degrees, origin=(0, 0), use_radians=False)
    )
    return rotated


def load_land_for_italy(gdf: gpd.GeoDataFrame, pad_degrees: float = 5.0) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = gdf.total_bounds
    world = gpd.read_file(geodatasets.get_path("naturalearth land"))
    if world.crs is None:
        world = world.set_crs("EPSG:4326", allow_override=True)
    return world.cx[
        minx - pad_degrees : maxx + pad_degrees,
        miny - pad_degrees : maxy + pad_degrees,
    ]


def set_bounds(ax, bounds: tuple[float, float, float, float], pad_fraction: float = 0.06) -> None:
    minx, miny, maxx, maxy = bounds
    dx = maxx - minx
    dy = maxy - miny
    pad = max(dx, dy) * pad_fraction
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)


def plot_lonlat(gdf: gpd.GeoDataFrame, land: gpd.GeoDataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#eaf3f8")
    land.plot(ax=ax, color="#eeeeea", edgecolor="#888888", linewidth=0.4)
    gdf.plot(ax=ax, markersize=marker_sizes(gdf), color="#b4161b", alpha=0.45, linewidth=0)
    set_bounds(ax, tuple(gdf.total_bounds), pad_fraction=0.08)
    ax.set_aspect("equal")
    ax.set_title("INGV Italy earthquakes, lon/lat")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_rotated(
    gdf: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame,
    output_path: Path,
    rotation_degrees: float | None = None,
) -> float:
    local_crs = italy_local_crs(gdf)
    points_local = gdf.to_crs(local_crs)
    land_local = land.to_crs(local_crs)

    if rotation_degrees is None:
        rotation_degrees = principal_axis_rotation_degrees(points_local)
    points_rotated = rotate_geometries(points_local, rotation_degrees)
    land_rotated = rotate_geometries(land_local, rotation_degrees)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_facecolor("#eaf3f8")
    land_rotated.plot(ax=ax, color="#eeeeea", edgecolor="#888888", linewidth=0.35)
    points_rotated.plot(
        ax=ax,
        markersize=marker_sizes(points_rotated),
        color="#b4161b",
        alpha=0.45,
        linewidth=0,
    )
    set_bounds(ax, tuple(points_rotated.total_bounds), pad_fraction=0.08)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"INGV Italy earthquakes, rotated {rotation_degrees:.1f} degrees")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return rotation_degrees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot INGV Italy earthquake points on a normal and rotated map."
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=DEFAULT_ROTATION_DEGREES,
        help=(
            "Manual rotation angle in degrees. "
            "Default: use DEFAULT_ROTATION_DEGREES or compute angle from point PCA."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PLOTS_DIR.mkdir(exist_ok=True)
    gdf = load_ingv_txt(DATA_FILE)
    land = load_land_for_italy(gdf)

    plot_lonlat(gdf, land, NORMAL_PLOT)
    rotation_degrees = plot_rotated(gdf, land, ROTATED_PLOT, args.rotation)

    print(f"Loaded events: {len(gdf)}")
    bounds = tuple(round(float(v), 4) for v in gdf.total_bounds)
    print(f"Lon/lat bounds: {bounds}")
    print(f"Applied rotation: {rotation_degrees:.2f} degrees")
    print(f"Saved: {NORMAL_PLOT}")
    print(f"Saved: {ROTATED_PLOT}")


if __name__ == "__main__":
    main()
