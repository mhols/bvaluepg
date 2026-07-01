from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from italy_rotated_workflow import (
    DATA_FILE,
    DEFAULT_ROTATION_DEGREES,
    load_ingv_txt,
    load_land_for_plot,
    lonlat_to_rotated_km,
    italy_local_crs,
    principal_axis_rotation_degrees,
    rectangle_bounds,
    rotated_km_to_lonlat,
    rotate_geometry_to_rotated_km,
    marker_sizes,
    set_bounds,
    set_plot_bounds,
)


def main() -> None:
    print(f"Loading Italy data from: {Path(DATA_FILE).name}")
    gdf = load_ingv_txt(DATA_FILE)
    land = load_land_for_plot(gdf)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#eaf3f8")
    land.plot(ax=ax, color="#eeeeea", edgecolor="#888888", linewidth=0.4)
    gdf.plot(ax=ax, markersize=marker_sizes(gdf), color="#b4161b", alpha=0.45, linewidth=0)
    set_bounds(ax, tuple(gdf.total_bounds), pad_fraction=0.08)
    ax.set_aspect("equal")
    ax.set_title("1) Original INGV Italy events")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()

    local_crs = italy_local_crs(gdf)
    projected = gdf.to_crs(local_crs)
    x_m = projected.geometry.x.to_numpy()
    y_m = projected.geometry.y.to_numpy()

    rotation_degrees = DEFAULT_ROTATION_DEGREES
    if rotation_degrees is None:
        rotation_degrees = principal_axis_rotation_degrees(x_m, y_m)
    print(f"Rotation angle: {rotation_degrees:.2f} degrees")

    x_rot_km, y_rot_km = lonlat_to_rotated_km(
        gdf["longitude"].to_numpy(),
        gdf["latitude"].to_numpy(),
        local_crs,
        rotation_degrees,
    )
    gdf["x_rot_km"] = x_rot_km
    gdf["y_rot_km"] = y_rot_km

    land_rotated = land.to_crs(local_crs).copy()
    land_rotated["geometry"] = land_rotated.geometry.apply(
        lambda geom: rotate_geometry_to_rotated_km(geom, rotation_degrees)
    )
    rotated_gdf = gpd.GeoDataFrame(
        gdf.drop(columns="geometry"),
        geometry=gpd.points_from_xy(gdf["x_rot_km"], gdf["y_rot_km"]),
        crs=None,
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor("#eaf3f8")
    land_rotated.plot(ax=ax, color="#eeeeea", edgecolor="#888888", linewidth=0.35)
    rotated_gdf.plot(ax=ax, markersize=marker_sizes(gdf), color="#b4161b", alpha=0.45, linewidth=0)
    ax.set_aspect("equal")
    ax.set_title(f"2) Rotated Italy events ({rotation_degrees:.1f} degrees)")
    ax.set_xlabel("rotated x (km)")
    ax.set_ylabel("rotated y (km)")
    fig.tight_layout()

    bounds = rectangle_bounds(x_rot_km, y_rot_km)
    inside_mask = (
        (gdf["x_rot_km"] >= bounds["x_min_km"])
        & (gdf["x_rot_km"] <= bounds["x_max_km"])
        & (gdf["y_rot_km"] >= bounds["y_min_km"])
        & (gdf["y_rot_km"] <= bounds["y_max_km"])
    )
    selected_rotated = rotated_gdf[inside_mask]
    outside_rotated = rotated_gdf[~inside_mask]
    print(f"Selected events: {len(selected_rotated)} of {len(gdf)}")
    print(f"Rectangle bounds km: {bounds}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor("#eaf3f8")
    land_rotated.plot(ax=ax, color="#eeeeea", edgecolor="#888888", linewidth=0.35)
    outside_rotated.plot(ax=ax, markersize=5, color="#777777", alpha=0.22, linewidth=0)
    selected_rotated.plot(ax=ax, markersize=8, color="#b4161b", alpha=0.55, linewidth=0)
    rectangle = gpd.GeoSeries(
        [
            Polygon(
                [
                    (bounds["x_min_km"], bounds["y_min_km"]),
                    (bounds["x_max_km"], bounds["y_min_km"]),
                    (bounds["x_max_km"], bounds["y_max_km"]),
                    (bounds["x_min_km"], bounds["y_max_km"]),
                ]
            )
        ]
    )
    rectangle.boundary.plot(ax=ax, color="#111111", linewidth=1.3)
    set_plot_bounds(ax, rotated_gdf["x_rot_km"], rotated_gdf["y_rot_km"], bounds)
    ax.set_aspect("equal")
    ax.set_title("3) Rotated rectangle cut")
    ax.set_xlabel("rotated x (km)")
    ax.set_ylabel("rotated y (km)")
    fig.tight_layout()

    selected = gdf[inside_mask].copy()
    lon_back, lat_back = rotated_km_to_lonlat(
        selected["x_rot_km"].to_numpy(),
        selected["y_rot_km"].to_numpy(),
        local_crs,
        rotation_degrees,
    )
    selected["longitude_from_rotated"] = lon_back
    selected["latitude_from_rotated"] = lat_back
    selected_back = gpd.GeoDataFrame(
        selected.drop(columns="geometry"),
        geometry=gpd.points_from_xy(selected["longitude_from_rotated"], selected["latitude_from_rotated"]),
        crs="EPSG:4326",
    )

    max_lon_error = (selected["longitude"] - selected["longitude_from_rotated"]).abs().max()
    max_lat_error = (selected["latitude"] - selected["latitude_from_rotated"]).abs().max()
    print(f"Roundtrip max lon error: {max_lon_error:.3e}")
    print(f"Roundtrip max lat error: {max_lat_error:.3e}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#eaf3f8")
    land.plot(ax=ax, color="#eeeeea", edgecolor="#888888", linewidth=0.4)
    selected_back.plot(ax=ax, markersize=8, color="#2468a2", alpha=0.55, linewidth=0)
    set_bounds(ax, tuple(selected_back.total_bounds), pad_fraction=0.12)
    ax.set_aspect("equal")
    ax.set_title("4) Back-transformed selected events")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()


    plt.show()


if __name__ == "__main__":
    main()
