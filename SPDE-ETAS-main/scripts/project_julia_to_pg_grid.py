"""Project SPDE mesh intensity samples onto the fixed 20 km PG grid.

The SPDE field is piecewise linear on the triangle mesh.  This script integrates
that field exactly over every triangle/cell intersection.  Boundary cells are
clipped to the fixed analysis window before integration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "mcmc_results" / "italy_full_technical"
DEFAULT_GRID_FILE = PROJECT_ROOT / "data" / "italy_mc3_comparison" / "count_grids.npz"
DEFAULT_META_FILE = PROJECT_ROOT / "data" / "italy_mc3_comparison" / "metadata.json"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--grid-file", type=Path, default=DEFAULT_GRID_FILE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META_FILE)
    return parser.parse_args()


def read_run_metadata(path: Path) -> dict[str, str]:
    result = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            key, value = line.rstrip("\n").split("=", 1)
            result[key] = value
    return result


def clip_polygon(vertices, axis: int, boundary: float, keep_greater: bool):
    if not vertices:
        return []

    def inside(vertex) -> bool:
        value = vertex[0][axis]
        return value >= boundary - 1e-10 if keep_greater else value <= boundary + 1e-10

    output = []
    previous = vertices[-1]
    previous_inside = inside(previous)
    for current in vertices:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[0][axis] - previous[0][axis]
            fraction = 0.0 if abs(denominator) < 1e-15 else (boundary - previous[0][axis]) / denominator
            point = previous[0] + fraction * (current[0] - previous[0])
            weights = previous[1] + fraction * (current[1] - previous[1])
            output.append((point, weights))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output


def triangle_rectangle_integral_weights(triangle, left, right, bottom, top):
    vertices = [
        (triangle[0], np.array([1.0, 0.0, 0.0])),
        (triangle[1], np.array([0.0, 1.0, 0.0])),
        (triangle[2], np.array([0.0, 0.0, 1.0])),
    ]
    vertices = clip_polygon(vertices, 0, left, True)
    vertices = clip_polygon(vertices, 0, right, False)
    vertices = clip_polygon(vertices, 1, bottom, True)
    vertices = clip_polygon(vertices, 1, top, False)
    if len(vertices) < 3:
        return np.zeros(3)

    integral_weights = np.zeros(3)
    point0, weights0 = vertices[0]
    for index in range(1, len(vertices) - 1):
        point1, weights1 = vertices[index]
        point2, weights2 = vertices[index + 1]
        edge1 = point1 - point0
        edge2 = point2 - point0
        area = abs(edge1[0] * edge2[1] - edge1[1] * edge2[0]) / 2.0
        integral_weights += area * (weights0 + weights1 + weights2) / 3.0
    return integral_weights


def main() -> None:
    args = arguments()
    run_dir = args.run_dir.resolve()
    run_meta = read_run_metadata(run_dir / "run_metadata.txt")
    input_catalog = Path(run_meta["data_file"])
    elapsed_days = float(np.loadtxt(input_catalog, usecols=0).max())

    mesh_points = np.loadtxt(run_dir / "mesh_points.tsv")
    mesh_cells = np.loadtxt(run_dir / "mesh_cells.tsv", dtype=int) - 1
    intensity_samples = np.loadtxt(run_dir / "chains_intensity.csv", delimiter=",")
    intensity_samples = np.atleast_2d(intensity_samples)
    if intensity_samples.shape[1] != len(mesh_points):
        raise ValueError("Intensity sample width does not match mesh point count.")
    mesh_points_km = mesh_points * float(run_meta["km_per_model_unit"])

    with np.load(args.grid_file) as grid:
        counts_all = grid["counts_all"]
        counts_nnd = grid["counts_nnd_kept"]
        x_edges = grid["x_edges"]
        y_edges = grid["y_edges"]
    with args.metadata.open("r", encoding="utf-8") as stream:
        comparison_meta = json.load(stream)
    bounds = comparison_meta["analysis_contract"]["bounds_rotated_km"]
    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["y_min"])
    y_max = float(bounds["y_max"])
    km_per_model_unit = float(run_meta["km_per_model_unit"])
    mesh_points_km[:, 0] += x_min
    mesh_points_km[:, 1] += y_min
    triangles_km = mesh_points_km[mesh_cells]
    triangle_x_min = triangles_km[:, :, 0].min(axis=1)
    triangle_x_max = triangles_km[:, :, 0].max(axis=1)
    triangle_y_min = triangles_km[:, :, 1].min(axis=1)
    triangle_y_max = triangles_km[:, :, 1].max(axis=1)

    ny, nx = counts_all.shape
    expected_samples = np.full((len(intensity_samples), ny, nx), np.nan)
    rows = []
    for iy in range(ny):
        for ix in range(nx):
            left = max(float(x_edges[ix]), x_min)
            right = min(float(x_edges[ix + 1]), x_max)
            bottom = max(float(y_edges[iy]), y_min)
            top = min(float(y_edges[iy + 1]), y_max)
            area_km2 = max(right - left, 0.0) * max(top - bottom, 0.0)
            if area_km2 == 0.0:
                rows.append((ix, iy, left, right, bottom, top, area_km2, np.nan, np.nan))
                continue

            mesh_integral_weights_km2 = np.zeros(len(mesh_points))
            candidates = np.flatnonzero(
                (triangle_x_max >= left)
                & (triangle_x_min <= right)
                & (triangle_y_max >= bottom)
                & (triangle_y_min <= top)
            )
            for triangle_index in candidates:
                local_weights = triangle_rectangle_integral_weights(
                    triangles_km[triangle_index], left, right, bottom, top
                )
                mesh_integral_weights_km2[mesh_cells[triangle_index]] += local_weights
            integrated_intensity_model2 = (
                intensity_samples @ mesh_integral_weights_km2 / km_per_model_unit**2
            )
            expected = integrated_intensity_model2 * elapsed_days
            expected_samples[:, iy, ix] = expected
            rows.append(
                (
                    ix,
                    iy,
                    left,
                    right,
                    bottom,
                    top,
                    area_km2,
                    float(expected.mean()),
                    float(expected.std(ddof=1)) if len(expected) > 1 else 0.0,
                )
            )

    output_npz = run_dir / "julia_expected_counts_on_pg_grid.npz"
    output_csv = run_dir / "julia_pg_grid_summary.csv"
    np.savez_compressed(
        output_npz,
        expected_count_samples=expected_samples,
        expected_count_mean=np.nanmean(expected_samples, axis=0),
        expected_count_sd=np.nanstd(expected_samples, axis=0, ddof=1),
        observed_all=counts_all,
        observed_nnd_kept=counts_nnd,
        x_edges=x_edges,
        y_edges=y_edges,
    )
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "grid_ix",
                "grid_iy",
                "x_left_rot_km",
                "x_right_clipped_rot_km",
                "y_bottom_rot_km",
                "y_top_clipped_rot_km",
                "clipped_area_km2",
                "julia_expected_count_mean",
                "julia_expected_count_sd",
                "observed_all",
                "observed_nnd_kept",
            ]
        )
        for row in rows:
            ix, iy = row[0], row[1]
            writer.writerow((*row, int(counts_all[iy, ix]), int(counts_nnd[iy, ix])))

    print(f"Grid: {nx} x {ny}; Julia samples: {len(intensity_samples)}")
    print(f"Elapsed days: {elapsed_days:.6f}")
    print(f"Expected-count samples: {output_npz}")
    print(f"Cell summary: {output_csv}")


if __name__ == "__main__":
    main()
