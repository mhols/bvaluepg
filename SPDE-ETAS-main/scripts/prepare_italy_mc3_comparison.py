"""Build the fixed Mc=3 Italy comparison data set and rerun NND.

The script reuses the established projection and NND implementation from the
parent BvaluePG repository, but writes all new products inside SPDE-ETAS-main.
It also creates separate 20 km count grids for all and NND-kept events.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BVALUEPG_ROOT = PROJECT_ROOT.parent
SOURCE_SCRIPT = BVALUEPG_ROOT / "data" / "preprocess_nnd_rot_cut_bin.py"
OUTPUT_DIR = PROJECT_ROOT / "data" / "italy_mc3_comparison"

MC = 3.0
YEAR_MIN = 2015.0
YEAR_MAX = 2026.5
NND_D = 1.6
NND_B = 1.0
ETA_THRESHOLD_LOG10 = -4.6
RANDOM_SEED = 0
GAMMA_DEG = 41.75252483646699
ROTATION_DEGREES = -45.0
CELL_SIZE_KM = 20.0
BOUNDS_ROTATED_KM = (
    265.7071505562919,
    1269.7572914814305,
    -1414.1965686237359,
    -34.860497554831255,
)


def load_preprocessing_module():
    spec = importlib.util.spec_from_file_location("bvaluepg_italy_preprocessing", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import preprocessing module: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_grid(events):
    x_min, x_max, y_min, y_max = BOUNDS_ROTATED_KM
    x_edges = x_min + np.arange(int(np.ceil((x_max - x_min) / CELL_SIZE_KM)) + 1) * CELL_SIZE_KM
    y_edges = y_min + np.arange(int(np.ceil((y_max - y_min) / CELL_SIZE_KM)) + 1) * CELL_SIZE_KM
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    ix = np.digitize(events["x_rot_km"].to_numpy(float), x_edges) - 1
    iy = np.digitize(events["y_rot_km"].to_numpy(float), y_edges) - 1
    inside = events["inside_final_cut"].to_numpy(bool)
    in_grid = inside & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    result = events.copy()
    result["grid_ix"] = np.where(in_grid, ix, -1)
    result["grid_iy"] = np.where(in_grid, iy, -1)
    result["global_bin_id"] = np.where(in_grid, iy * nx + ix, -1)

    counts_all = np.zeros((ny, nx), dtype=int)
    counts_nnd = np.zeros((ny, nx), dtype=int)
    np.add.at(counts_all, (iy[in_grid], ix[in_grid]), 1)
    kept = in_grid & result["decluster_kept"].to_numpy(bool)
    np.add.at(counts_nnd, (iy[kept], ix[kept]), 1)
    return result, counts_all, counts_nnd, x_edges, y_edges


def main() -> None:
    pipeline = load_preprocessing_module()
    pipeline.MIN_MAGNITUDE = MC
    pipeline.MAX_MAGNITUDE = None
    pipeline.YEAR_MIN = YEAR_MIN
    pipeline.YEAR_MAX = YEAR_MAX
    pipeline.NND_D = NND_D
    pipeline.NND_B = NND_B
    pipeline.ETA_THRESHOLD_LOG10 = ETA_THRESHOLD_LOG10
    pipeline.RANDOM_SEED = RANDOM_SEED
    pipeline.GAMMA_DEG = GAMMA_DEG
    pipeline.ROTATION_DEGREES = ROTATION_DEGREES

    events = pipeline.load_catalog(pipeline.INPUT_FILE)
    events = pipeline.filter_catalog(events)
    events, projection_meta = pipeline.project_lonlat_to_xy_km(events)
    nnd = pipeline.run_nnd_declustering(pipeline.make_eqcat(events))
    events = pipeline.add_nnd_status(events, nnd)
    events = pipeline.add_rotated_coordinates(events)
    events = pipeline.add_cut_status(events, BOUNDS_ROTATED_KM)
    events, counts_all, counts_nnd, x_edges, y_edges = fixed_grid(events)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events_path = OUTPUT_DIR / "events.csv"
    grids_path = OUTPUT_DIR / "count_grids.npz"
    metadata_path = OUTPUT_DIR / "metadata.json"
    events.to_csv(events_path, index=False, sep="|")
    np.savez_compressed(
        grids_path,
        counts_all=counts_all,
        counts_nnd_kept=counts_nnd,
        x_edges=x_edges,
        y_edges=y_edges,
    )

    inside = events["inside_final_cut"].to_numpy(bool)
    kept_inside = inside & events["decluster_kept"].to_numpy(bool)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "matched_mc3_input_for_spde_etas_and_pg_comparison",
        "source": str(pipeline.INPUT_FILE.relative_to(BVALUEPG_ROOT)),
        "analysis_contract": {
            "minimum_magnitude": MC,
            "year_min": YEAR_MIN,
            "year_max": YEAR_MAX,
            "bounds_rotated_km": {
                "x_min": BOUNDS_ROTATED_KM[0],
                "x_max": BOUNDS_ROTATED_KM[1],
                "y_min": BOUNDS_ROTATED_KM[2],
                "y_max": BOUNDS_ROTATED_KM[3],
            },
            "projection_gamma_deg": GAMMA_DEG,
            "rotation_degrees": ROTATION_DEGREES,
            "cell_size_km": CELL_SIZE_KM,
            "grid_shape_ny_nx": list(counts_all.shape),
            "boundary_cells": "last row/column extend beyond analysis bounds; clipped area required for rate comparisons",
        },
        "nnd": {
            "implementation": "../data/src/clustering.py::NND_eta",
            "D": NND_D,
            "b": NND_B,
            "Mc": MC,
            "eta_threshold_log10": ETA_THRESHOLD_LOG10,
            "random_seed": RANDOM_SEED,
        },
        "projection": projection_meta,
        "event_counts": {
            "after_magnitude_time_filter": int(len(events)),
            "inside_analysis_window": int(inside.sum()),
            "nnd_kept_inside": int(kept_inside.sum()),
            "nnd_triggered_inside": int((inside & ~events["decluster_kept"].to_numpy(bool)).sum()),
            "counts_all_sum": int(counts_all.sum()),
            "counts_nnd_kept_sum": int(counts_nnd.sum()),
        },
        "outputs": {
            "events": str(events_path.relative_to(PROJECT_ROOT)),
            "count_grids": str(grids_path.relative_to(PROJECT_ROOT)),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    print(json.dumps(metadata["event_counts"], indent=2))
    print(f"Events: {events_path}")
    print(f"Count grids: {grids_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
