"""Create a preliminary, unit-matched SPDE-ETAS/PG Italy comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_DIR = PROJECT_ROOT / "mcmc_results" / "italy_pg_comparison"
DEFAULT_SPDE_DIR = PROJECT_ROOT / "mcmc_results" / "italy_full_100iter"
DEFAULT_GRID = PROJECT_ROOT / "data" / "italy_mc3_comparison" / "count_grids.npz"
DEFAULT_META = PROJECT_ROOT / "data" / "italy_mc3_comparison" / "metadata.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "mcmc_results" / "italy_method_comparison"
DAYS_PER_YEAR = 365.25


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg-dir", type=Path, default=DEFAULT_PG_DIR)
    parser.add_argument("--spde-dir", type=Path, default=DEFAULT_SPDE_DIR)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    return parser.parse_args()


def read_key_values(path: Path) -> dict[str, str]:
    values = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            key, value = line.rstrip("\n").split("=", 1)
            values[key] = value
    return values


def clipped_cell_areas(x_edges, y_edges, bounds) -> np.ndarray:
    widths = np.maximum(
        np.minimum(x_edges[1:], bounds["x_max"]) - np.maximum(x_edges[:-1], bounds["x_min"]),
        0.0,
    )
    heights = np.maximum(
        np.minimum(y_edges[1:], bounds["y_max"]) - np.maximum(y_edges[:-1], bounds["y_min"]),
        0.0,
    )
    return heights[:, None] * widths[None, :]


def rotated_to_lonlat(x_rot, y_rot, gamma_deg, rotation_deg, earth_radius):
    # Inverse of the preprocessing rotation followed by inverse equirectangular projection.
    angle = np.deg2rad(-rotation_deg)
    x_proj = np.cos(angle) * x_rot - np.sin(angle) * y_rot
    y_proj = np.sin(angle) * x_rot + np.cos(angle) * y_rot
    lon = np.rad2deg(x_proj / (earth_radius * np.cos(np.deg2rad(gamma_deg))))
    lat = gamma_deg + np.rad2deg(y_proj / earth_radius)
    return lon, lat


def field_summary(name, field, mask, x_centers, y_centers, metadata):
    values = field[mask]
    masked = np.where(mask, field, -np.inf)
    iy, ix = np.unravel_index(np.argmax(masked), field.shape)
    x = float(x_centers[ix])
    y = float(y_centers[iy])
    projection = metadata["projection"]
    contract = metadata["analysis_contract"]
    lon, lat = rotated_to_lonlat(
        x,
        y,
        float(projection["gamma_deg"]),
        float(contract["rotation_degrees"]),
        float(projection["EarthRadius"]),
    )
    return {
        "method": name,
        "minimum_events_per_km2_year": float(values.min()),
        "median_events_per_km2_year": float(np.median(values)),
        "maximum_events_per_km2_year": float(values.max()),
        "max_x_rot_km": x,
        "max_y_rot_km": y,
        "max_lon_deg": float(lon),
        "max_lat_deg": float(lat),
    }


def high_mask(field, mask, quantile):
    threshold = float(np.quantile(field[mask], quantile))
    return mask & (field >= threshold), threshold


def plot_fields(fields, x_edges, y_edges, output_path):
    positive = np.concatenate([values[np.isfinite(values) & (values > 0)] for values in fields.values()])
    vmin, vmax = np.quantile(positive, [0.01, 0.995])
    norm = LogNorm(vmin=max(float(vmin), 1e-8), vmax=float(vmax))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True, sharex=True, sharey=True)
    image = None
    for ax, (name, values) in zip(axes, fields.items()):
        image = ax.pcolormesh(x_edges, y_edges, values, shading="flat", cmap="viridis", norm=norm)
        ax.set_title(name)
        ax.set_xlabel("rotated x [km]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("rotated y [km]")
    fig.colorbar(image, ax=axes, label="background intensity [events / km² / year]")
    fig.suptitle("Shared log scale; colour limits use the pooled 1st–99.5th percentiles")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_high_regions(highs, x_edges, y_edges, quantile, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True, sharex=True, sharey=True)
    for ax, (name, values) in zip(axes, highs.items()):
        ax.pcolormesh(x_edges, y_edges, values.astype(float), shading="flat", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel("rotated x [km]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("rotated y [km]")
    fig.suptitle(f"Cells in the highest {(1.0 - quantile) * 100:g}% of background intensity")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    with np.load(args.grid) as grid:
        x_edges = grid["x_edges"].astype(float)
        y_edges = grid["y_edges"].astype(float)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    bounds = metadata["analysis_contract"]["bounds_rotated_km"]
    areas = clipped_cell_areas(x_edges, y_edges, bounds)
    nominal_area = float(metadata["analysis_contract"]["cell_size_km"]) ** 2
    full_cells = np.isclose(areas, nominal_area)

    spde_meta = read_key_values(args.spde_dir / "run_metadata.txt")
    elapsed_days = float(np.loadtxt(spde_meta["data_file"], usecols=0).max())
    if elapsed_days <= 0:
        raise ValueError("SPDE catalogue duration must be positive.")

    with np.load(args.pg_dir / "pg_all.npz") as data:
        pg_all_samples = data["rate_samples"] * DAYS_PER_YEAR / (areas[None, :, :] * elapsed_days)
    with np.load(args.pg_dir / "pg_nnd.npz") as data:
        pg_nnd_samples = data["rate_samples"] * DAYS_PER_YEAR / (areas[None, :, :] * elapsed_days)
    with np.load(args.spde_dir / "julia_expected_counts_on_pg_grid.npz") as data:
        spde_samples = data["expected_count_samples"] * DAYS_PER_YEAR / (areas[None, :, :] * elapsed_days)

    fields = {
        "SPDE-ETAS": np.mean(spde_samples, axis=0),
        "PG-all": np.mean(pg_all_samples, axis=0),
        "PG-NND": np.mean(pg_nnd_samples, axis=0),
    }
    sds = {
        "SPDE-ETAS": np.std(spde_samples, axis=0, ddof=1),
        "PG-all": np.std(pg_all_samples, axis=0, ddof=1),
        "PG-NND": np.std(pg_nnd_samples, axis=0, ddof=1),
    }
    summaries = [field_summary(name, values, full_cells, x_centers, y_centers, metadata) for name, values in fields.items()]
    highs = {}
    thresholds = {}
    for name, values in fields.items():
        highs[name], thresholds[name] = high_mask(values, full_cells, args.high_quantile)

    comparisons = []
    for other in ("PG-all", "PG-NND"):
        correlation = spearmanr(fields["SPDE-ETAS"][full_cells], fields[other][full_cells]).statistic
        intersection = int(np.count_nonzero(highs["SPDE-ETAS"] & highs[other]))
        union = int(np.count_nonzero(highs["SPDE-ETAS"] | highs[other]))
        comparisons.append(
            {
                "pair": f"SPDE-ETAS vs {other}",
                "spearman_full_cells": float(correlation),
                "high_region_intersection_cells": intersection,
                "high_region_union_cells": union,
                "high_region_jaccard": float(intersection / union) if union else float("nan"),
            }
        )

    np.savez_compressed(
        args.output_dir / "physical_intensity_fields.npz",
        spde_samples=spde_samples,
        pg_all_samples=pg_all_samples,
        pg_nnd_samples=pg_nnd_samples,
        spde_mean=fields["SPDE-ETAS"],
        pg_all_mean=fields["PG-all"],
        pg_nnd_mean=fields["PG-NND"],
        spde_sd=sds["SPDE-ETAS"],
        pg_all_sd=sds["PG-all"],
        pg_nnd_sd=sds["PG-NND"],
        clipped_cell_area_km2=areas,
        full_cell_mask=full_cells,
        x_edges=x_edges,
        y_edges=y_edges,
    )
    plot_fields(fields, x_edges, y_edges, args.output_dir / "01_background_intensity_comparison.png")
    plot_high_regions(highs, x_edges, y_edges, args.high_quantile, args.output_dir / "02_high_intensity_regions.png")

    mesh_points = np.loadtxt(args.spde_dir / "mesh_points.tsv")
    mesh_cells = np.loadtxt(args.spde_dir / "mesh_cells.tsv", dtype=int) - 1
    native_samples = np.atleast_2d(np.loadtxt(args.spde_dir / "chains_intensity.csv", delimiter=","))
    km_per_unit = float(spde_meta["km_per_model_unit"])
    mesh_x = mesh_points[:, 0] * km_per_unit + float(bounds["x_min"])
    mesh_y = mesh_points[:, 1] * km_per_unit + float(bounds["y_min"])
    native_mean = native_samples.mean(axis=0) * DAYS_PER_YEAR / km_per_unit**2
    triangulation = mtri.Triangulation(mesh_x, mesh_y, mesh_cells)
    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    artist = ax.tripcolor(triangulation, native_mean, shading="gouraud", cmap="viridis")
    ax.triplot(triangulation, color="white", linewidth=0.25, alpha=0.5)
    ax.set(title="SPDE-ETAS posterior mean on native mesh", xlabel="rotated x [km]", ylabel="rotated y [km]")
    ax.set_aspect("equal")
    fig.colorbar(artist, ax=ax, label="background intensity [events / km² / year]")
    fig.savefig(args.output_dir / "03_spde_native_mesh_intensity.png", dpi=180)
    plt.close(fig)

    with (args.output_dir / "method_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    def chain_total_diagnostic(samples):
        totals = samples.sum(axis=(1, 2))
        lag1 = float(np.corrcoef(totals[:-1], totals[1:])[0, 1]) if len(totals) > 2 else float("nan")
        return {
            "posterior_expected_total_mean": float(totals.mean()),
            "posterior_expected_total_sd": float(totals.std(ddof=1)),
            "posterior_expected_total_min": float(totals.min()),
            "posterior_expected_total_max": float(totals.max()),
            "total_lag1_correlation": lag1,
        }

    total_diagnostics = {
        "SPDE-ETAS": chain_total_diagnostic(spde_samples * areas[None, :, :] * elapsed_days / DAYS_PER_YEAR),
        "PG-all": chain_total_diagnostic(pg_all_samples * areas[None, :, :] * elapsed_days / DAYS_PER_YEAR),
        "PG-NND": chain_total_diagnostic(pg_nnd_samples * areas[None, :, :] * elapsed_days / DAYS_PER_YEAR),
    }
    total_diagnostics["SPDE-ETAS"]["observed_events"] = int(metadata["event_counts"]["counts_all_sum"])
    total_diagnostics["PG-all"]["observed_events"] = int(metadata["event_counts"]["counts_all_sum"])
    total_diagnostics["PG-NND"]["observed_events"] = int(metadata["event_counts"]["counts_nnd_kept_sum"])

    result = {
        "status": "preliminary_diagnostic_not_scientific_fit",
        "unit": "events per km2 per year",
        "elapsed_days": elapsed_days,
        "comparison_cells": int(full_cells.sum()),
        "excluded_partial_boundary_cells": int((areas > 0).sum() - full_cells.sum()),
        "high_quantile": args.high_quantile,
        "high_thresholds": thresholds,
        "method_summaries": summaries,
        "spatial_comparisons": comparisons,
        "total_diagnostics": total_diagnostics,
        "assumptions": [
            "PG rate is expected events per raster cell over the complete catalogue period.",
            "SPDE chain_intensity is events per model-area unit per day.",
            "One SPDE model length unit is 100 km.",
            "Partial boundary cells are shown but excluded from numerical comparisons.",
        ],
    }
    with (args.output_dir / "comparison_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    summary_by_method = {row["method"]: row for row in summaries}
    comparison_by_pair = {row["pair"]: row for row in comparisons}
    brief = f"""# Preliminary Italy background-intensity comparison

Status: diagnostic pilot for discussion, not a converged scientific result.

## Matched setup

- Catalogue: Mc = 3.0, 3,622 events in the common spatial window
- NND-kept catalogue: 1,694 events
- Period represented in the model: {elapsed_days:.3f} days
- Common display grid: 20 km; numerical comparison uses {int(full_cells.sum()):,} complete cells
- Unit used below: events per km2 per year
- PG runs: 500 iterations, 100 burn-in, thinning 4, 100 retained samples
- SPDE-ETAS run: 100 iterations, 10 burn-in, 90 retained samples

## Preliminary results

| Method | Minimum | Median | Maximum | Maximum lon | Maximum lat | Expected total | Observed input |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for name in ("SPDE-ETAS", "PG-all", "PG-NND"):
        row = summary_by_method[name]
        total = total_diagnostics[name]
        brief += (
            f"| {name} | {row['minimum_events_per_km2_year']:.3g} | "
            f"{row['median_events_per_km2_year']:.3g} | {row['maximum_events_per_km2_year']:.3g} | "
            f"{row['max_lon_deg']:.3f} | {row['max_lat_deg']:.3f} | "
            f"{total['posterior_expected_total_mean']:.1f} | {total['observed_events']} |\n"
        )
    spde_all = comparison_by_pair["SPDE-ETAS vs PG-all"]
    spde_nnd = comparison_by_pair["SPDE-ETAS vs PG-NND"]
    brief += f"""
## Spatial agreement

- SPDE-ETAS vs PG-all: Spearman = {spde_all['spearman_full_cells']:.3f}; top-5% Jaccard = {spde_all['high_region_jaccard']:.3f}
- SPDE-ETAS vs PG-NND: Spearman = {spde_nnd['spearman_full_cells']:.3f}; top-5% Jaccard = {spde_nnd['high_region_jaccard']:.3f}
- PG-NND is slightly closer to SPDE-ETAS by both preliminary measures.
- PG-all has an extreme maximum near lon {summary_by_method['PG-all']['max_lon_deg']:.3f}, lat {summary_by_method['PG-all']['max_lat_deg']:.3f}, consistent with a cluster dominating an all-event count grid.

## Questions for Sofiane

1. Is `chain_intensity` indeed measured in events per model-area unit per day?
2. Is the intended real-data mesh adaptive? The current script uses one global maximum triangle area (`a0.5`).
3. Should comparison use node intensity, interpolated intensity, or triangle-integrated intensity?
4. What chain length, number of chains and convergence criteria would you recommend?
5. How should we interpret the difference between the SPDE expected background total and sampled background allocations?

## Important limitations

- All values depend on the unit assumptions above; Sofiane should confirm them.
- The chains are short pilots. No R-hat can be calculated from one chain.
- The PG correlation length is 4 km, only 0.2 cells on this 20-km grid, so the PG maps are intentionally rough at this resolution.
- PG expected totals exceed their observed input totals, especially PG-all. Absolute levels therefore need model/prior calibration before scientific interpretation.
- Partial boundary cells are displayed but excluded from min/max, correlation and overlap calculations.

## Reproduce

From `SPDE-ETAS-main` with the PolyaGamma Python environment:

```bash
MPLBACKEND=Agg python scripts/run_pg_italy_comparison.py --n-iter 500 --burn-in 100 --thin 4
MPLBACKEND=Agg python scripts/compare_background_intensity.py
```
"""
    (args.output_dir / "meeting_brief.md").write_text(brief, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
