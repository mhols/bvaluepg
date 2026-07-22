from __future__ import annotations

"""Diagnostics for the new NND-before-cut Italy preprocessing pipeline.

The script only reads existing pipeline outputs and the previous cut-before-NND
catalogue. Figures are displayed with ``plt.show()`` and are never saved.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
# Input parameters. Change these paths here before running the script.
INPUT_PREFIX = DATA_DIR / "italy_nnd_rotate_cut_grid_Mc_2.5_eta_-4.60"
OLD_CATALOG = DATA_DIR / "italy_ingv_rotated_rect_events_declustered_Mc_2.5_eta_-4.60.csv"


def related_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_inputs(prefix: Path, old_catalog_path: Path):
    events = pd.read_csv(related_path(prefix, "_events.csv"), low_memory=False)
    old = pd.read_csv(old_catalog_path, low_memory=False)
    with np.load(related_path(prefix, "_counts.npz")) as data:
        new_counts = data["counts"].astype(int)
        x_edges = data["x_edges"]
        y_edges = data["y_edges"]
    with related_path(prefix, "_meta.json").open("r", encoding="utf-8") as stream:
        meta = json.load(stream)

    required_new = {
        "event_id",
        "x_rot_km",
        "y_rot_km",
        "nnd_log10_eta",
        "nnd_is_triggered",
        "decluster_kept",
        "inside_final_cut",
    }
    required_old = {"event_id", "x_rot_km", "y_rot_km", "kept"}
    if missing := required_new.difference(events.columns):
        raise ValueError(f"New event file is missing columns: {sorted(missing)}")
    if missing := required_old.difference(old.columns):
        raise ValueError(f"Old catalogue is missing columns: {sorted(missing)}")
    if list(new_counts.shape) != meta["grid"]["shape_ny_nx"]:
        raise ValueError("Count-grid shape does not match metadata")

    events["decluster_kept"] = boolean_series(events["decluster_kept"])
    events["nnd_is_triggered"] = boolean_series(events["nnd_is_triggered"])
    events["inside_final_cut"] = boolean_series(events["inside_final_cut"])
    old["kept"] = boolean_series(old["kept"])
    old_kept = old[old["kept"]].copy()
    old_counts_xy, _, _ = np.histogram2d(
        old_kept["x_rot_km"],
        old_kept["y_rot_km"],
        bins=[x_edges, y_edges],
    )
    old_counts = old_counts_xy.T.astype(int)
    return events, old, new_counts, old_counts, x_edges, y_edges, meta


def comparison_table(events: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    new_inside = events.loc[events["inside_final_cut"], ["event_id", "decluster_kept"]].copy()
    old_status = old[["event_id", "kept"]].copy()
    new_inside["event_id"] = new_inside["event_id"].astype(str)
    old_status["event_id"] = old_status["event_id"].astype(str)
    merged = new_inside.merge(old_status, on="event_id", how="inner", validate="one_to_one")
    merged = merged.rename(columns={"decluster_kept": "new_kept", "kept": "old_kept"})
    return merged


def print_summary(
    events: pd.DataFrame,
    old: pd.DataFrame,
    new_counts: np.ndarray,
    old_counts: np.ndarray,
    matched: pd.DataFrame,
    meta: dict,
) -> None:
    both = int((matched["new_kept"] & matched["old_kept"]).sum())
    new_only = int((matched["new_kept"] & ~matched["old_kept"]).sum())
    old_only = int((~matched["new_kept"] & matched["old_kept"]).sum())
    neither = int((~matched["new_kept"] & ~matched["old_kept"]).sum())
    print("\nItaly NND pipeline diagnostics")
    print("=" * 40)
    print(f"NND threshold log10(eta): {meta['nnd']['eta_threshold_log10']:.2f}")
    print(f"generous catalogue events: {len(events)}")
    print(f"events inside final cut: {int(events['inside_final_cut'].sum())}")
    print(f"new NND-before-cut background events: {int(new_counts.sum())}")
    print(f"old cut-before-NND background events: {int(old_counts.sum())}")
    print(f"matched final-cut events: {len(matched)} of {len(old)}")
    print("\nEvent-level kept-status comparison")
    print(f"kept by both: {both}")
    print(f"new only:     {new_only}")
    print(f"old only:     {old_only}")
    print(f"kept by neither: {neither}")
    print(f"status changes: {new_only + old_only}")
    print("\nCount-grid comparison")
    print(f"cells with changed counts: {int((new_counts != old_counts).sum())}")
    print(f"maximum absolute cell difference: {int(np.abs(new_counts - old_counts).max())}")
    print(f"L1 difference across cells: {int(np.abs(new_counts - old_counts).sum())}")


def plot_nnd_diagnostics(events: pd.DataFrame, threshold: float) -> None:
    eta = events["nnd_log10_eta"].dropna().to_numpy()
    inside = events[events["inside_final_cut"]]
    kept = inside[inside["decluster_kept"]]
    triggered = inside[~inside["decluster_kept"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    axes[0].hist(eta, bins=np.arange(np.floor(eta.min()), np.ceil(eta.max()) + 0.2, 0.2), color="0.4")
    axes[0].axvline(threshold, color="C3", linestyle="--", linewidth=2, label=f"threshold {threshold:.2f}")
    axes[0].set_title("NND distribution before final cut")
    axes[0].set_xlabel("log10(eta)")
    axes[0].set_ylabel("event pairs")
    axes[0].legend()

    axes[1].scatter(triggered["x_rot_km"], triggered["y_rot_km"], s=3, color="0.65", alpha=0.45, label=f"triggered ({len(triggered)})")
    axes[1].scatter(kept["x_rot_km"], kept["y_rot_km"], s=4, color="C0", alpha=0.65, label=f"kept ({len(kept)})")
    axes[1].set_title("New NND-before-cut classification")
    axes[1].set_xlabel("rotated x [km]")
    axes[1].set_ylabel("rotated y [km]")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(markerscale=3)


def plot_grid_comparison(new_counts, old_counts, x_edges, y_edges) -> None:
    difference = new_counts - old_counts
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    limit = max(1, int(np.abs(difference).max()))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    panels = (
        (old_counts, "Old: cut then NND", "magma", None),
        (new_counts, "New: NND then cut", "magma", None),
        (difference, "New minus old counts", "coolwarm", TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)),
    )
    vmax = max(int(old_counts.max()), int(new_counts.max()))
    for ax, (image, title, cmap, norm) in zip(axes, panels):
        kwargs = {"norm": norm} if norm is not None else {"vmin": 0, "vmax": vmax}
        artist = ax.imshow(image, origin="lower", extent=extent, aspect="auto", interpolation="nearest", cmap=cmap, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("rotated x [km]")
        ax.set_ylabel("rotated y [km]")
        fig.colorbar(artist, ax=ax, label="events per cell")


def plot_count_distributions(new_counts: np.ndarray, old_counts: np.ndarray) -> None:
    old_flat = old_counts.ravel()
    new_flat = new_counts.ravel()
    max_count = int(max(old_flat.max(), new_flat.max()))
    bins = np.arange(-0.5, max_count + 1.5, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist(old_flat, bins=bins, histtype="step", linewidth=2, label="old cut then NND")
    axes[0].hist(new_flat, bins=bins, histtype="step", linewidth=2, label="new NND then cut")
    axes[0].set_yscale("log")
    axes[0].set_title("All cells, including zeros")
    axes[0].set_xlabel("count per cell")
    axes[0].set_ylabel("number of cells (log scale)")
    axes[0].legend()

    axes[1].scatter(old_flat, new_flat, s=8, alpha=0.25)
    axes[1].plot([0, max_count], [0, max_count], color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Cell-wise count comparison")
    axes[1].set_xlabel("old count")
    axes[1].set_ylabel("new count")
    axes[1].set_aspect("equal", adjustable="box")


def main() -> None:
    events, old, new_counts, old_counts, x_edges, y_edges, meta = load_inputs(INPUT_PREFIX, OLD_CATALOG)
    matched = comparison_table(events, old)
    print_summary(events, old, new_counts, old_counts, matched, meta)
    plot_nnd_diagnostics(events, float(meta["nnd"]["eta_threshold_log10"]))
    plot_grid_comparison(new_counts, old_counts, x_edges, y_edges)
    plot_count_distributions(new_counts, old_counts)
    print("\nFigures are shown interactively and are not saved.")
    plt.show()


if __name__ == "__main__":
    main()
