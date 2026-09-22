"""Run matched PG-all and PG-NND pilots for the Mc=3 Italy comparison.

The saved PG rates are expected event counts per grid cell over the complete
catalogue period.  Conversion to physical intensity is deliberately handled by
``compare_background_intensity.py``, where clipped boundary-cell areas are
available as well.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BVALUEPG_ROOT = PROJECT_ROOT.parent
SOURCE_DIR = BVALUEPG_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from covariance_kernels import precision_matern
from polyagammadensity import PolyaGammaDensity2D, inv_sigmoid


DEFAULT_INPUT = PROJECT_ROOT / "data" / "italy_mc3_comparison" / "count_grids.npz"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "italy_mc3_comparison" / "metadata.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "mcmc_results" / "italy_pg_comparison"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--thin", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--rho-km", type=float, default=4.0)
    parser.add_argument("--prior-variance", type=float, default=1.0)
    parser.add_argument("--boundary", choices=("zero", "symmetric"), default="symmetric")
    return parser.parse_args()


def choose_lambda(counts: np.ndarray) -> float:
    p995 = float(np.percentile(counts, 99.5))
    return float(max(int(counts.max()) + 2, np.ceil(1.35 * p995), 1.0))


def run_one(
    label: str,
    counts: np.ndarray,
    cell_size_km: float,
    args: argparse.Namespace,
) -> dict:
    ny, nx = counts.shape
    lam = choose_lambda(counts)
    mean_count = float(counts.mean())
    prior_probability = float(np.clip(mean_count / lam, 1e-6, 1.0 - 1e-6))
    prior_mean = float(inv_sigmoid(prior_probability))

    # rho is specified in kilometres, while the precision builder works in
    # grid-cell units.  For the 20 km grid, rho=4 km therefore means 0.2 cells.
    rho_cells = float(args.rho_km / cell_size_km)
    precision = precision_matern(
        n=ny,
        m=nx,
        rho=rho_cells,
        v2=args.prior_variance,
        boundary=args.boundary,
    )
    model = PolyaGammaDensity2D(
        prior_mean=np.full(ny * nx, prior_mean),
        prior_precision=precision,
        sparse=True,
        lam=lam,
        n=ny,
        m=nx,
    )
    model.set_nobs(counts.ravel(order="C"))

    f_samples = []
    rate_samples = []
    started = time.perf_counter()
    for f_sample in model.sample_posterior(
        n_iter=args.n_iter,
        burn_in=args.burn_in,
        thin=args.thin,
        initial_f=np.full(ny * nx, prior_mean),
        random_seed=args.seed,
    ):
        f_samples.append(model.scanorder_to_image(f_sample))
        rate_samples.append(model.scanorder_to_image(model.field_from_f(f_sample)))
    elapsed_seconds = time.perf_counter() - started

    if not rate_samples:
        raise ValueError("No posterior samples retained; check n-iter, burn-in and thin.")
    f_samples = np.asarray(f_samples)
    rate_samples = np.asarray(rate_samples)
    output_path = args.output_dir / f"pg_{label}.npz"
    np.savez_compressed(
        output_path,
        f_samples=f_samples,
        rate_samples=rate_samples,
        rate_mean=rate_samples.mean(axis=0),
        rate_sd=rate_samples.std(axis=0, ddof=1) if len(rate_samples) > 1 else np.zeros_like(rate_samples[0]),
        counts=counts,
    )
    return {
        "label": label,
        "events": int(counts.sum()),
        "nonzero_cells": int((counts > 0).sum()),
        "lambda": lam,
        "prior_mean": prior_mean,
        "rho_km": float(args.rho_km),
        "rho_cells": rho_cells,
        "retained_samples": int(len(rate_samples)),
        "elapsed_seconds": elapsed_seconds,
        "output": str(output_path.resolve()),
    }


def main() -> None:
    args = arguments()
    if not (0 <= args.burn_in < args.n_iter):
        raise ValueError("Require 0 <= burn-in < n-iter.")
    if args.thin < 1:
        raise ValueError("thin must be positive.")

    with np.load(args.input) as grid:
        counts_all = grid["counts_all"].astype(int)
        counts_nnd = grid["counts_nnd_kept"].astype(int)
        x_edges = grid["x_edges"].astype(float)
        y_edges = grid["y_edges"].astype(float)
    with args.metadata.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    expected_shape = tuple(metadata["analysis_contract"]["grid_shape_ny_nx"])
    if counts_all.shape != expected_shape or counts_nnd.shape != expected_shape:
        raise ValueError("Count grids do not match the shape in metadata.json.")
    cell_size_km = float(metadata["analysis_contract"]["cell_size_km"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "grid.npz", x_edges=x_edges, y_edges=y_edges)
    summaries = [
        run_one("all", counts_all, cell_size_km, args),
        run_one("nnd", counts_nnd, cell_size_km, args),
    ]
    run_metadata = {
        "purpose": "preliminary_matched_pg_spde_comparison",
        "input": str(args.input.resolve()),
        "n_iter": args.n_iter,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "seed": args.seed,
        "prior_variance": args.prior_variance,
        "boundary": args.boundary,
        "cell_size_km": cell_size_km,
        "runs": summaries,
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2)
        stream.write("\n")
    print(json.dumps(run_metadata, indent=2))


if __name__ == "__main__":
    main()
