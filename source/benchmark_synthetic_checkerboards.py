"""
Benchmark Polya-Gamma estimator on synthetic checkerboard data.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# --- configuration (edit here) ------------------------------------------------

# Grid structure: total grid size is n = nn * ncheck (square grid)
NN_LIST = [4, 8, 16, 24]  # block size (cells per checkerboard square)
NCHECK_LIST = [4, 6]  # number of checkerboard blocks per axis

# Intensity scale: larger lam should yield more events
LAM_LIST = [5.0, 10.0, 30.0]


# Checkerboard intensity pattern as fractions of lam (must satisfy 0 < low < high < lam for PG)
LOW_FRAC = 0.35
HIGH_FRAC = 0.65

# Prior covariance kernel parameters
KERNEL_NAME = "matern_2_3"  # one of: "gaussian", "matern_1_2", "matern_2_3", "matern_3_5"
RHO = 16.0
V2 = 0.1

# Runtime / sampling budget
REPETITIONS = 5
N_SAMPLES_KEEP = 10 #50
BURN_IN = 0 #50
THIN = 2

# Output
OUTPUT_DIRNAME = "results"
OUTPUT_FILENAME = "benchmark_synthetic_checkerboards.csv"


# Reproducibility: base seed; each (config, repetition, estimator) derives from it.
BASE_SEED = 12345



# --- imports from project (sys.path setup) -------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# Make sure `import syntheticdata` works when called from repo root.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Modules in import matplotlib at import-time even when we don't plot.
# To keep this benchmark minimal and runnable in headless envs, stub matplotlib if missing.
try:
    import matplotlib.pyplot as _plt  # noqa: F401
except Exception:
    import types

    _mpl = types.ModuleType("matplotlib")
    _pyplot = types.ModuleType("matplotlib.pyplot")
    sys.modules.setdefault("matplotlib", _mpl)
    sys.modules.setdefault("matplotlib.pyplot", _pyplot)

import syntheticdata as sd
import covariance_kernels as ck
import polyagammadensity as pgd


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    cls: type


ESTIMATORS: list[EstimatorSpec] = [
    EstimatorSpec(name="PolyaGammaDensity2D", cls=pgd.PolyaGammaDensity2D),
    # TODO benchmark design for RampDensity2D:
    # Before adding RampDensity2D back in, decide how often the Gaussian mixture
    # approximation should be built and where it should be cached/read from.
    # Building the mixture once, loading it from disk, or rebuilding it for each
    # configuration measure different costs and will affect the runtime comparison.
    # EstimatorSpec(name="RampDensity2D", cls=pgd.RampDensity2D),
]


def _make_covariance(n: int, m: int, rho: float, v2: float, kernel_name: str) -> np.ndarray:
    if kernel_name == "gaussian":
        return ck.spatial_covariance_gaussian(n, m, rho, v2)
    if kernel_name == "matern_1_2":
        return ck.spatial_covariance_matern_1_2(n, m, rho, v2)
    if kernel_name == "matern_2_3":
        return ck.spatial_covariance_matern_2_3(n, m, rho, v2)
    if kernel_name == "matern_3_5":
        return ck.spatial_covariance_matern_3_5(n, m, rho, v2)
    raise ValueError(f"Unknown kernel_name={kernel_name!r}")


def _seed_for(base_seed: int, *parts: object) -> int:
    """
    Stable, deterministic seed from a base seed and a tuple of config parts.
    Avoid Python's built-in hash (salted per process).
    """
    s = str(base_seed) + "|" + "|".join(map(str, parts))
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _n_iter_total(n_keep: int, burn_in: int, thin: int) -> int:
    """
    PolyaGammaDensity.sample_posterior iterates `range(n_iter)` and yields after burn_in/thin.
    Be careful: The two samplers interpret (n_iter, burn_in, thin) differently:
    - PolyaGammaDensity.sample_posterior iterates `range(n_iter)` and yields after burn_in/thin.
    - SmoothRampMixin.sample_posterior runs total_iter = burn_in + n_iter * thin (where n_iter ~ kept samples).
    return burn_in + n_keep * thin
    """
    return burn_in + n_keep * thin



def _make_estimator(
    estimator_cls: type,
    n: int,
    m: int,
    lam: float,
) -> object:
    """
    Create estimator instance with kwargs expected by existing code.
    """
    kwargs = {"n": n, "m": m, "lam": float(lam)}
    return estimator_cls(**kwargs)


def run_one(
    estimator: EstimatorSpec,
    nn: int,
    ncheck: int,
    lam: float,
    repetition: int,
) -> dict:
    n = nn * ncheck
    m = n

    t_total0 = time.perf_counter()
    est = _make_estimator(estimator.cls, n=n, m=m, lam=lam)

    # Checkerboard intensity pattern (field-space), then map to latent parameters f.
    low = float(LOW_FRAC * lam)
    high = float(HIGH_FRAC * lam)
    # Ensure intensities are strictly inside (0, lam) so f_from_field is finite.
    eps = 1e-6
    low = float(np.clip(low, eps, lam - eps))
    high = float(np.clip(high, eps, lam - eps))

    intensity_img = sd.checkerboard(nn=nn, ncheck=ncheck, a=low, b=high)
    intensity = intensity_img.ravel()

    prior_mean = est.f_from_field(intensity)
    prior_cov = _make_covariance(n, m, rho=RHO, v2=V2, kernel_name=KERNEL_NAME)
    est.set_prior_Gaussian(prior_mean, prior_cov)

    # Synthetic data from induced field.
    rng_seed = _seed_for(BASE_SEED, estimator.name, nn, ncheck, lam, repetition, "data")
    np.random.seed(rng_seed)
    data = est.random_events_from_field(est.field_from_f(est.prior_mean))
    n_events = int(np.sum(data))
    est.set_data(data)

    # Time posterior sampling (consume generator).
    sample_seed = _seed_for(BASE_SEED, estimator.name, nn, ncheck, lam, repetition, "sample")
    n_iter = _n_iter_total(N_SAMPLES_KEEP, BURN_IN, THIN)

    t0 = time.perf_counter()
    for _f in est.sample_posterior(
        n_iter=n_iter,
        burn_in=BURN_IN,
        thin=THIN,
        initial_f=np.array(est.prior_mean, copy=True),
        random_seed=sample_seed,
    ):
        pass
    runtime_sampling_s = time.perf_counter() - t0
    runtime_total_s = time.perf_counter() - t_total0

    return {
        "estimator": estimator.name,
        "nn": int(nn),
        "ncheck": int(ncheck),
        "n": int(n),
        "lam": float(lam),
        "n_events": int(n_events),
        "rep": int(repetition),
        "kernel": str(KERNEL_NAME),
        "rho": float(RHO),
        "v2": float(V2),
        "n_keep": int(N_SAMPLES_KEEP),
        "burn_in": int(BURN_IN),
        "thin": int(THIN),
        "runtime_s": float(runtime_total_s),
        "runtime_sampling_s": float(runtime_sampling_s),
        "seed_data": int(rng_seed),
        "seed_sample": int(sample_seed),
    }


def main() -> Path:
    rows: list[dict] = []

    for nn in NN_LIST:
        for ncheck in NCHECK_LIST:
            for lam in LAM_LIST:
                for rep in range(REPETITIONS):
                    for estimator in ESTIMATORS:
                        print(
                            f"Running {estimator.name}: "
                            f"nn={nn}, ncheck={ncheck}, lam={lam}, rep={rep}"
                        )
                        row = run_one(
                            estimator=estimator,
                            nn=int(nn),
                            ncheck=int(ncheck),
                            lam=float(lam),
                            repetition=int(rep),
                        )
                        rows.append(row)
                        print(
                            f"  done: n_events={row['n_events']}, "
                            f"runtime_s={row['runtime_s']:.3f}, "
                            f"sampling_s={row['runtime_sampling_s']:.3f}"
                        )

    df = pd.DataFrame(rows)

    out_dir = REPO_ROOT / OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.groupby(['estimator', 'lam'])['n_events'].mean().reset_index())

    return out_path


if __name__ == "__main__":
    main()

