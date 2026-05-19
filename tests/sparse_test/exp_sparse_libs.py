#!/usr/bin/env python3
"""
Benchmark dense and sparse Cholesky-like factorizations for PG-Gibbs matrices.

The benchmark is intentionally self-contained and does not import project code.
It builds two matrix families on an n x n grid:

1. A dense PG precision matrix built from a Matern-3/2 covariance matrix
   like source/syntheticdata.py uses:

       Sigma0 = spatial_covariance_matern_2_3(n, n, rho, v2)
       A = Sigma0^{-1} + diag(w)

2. A sparse precision-like SPD matrix based on a 2D grid Laplacian.

Backends can be selected with --backends. PyTorch and TensorFlow are not part
of the default backend set because they can conflict with SciPy/CHOLMOD OpenMP
runtimes in the same Python process on macOS.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import scipy.linalg as sla
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception as exc:
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


DEFAULT_GRID_SIZES = (32, 64)
DEFAULT_REPETITIONS = 3
DEFAULT_WARMUPS = 1
DEFAULT_BACKENDS = ("scipy-sparse", "cholmod", "numpy", "scipy-dense")
VALID_BACKENDS = ("scipy-sparse", "cholmod", "numpy", "scipy-dense", "torch", "tensorflow")
BYTES_PER_MB = 1024.0 * 1024.0


@dataclass(frozen=True)
class MatrixStats:
    nnz: int
    density: float
    memory_mb: float


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    matrix_type: str
    grid_n: int
    dim: int
    nnz: int
    density: float
    memory_mb: float
    factor_nnz: int | None
    factor_memory_mb: float | None
    mean_time_s: float | None
    std_time_s: float | None
    status: str


def matern32_covariance(
    grid_n: int,
    rho: float = 16.0,
    variance: float = 0.1,
    jitter: float = 1e-6,
) -> np.ndarray:
    """
    Dense Matern-3/2 covariance-like SPD matrix on a square grid.
    """
    coords = np.indices((grid_n, grid_n), dtype=np.float64).reshape(2, -1).T
    dx = coords[:, 0, None] - coords[:, 0]
    dy = coords[:, 1, None] - coords[:, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    scaled = math.sqrt(3.0) * dist / rho
    cov = variance * (1.0 + scaled) * np.exp(-scaled)
    cov.flat[:: cov.shape[0] + 1] += jitter
    return np.ascontiguousarray(cov)


def dense_precision_from_matern32_covariance(
    grid_n: int,
    rho: float = 16.0,
    variance: float = 0.1,
    jitter: float = 1e-6,
) -> np.ndarray:
    """
    Build Sigma0^{-1} from the dense Matern-3/2 covariance used by syntheticdata.

    This mirrors the expensive setup in polyagammadensity.py:

        L = cholesky(Sigma0)
        X = solve_triangular(L, I)
        Sigma0_inv = solve_triangular(L.T, X)
    """
    cov = matern32_covariance(grid_n, rho=rho, variance=variance, jitter=jitter)
    identity = np.eye(cov.shape[0], dtype=cov.dtype)
    chol = sla.cholesky(cov, lower=True, check_finite=False)
    x = sla.solve_triangular(chol, identity, lower=True, check_finite=False)
    precision = sla.solve_triangular(chol.T, x, lower=False, check_finite=False)
    precision = 0.5 * (precision + precision.T)
    return np.ascontiguousarray(precision)


def sparse_precision_matrix(
    grid_n: int,
    tau: float = 1.0,
    alpha: float = 0.2,
    fmt: str = "csc",
) -> sp.spmatrix:
    """
    Sparse precision-like SPD matrix Q = tau I + alpha L for a 2D grid.
    """
    one_dim = sp.diags(
        diagonals=[-np.ones(grid_n - 1), 2.0 * np.ones(grid_n), -np.ones(grid_n - 1)],
        offsets=[-1, 0, 1],
        shape=(grid_n, grid_n),
        format="csr",
    )
    identity = sp.eye(grid_n, format="csr")
    laplacian = sp.kron(identity, one_dim, format="csr") + sp.kron(one_dim, identity, format="csr")
    q = tau * sp.eye(grid_n * grid_n, format="csr") + alpha * laplacian
    return q.asformat(fmt)


def positive_weights(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    PG-like positive diagonal contribution.
    """
    return rng.gamma(shape=2.0, scale=0.25, size=dim) + 1e-3


def dense_stats(matrix: np.ndarray) -> MatrixStats:
    nnz = int(np.count_nonzero(matrix))
    density = nnz / float(matrix.size)
    return MatrixStats(nnz=nnz, density=density, memory_mb=matrix.nbytes / BYTES_PER_MB)


def sparse_stats(matrix: sp.spmatrix) -> MatrixStats:
    nnz = int(matrix.nnz)
    density = nnz / float(matrix.shape[0] * matrix.shape[1])
    if hasattr(matrix, "data") and hasattr(matrix, "indices") and hasattr(matrix, "indptr"):
        memory_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    else:
        matrix = matrix.tocsr()
        memory_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return MatrixStats(nnz=nnz, density=density, memory_mb=memory_bytes / BYTES_PER_MB)


def sparse_factor_memory_mb(matrix: sp.spmatrix) -> float:
    if hasattr(matrix, "data") and hasattr(matrix, "indices") and hasattr(matrix, "indptr"):
        memory_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    else:
        matrix = matrix.tocsc()
        memory_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return float(memory_bytes) / BYTES_PER_MB


def cholmod_factor_stats(factor: object) -> tuple[int | None, float | None]:
    """
    Return storage stats for scikit-sparse/CHOLMOD factor variants.

    Some installations return a Factor object with L(), others return
    (L, permutation) directly.
    """
    if isinstance(factor, tuple):
        lower = factor[0]
    else:
        lower = factor.L()
    return int(lower.nnz), sparse_factor_memory_mb(lower)


def torch_dense_memory_mb(tensor: object) -> float:
    return float(tensor.element_size() * tensor.nelement()) / BYTES_PER_MB


def torch_sparse_memory_mb(tensor: object) -> float:
    tensor = tensor.coalesce()
    values = tensor.values()
    indices = tensor.indices()
    memory_bytes = values.element_size() * values.nelement()
    memory_bytes += indices.element_size() * indices.nelement()
    return float(memory_bytes) / BYTES_PER_MB


def tensorflow_dense_memory_mb(tensor: object) -> float:
    return float(tensor.numpy().nbytes) / BYTES_PER_MB


def tensorflow_sparse_memory_mb(tensor: object) -> float:
    memory_bytes = tensor.values.numpy().nbytes + tensor.indices.numpy().nbytes
    return float(memory_bytes) / BYTES_PER_MB


def time_repeated(
    label: str,
    make_matrix: Callable[[np.random.Generator], object],
    factorize: Callable[[object], object],
    repetitions: int,
    warmups: int,
    seed: int,
) -> tuple[float | None, float | None, str]:
    """
    Measure factorization time only. Matrix construction is outside the timer.
    """
    rng = np.random.default_rng(seed)

    try:
        for _ in range(warmups):
            matrix = make_matrix(rng)
            factorize(matrix)
            del matrix
        gc.collect()

        times: list[float] = []
        for _ in range(repetitions):
            matrix = make_matrix(rng)
            t0 = time.perf_counter()
            factor = factorize(matrix)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            del factor, matrix
        gc.collect()
    except Exception as exc:
        return None, None, f"ERROR: {type(exc).__name__}: {exc}"

    if not times:
        return None, None, f"SKIP: no repetitions for {label}"

    mean_time = statistics.fmean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_time, std_time, "OK"


def inspect_factor(
    make_matrix: Callable[[np.random.Generator], object],
    factorize: Callable[[object], object],
    factor_stats: Callable[[object], tuple[int | None, float | None]],
    seed: int,
) -> tuple[int | None, float | None, str | None]:
    """
    Run one extra factorization to collect fill-in/storage statistics.

    This keeps timing focused on the repeated benchmark while still reporting
    roughly how large the factor object is.
    """
    rng = np.random.default_rng(seed)
    try:
        matrix = make_matrix(rng)
        factor = factorize(matrix)
        factor_nnz, factor_memory_mb = factor_stats(factor)
        del factor, matrix
        gc.collect()
        return factor_nnz, factor_memory_mb, None
    except Exception as exc:
        return None, None, f"factor stats unavailable: {type(exc).__name__}: {exc}"


def add_diagonal_dense(base: np.ndarray, w: np.ndarray) -> np.ndarray:
    matrix = base.copy(order="C")
    matrix.flat[:: matrix.shape[0] + 1] += w
    return matrix


def add_diagonal_sparse(base: sp.spmatrix, w: np.ndarray) -> sp.csc_matrix:
    return (base + sp.diags(w, format="csc")).tocsc()


def benchmark_numpy_scipy_dense(
    grid_n: int,
    dense_precision_base: np.ndarray,
    sparse_base: sp.spmatrix,
    repetitions: int,
    warmups: int,
    seed: int,
    run_numpy: bool,
    run_scipy_dense: bool,
) -> list[BenchmarkResult]:
    dim = grid_n * grid_n
    results: list[BenchmarkResult] = []

    for matrix_type, base, make_matrix in (
        ("dense_matern_precision", dense_precision_base, lambda rng: add_diagonal_dense(dense_precision_base, positive_weights(dim, rng))),
        ("sparse_precision_as_dense", sparse_base, lambda rng: add_diagonal_sparse(sparse_base, positive_weights(dim, rng)).toarray()),
    ):
        stats = dense_stats(np.asarray(base if isinstance(base, np.ndarray) else base.toarray()))
        factor_nnz = dim * (dim + 1) // 2
        factor_memory_mb = stats.memory_mb

        if run_numpy:
            mean_time, std_time, status = time_repeated(
                "numpy.linalg.cholesky",
                make_matrix,
                np.linalg.cholesky,
                repetitions,
                warmups,
                seed + 11,
            )
            results.append(
                BenchmarkResult(
                    backend="NumPy dense Cholesky",
                    matrix_type=matrix_type,
                    grid_n=grid_n,
                    dim=dim,
                    nnz=stats.nnz,
                    density=stats.density,
                    memory_mb=stats.memory_mb,
                    factor_nnz=factor_nnz,
                    factor_memory_mb=factor_memory_mb,
                    mean_time_s=mean_time,
                    std_time_s=std_time,
                    status=status,
                )
            )

        if run_scipy_dense:
            mean_time, std_time, status = time_repeated(
                "scipy.linalg.cholesky",
                make_matrix,
                lambda matrix: sla.cholesky(matrix, lower=True, check_finite=False, overwrite_a=False),
                repetitions,
                warmups,
                seed + 12,
            )
            results.append(
                BenchmarkResult(
                    backend="SciPy dense Cholesky",
                    matrix_type=matrix_type,
                    grid_n=grid_n,
                    dim=dim,
                    nnz=stats.nnz,
                    density=stats.density,
                    memory_mb=stats.memory_mb,
                    factor_nnz=factor_nnz,
                    factor_memory_mb=factor_memory_mb,
                    mean_time_s=mean_time,
                    std_time_s=std_time,
                    status=status,
                )
            )

    return results


def benchmark_scipy_sparse_storage_and_lu(
    grid_n: int,
    sparse_base: sp.spmatrix,
    repetitions: int,
    warmups: int,
    seed: int,
) -> list[BenchmarkResult]:
    dim = grid_n * grid_n
    w0 = positive_weights(dim, np.random.default_rng(seed + 20))
    csr = (sparse_base.tocsr() + sp.diags(w0, format="csr")).tocsr()
    csc = csr.tocsc()
    results: list[BenchmarkResult] = []

    for fmt_name, matrix in (("scipy_sparse_csr_storage", csr), ("scipy_sparse_csc_storage", csc)):
        stats = sparse_stats(matrix)
        results.append(
            BenchmarkResult(
                backend="SciPy sparse storage",
                matrix_type=fmt_name,
                grid_n=grid_n,
                dim=dim,
                nnz=stats.nnz,
                density=stats.density,
                memory_mb=stats.memory_mb,
                factor_nnz=None,
                factor_memory_mb=None,
                mean_time_s=None,
                std_time_s=None,
                status="OK: storage only",
            )
        )

    stats = sparse_stats(csc)
    factor_nnz, factor_memory_mb, factor_status = inspect_factor(
        lambda rng: add_diagonal_sparse(sparse_base, positive_weights(dim, rng)),
        lambda matrix: spla.splu(matrix),
        lambda factor: (
            int(factor.L.nnz + factor.U.nnz),
            sparse_factor_memory_mb(factor.L) + sparse_factor_memory_mb(factor.U),
        ),
        seed + 22,
    )
    mean_time, std_time, status = time_repeated(
        "scipy.sparse.linalg.splu",
        lambda rng: add_diagonal_sparse(sparse_base, positive_weights(dim, rng)),
        lambda matrix: spla.splu(matrix),
        repetitions,
        warmups,
        seed + 21,
    )
    if factor_status is not None and status == "OK":
        status = f"OK; {factor_status}"
    results.append(
        BenchmarkResult(
            backend="SciPy sparse LU (splu)",
            matrix_type="sparse_precision",
            grid_n=grid_n,
            dim=dim,
            nnz=stats.nnz,
            density=stats.density,
            memory_mb=stats.memory_mb,
            factor_nnz=factor_nnz,
            factor_memory_mb=factor_memory_mb,
            mean_time_s=mean_time,
            std_time_s=std_time,
            status=status,
        )
    )

    return results


def benchmark_cholmod(
    grid_n: int,
    sparse_base: sp.spmatrix,
    repetitions: int,
    warmups: int,
    seed: int,
) -> BenchmarkResult:
    dim = grid_n * grid_n
    stats = sparse_stats(sparse_base)

    try:
        from sksparse.cholmod import cholesky
    except Exception as exc:
        return BenchmarkResult(
            backend="scikit-sparse CHOLMOD",
            matrix_type="sparse_precision",
            grid_n=grid_n,
            dim=dim,
            nnz=stats.nnz,
            density=stats.density,
            memory_mb=stats.memory_mb,
            factor_nnz=None,
            factor_memory_mb=None,
            mean_time_s=None,
            std_time_s=None,
            status=f"SKIP: {type(exc).__name__}: {exc}",
        )

    factor_nnz, factor_memory_mb, factor_status = inspect_factor(
        lambda rng: add_diagonal_sparse(sparse_base, positive_weights(dim, rng)),
        lambda matrix: cholesky(matrix),
        cholmod_factor_stats,
        seed + 32,
    )
    mean_time, std_time, status = time_repeated(
        "sksparse.cholmod.cholesky",
        lambda rng: add_diagonal_sparse(sparse_base, positive_weights(dim, rng)),
        lambda matrix: cholesky(matrix),
        repetitions,
        warmups,
        seed + 31,
    )
    if factor_status is not None and status == "OK":
        status = f"OK; {factor_status}"
    return BenchmarkResult(
        backend="scikit-sparse CHOLMOD",
        matrix_type="sparse_precision",
        grid_n=grid_n,
        dim=dim,
        nnz=stats.nnz,
        density=stats.density,
        memory_mb=stats.memory_mb,
        factor_nnz=factor_nnz,
        factor_memory_mb=factor_memory_mb,
        mean_time_s=mean_time,
        std_time_s=std_time,
        status=status,
    )


def benchmark_torch(
    grid_n: int,
    dense_precision_base: np.ndarray,
    sparse_base: sp.spmatrix,
    repetitions: int,
    warmups: int,
    seed: int,
) -> list[BenchmarkResult]:
    dim = grid_n * grid_n
    stats = dense_stats(dense_precision_base)
    sparse_stats_value = sparse_stats(sparse_base)

    try:
        import torch
    except Exception as exc:
        skip = f"SKIP: {type(exc).__name__}: {exc}"
        return [
            BenchmarkResult("PyTorch dense Cholesky", "dense_matern_precision", grid_n, dim, stats.nnz, stats.density, stats.memory_mb, None, None, None, None, skip),
            BenchmarkResult("PyTorch sparse storage", "sparse_precision_coo", grid_n, dim, sparse_stats_value.nnz, sparse_stats_value.density, sparse_stats_value.memory_mb, None, None, None, None, skip),
        ]

    def make_dense(rng: np.random.Generator) -> object:
        return torch.from_numpy(add_diagonal_dense(dense_precision_base, positive_weights(dim, rng)))

    def factorize(matrix: object) -> object:
        factor = torch.linalg.cholesky(matrix)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return factor

    sample_tensor = torch.from_numpy(dense_precision_base)
    memory_mb = torch_dense_memory_mb(sample_tensor)
    mean_time, std_time, status = time_repeated(
        "torch.linalg.cholesky",
        make_dense,
        factorize,
        repetitions,
        warmups,
        seed + 41,
    )

    coo = sparse_base.tocoo()
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.int64)
    values = torch.tensor(coo.data)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()
    sparse_memory_mb = torch_sparse_memory_mb(sparse_tensor)

    return [
        BenchmarkResult(
            backend="PyTorch dense Cholesky",
            matrix_type="dense_matern_precision",
            grid_n=grid_n,
            dim=dim,
            nnz=stats.nnz,
            density=stats.density,
            memory_mb=memory_mb,
            factor_nnz=dim * (dim + 1) // 2,
            factor_memory_mb=memory_mb,
            mean_time_s=mean_time,
            std_time_s=std_time,
            status=status,
        ),
        BenchmarkResult(
            backend="PyTorch sparse storage",
            matrix_type="sparse_precision_coo",
            grid_n=grid_n,
            dim=dim,
            nnz=sparse_stats_value.nnz,
            density=sparse_stats_value.density,
            memory_mb=sparse_memory_mb,
            factor_nnz=None,
            factor_memory_mb=None,
            mean_time_s=None,
            std_time_s=None,
            status="OK: storage only",
        ),
    ]


def benchmark_tensorflow(
    grid_n: int,
    dense_precision_base: np.ndarray,
    sparse_base: sp.spmatrix,
    repetitions: int,
    warmups: int,
    seed: int,
) -> list[BenchmarkResult]:
    dim = grid_n * grid_n
    stats = dense_stats(dense_precision_base)
    sparse_stats_value = sparse_stats(sparse_base)

    try:
        import tensorflow as tf
    except Exception as exc:
        skip = f"SKIP: {type(exc).__name__}: {exc}"
        return [
            BenchmarkResult("TensorFlow dense Cholesky", "dense_matern_precision", grid_n, dim, stats.nnz, stats.density, stats.memory_mb, None, None, None, None, skip),
            BenchmarkResult("TensorFlow sparse storage", "sparse_precision_coo", grid_n, dim, sparse_stats_value.nnz, sparse_stats_value.density, sparse_stats_value.memory_mb, None, None, None, None, skip),
        ]

    def make_dense(rng: np.random.Generator) -> object:
        return tf.convert_to_tensor(add_diagonal_dense(dense_precision_base, positive_weights(dim, rng)))

    def factorize(matrix: object) -> object:
        return tf.linalg.cholesky(matrix).numpy()

    sample_tensor = tf.convert_to_tensor(dense_precision_base)
    memory_mb = tensorflow_dense_memory_mb(sample_tensor)
    mean_time, std_time, status = time_repeated(
        "tf.linalg.cholesky",
        make_dense,
        factorize,
        repetitions,
        warmups,
        seed + 51,
    )

    coo = sparse_base.tocoo()
    sparse_tensor = tf.sparse.SparseTensor(
        indices=np.vstack((coo.row, coo.col)).T.astype(np.int64),
        values=coo.data,
        dense_shape=coo.shape,
    )
    sparse_memory_mb = tensorflow_sparse_memory_mb(sparse_tensor)

    return [
        BenchmarkResult(
            backend="TensorFlow dense Cholesky",
            matrix_type="dense_matern_precision",
            grid_n=grid_n,
            dim=dim,
            nnz=stats.nnz,
            density=stats.density,
            memory_mb=memory_mb,
            factor_nnz=dim * (dim + 1) // 2,
            factor_memory_mb=memory_mb,
            mean_time_s=mean_time,
            std_time_s=std_time,
            status=status,
        ),
        BenchmarkResult(
            backend="TensorFlow sparse storage",
            matrix_type="sparse_precision_coo",
            grid_n=grid_n,
            dim=dim,
            nnz=sparse_stats_value.nnz,
            density=sparse_stats_value.density,
            memory_mb=sparse_memory_mb,
            factor_nnz=None,
            factor_memory_mb=None,
            mean_time_s=None,
            std_time_s=None,
            status="OK: storage only",
        ),
    ]


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}g}"


def print_results_table(results: list[BenchmarkResult]) -> None:
    headers = [
        "backend",
        "matrix type",
        "grid n",
        "N",
        "nnz",
        "density",
        "memory MB",
        "factor nnz",
        "factor MB",
        "mean chol/fact s",
        "std s",
        "status",
    ]
    rows = [
        [
            row.backend,
            row.matrix_type,
            str(row.grid_n),
            str(row.dim),
            str(row.nnz),
            format_float(row.density, 4),
            format_float(row.memory_mb, 4),
            str(row.factor_nnz) if row.factor_nnz is not None else "-",
            format_float(row.factor_memory_mb, 4),
            format_float(row.mean_time_s, 5),
            format_float(row.std_time_s, 4),
            row.status,
        ]
        for row in results
    ]

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]

    def line(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    print(line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(line(row))


def parse_grid_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise argparse.ArgumentTypeError("At least one grid size is required.")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("Grid sizes must be positive integers.")
    return sizes


def parse_backends(value: str) -> tuple[str, ...]:
    requested = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not requested:
        raise argparse.ArgumentTypeError("At least one backend is required.")

    expanded: list[str] = []
    for backend in requested:
        if backend == "all":
            expanded.extend(VALID_BACKENDS)
        elif backend == "scipy":
            expanded.extend(("scipy-sparse", "scipy-dense"))
        elif backend == "dense":
            expanded.extend(("numpy", "scipy-dense", "torch", "tensorflow"))
        elif backend == "sparse":
            expanded.extend(("scipy-sparse", "cholmod"))
        elif backend in VALID_BACKENDS:
            expanded.append(backend)
        else:
            valid = ", ".join((*VALID_BACKENDS, "all", "scipy", "dense", "sparse"))
            raise argparse.ArgumentTypeError(f"Unknown backend {backend!r}. Valid values: {valid}.")

    deduplicated = tuple(dict.fromkeys(expanded))
    if not deduplicated:
        raise argparse.ArgumentTypeError("At least one backend is required.")
    return deduplicated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=parse_grid_sizes,
        default=DEFAULT_GRID_SIZES,
        help="Comma-separated grid side lengths. Matrix dimension is N=n*n. Default: 32,64.",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--rho", type=float, default=16.0)
    parser.add_argument("--variance", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument(
        "--backends",
        type=parse_backends,
        default=DEFAULT_BACKENDS,
        help=(
            "Comma-separated backend list. Valid values: "
            "scipy-sparse, cholmod, numpy, scipy-dense, torch, tensorflow. "
            "Aliases: sparse, dense, scipy, all. "
            "Default: scipy-sparse,cholmod,numpy,scipy-dense."
        ),
    )
    parser.add_argument(
        "--max-dense-dim",
        type=int,
        default=4096,
        help="Skip dense backends above this matrix dimension. Default keeps n=64 enabled.",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip PyTorch benchmarks even when torch is installed.",
    )
    parser.add_argument(
        "--skip-tensorflow",
        action="store_true",
        help="Skip TensorFlow benchmarks even when tensorflow is installed.",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.max_dense_dim < 1:
        parser.error("--max-dense-dim must be at least 1")
    backends = set(args.backends)
    if args.skip_torch:
        backends.discard("torch")
    if args.skip_tensorflow:
        backends.discard("tensorflow")
    if not backends:
        parser.error("No backends left after applying --skip-torch/--skip-tensorflow")
    args.backends = tuple(backend for backend in args.backends if backend in backends)
    return args


def main() -> None:
    if SCIPY_IMPORT_ERROR is not None:
        print(
            "This benchmark needs SciPy for dense and sparse matrix operations. "
            f"Import failed with {type(SCIPY_IMPORT_ERROR).__name__}: {SCIPY_IMPORT_ERROR}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    args = parse_args()
    all_results: list[BenchmarkResult] = []
    selected_backends = set(args.backends)

    print("Dense vs. sparse factorization benchmark")
    print(f"grid sizes: {args.sizes}; repetitions: {args.repetitions}; warmups: {args.warmups}")
    print(f"backends: {', '.join(args.backends)}")
    print()

    for grid_n in args.sizes:
        dim = grid_n * grid_n
        print(f"Building matrices for grid n={grid_n} (N={dim}) ...")

        sparse_base = sparse_precision_matrix(grid_n, tau=args.tau, alpha=args.alpha, fmt="csc")
        if "scipy-sparse" in selected_backends:
            all_results.extend(
                benchmark_scipy_sparse_storage_and_lu(
                    grid_n=grid_n,
                    sparse_base=sparse_base,
                    repetitions=args.repetitions,
                    warmups=args.warmups,
                    seed=args.seed + grid_n * 1000,
                )
            )
        if "cholmod" in selected_backends:
            all_results.append(
                benchmark_cholmod(
                    grid_n=grid_n,
                    sparse_base=sparse_base,
                    repetitions=args.repetitions,
                    warmups=args.warmups,
                    seed=args.seed + grid_n * 1000,
                )
            )

        needs_dense_precision = bool({"numpy", "scipy-dense", "torch", "tensorflow"} & selected_backends)

        if needs_dense_precision and dim > args.max_dense_dim:
            stats = sparse_stats(sparse_base)
            all_results.append(
                BenchmarkResult(
                    backend="Dense backends",
                    matrix_type="dense_matern_precision",
                    grid_n=grid_n,
                    dim=dim,
                    nnz=dim * dim,
                    density=1.0,
                    memory_mb=(dim * dim * np.dtype(np.float64).itemsize) / BYTES_PER_MB,
                    factor_nnz=dim * (dim + 1) // 2,
                    factor_memory_mb=(dim * dim * np.dtype(np.float64).itemsize) / BYTES_PER_MB,
                    mean_time_s=None,
                    std_time_s=None,
                    status=f"SKIP: N={dim} exceeds --max-dense-dim={args.max_dense_dim}",
                )
            )
            all_results.append(
                BenchmarkResult(
                    backend="Sparse baseline",
                    matrix_type="sparse_precision",
                    grid_n=grid_n,
                    dim=dim,
                    nnz=stats.nnz,
                    density=stats.density,
                    memory_mb=stats.memory_mb,
                    factor_nnz=None,
                    factor_memory_mb=None,
                    mean_time_s=None,
                    std_time_s=None,
                    status="OK: dense comparison skipped",
                )
            )
            del sparse_base
            gc.collect()
            continue

        dense_precision_base = None
        if needs_dense_precision:
            print("  building dense Matern covariance precision Sigma0_inv ...")
            setup_t0 = time.perf_counter()
            dense_precision_base = dense_precision_from_matern32_covariance(
                grid_n,
                rho=args.rho,
                variance=args.variance,
            )
            print(f"  dense precision setup: {time.perf_counter() - setup_t0:.3f} s")

        if {"numpy", "scipy-dense"} & selected_backends:
            if dense_precision_base is None:
                raise RuntimeError("dense_precision_base was not built")
            all_results.extend(
                benchmark_numpy_scipy_dense(
                    grid_n=grid_n,
                    dense_precision_base=dense_precision_base,
                    sparse_base=sparse_base,
                    repetitions=args.repetitions,
                    warmups=args.warmups,
                    seed=args.seed + grid_n * 1000,
                    run_numpy="numpy" in selected_backends,
                    run_scipy_dense="scipy-dense" in selected_backends,
                )
            )
        if "torch" in selected_backends:
            if dense_precision_base is None:
                raise RuntimeError("dense_precision_base was not built")
            all_results.extend(
                benchmark_torch(
                    grid_n=grid_n,
                    dense_precision_base=dense_precision_base,
                    sparse_base=sparse_base,
                    repetitions=args.repetitions,
                    warmups=args.warmups,
                    seed=args.seed + grid_n * 1000,
                )
            )
        if "tensorflow" in selected_backends:
            if dense_precision_base is None:
                raise RuntimeError("dense_precision_base was not built")
            all_results.extend(
                benchmark_tensorflow(
                    grid_n=grid_n,
                    dense_precision_base=dense_precision_base,
                    sparse_base=sparse_base,
                    repetitions=args.repetitions,
                    warmups=args.warmups,
                    seed=args.seed + grid_n * 1000,
                )
            )

        del dense_precision_base, sparse_base
        gc.collect()

    print()
    print_results_table(all_results)


if __name__ == "__main__":
    main()
