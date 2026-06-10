from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
sys.path.insert(0, str(SOURCE_DIR))

import covariance_kernels as ck


def _apply_laplacian(image, boundary):
    ny, nx = image.shape
    L = ck.laplacian_2d(ny, nx, boundary=boundary)
    return (L @ image.ravel(order="C")).reshape((ny, nx), order="C")


def test_symmetric_boundary_keeps_constant_field_in_laplacian_nullspace():
    L = ck.laplacian_2d(3, 4, boundary="symmetric")
    ones = np.ones(12)

    assert np.allclose(L @ ones, 0.0)


def test_zero_boundary_does_not_keep_constant_field_in_laplacian_nullspace():
    L = ck.laplacian_2d(3, 4, boundary="zero")
    ones = np.ones(12)

    assert not np.allclose(L @ ones, 0.0)


def test_symmetric_boundary_matches_expected_upper_left_corner():
    image = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    result = _apply_laplacian(image, boundary="symmetric")

    assert result[0, 0] == pytest.approx(2 * 1.0 - 2.0 - 4.0)


def test_symmetric_boundary_keeps_interior_five_point_stencil():
    image = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    result = _apply_laplacian(image, boundary="symmetric")

    assert result[1, 1] == pytest.approx(4 * 5.0 - 2.0 - 4.0 - 6.0 - 8.0)


def test_laplacian_2d_uses_row_major_scan_order_on_rectangular_grid():
    image = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    result = _apply_laplacian(image, boundary="symmetric")

    expected = np.array(
        [
            [
                2 * 1.0 - 2.0 - 4.0,
                2.0 - 1.0 + 2.0 - 3.0 + 2.0 - 5.0,
                2 * 3.0 - 2.0 - 6.0,
            ],
            [
                2 * 4.0 - 1.0 - 5.0,
                5.0 - 4.0 + 5.0 - 6.0 + 5.0 - 2.0,
                2 * 6.0 - 3.0 - 5.0,
            ],
        ]
    )

    assert np.allclose(result, expected)


def test_laplacian_rejects_unknown_boundary():
    with pytest.raises(ValueError, match="unknown boundary"):
        ck.laplacian_2d(2, 3, boundary="reflect")


def _save_boundary_plot(image, name, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zero = _apply_laplacian(image, boundary="zero")
    symmetric = _apply_laplacian(image, boundary="symmetric")
    diff = symmetric - zero

    vmax = max(abs(zero).max(), abs(symmetric).max(), abs(diff).max())
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)

    panels = [
        ("input", image, "viridis"),
        ("L image, zero", zero, "coolwarm"),
        ("L image, symmetric", symmetric, "coolwarm"),
        ("symmetric - zero", diff, "coolwarm"),
    ]
    for ax, (title, data, cmap) in zip(axes, panels):
        if title == "input":
            im = ax.imshow(data, cmap=cmap)
        else:
            im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_boundary_check_plots(output_dir=None):
    """Save visual diagnostics for zero vs symmetric boundary handling."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "plots" / "laplacian_boundary_checks"
    else:
        output_dir = Path(output_dir)

    constant = np.ones((8, 8))
    ramp = np.add.outer(np.arange(8.0), np.arange(8.0))
    block = np.zeros((16, 16))
    block[4:12, 4:12] = 1.0

    paths = [
        _save_boundary_plot(constant, "constant_field", output_dir),
        _save_boundary_plot(ramp, "linear_ramp", output_dir),
        _save_boundary_plot(block, "center_block", output_dir),
    ]
    return paths


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    for path in save_boundary_check_plots(target_dir):
        print(path)
