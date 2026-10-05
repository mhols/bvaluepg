import numpy as np
from polyagammapoisson.polyagammadensity import Mixin2D 
import scipy.sparse as sps
import scipy.sparse.linalg as sparse_linalg

import matplotlib.pyplot as plt


def laplacian_1d(n, boundary="zero"):
    """
    Positive 1D Laplacian for a line with configurable boundary handling.

    ``boundary="zero"`` keeps the previous zero-extension behavior.
    ``boundary="symmetric"`` mirrors the boundary value itself, e.g.
    u[-1] = u[0] and u[n] = u[n-1].
    """
    if n < 1:
        raise ValueError("n must be positive")

    main = 2.0 * np.ones(n)
    off = -np.ones(n - 1)

    if boundary == "zero":
        pass
    elif boundary == "symmetric":
        main[0] = 1.0
        main[-1] = 1.0
    else:
        raise ValueError(f"unknown boundary: {boundary}")

    return sps.diags(
        [off, main, off],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csr",
    )


def laplacian_2d(ny, nx, boundary="zero"):
    """
    Positive 2D Laplacian for row-major scan order on shape (ny, nx).

    For ``image.ravel(order="C")`` the x-direction is the fast index, hence
    L = L_y kron I_x + I_y kron L_x.
    """
    Ly = laplacian_1d(ny, boundary=boundary)
    Lx = laplacian_1d(nx, boundary=boundary)
    Iy = sps.eye(ny, format="csr")
    Ix = sps.eye(nx, format="csr")

    return (
        sps.kron(Ly, Ix, format="csr")
        + sps.kron(Iy, Lx, format="csr")
    )


def _d2(n, m):
    x, y = np.meshgrid( np.arange(m), np.arange(n))
    x = Mixin2D.image_to_scanorder(x)
    y = Mixin2D.image_to_scanorder(y)

    d2 = ( (y[:, None] - y[None, :])**2  + (x[:,None] - x[None, :])**2)

    return d2


def spatial_covariance_gaussian(n=None, m=None, rho=None, v2=None, **kwargs):
    """
    Docstring for spatial_covariance_gaussian
    
    :param n: number of gridpoints along "x-axis"
    :param m: number of gridpoints along "y-axis"
    :param rho: the spatial covariance (in units of integer grid)
    :param v: autocorrelation at origin (i.e. the "amplitude")
    """

    d2 = _d2(n, m)

    return v2 * np.exp(-d2/(2 * rho**2))  + 0.000001 * v2 * np.identity(n*m)


def spatial_covariance_matern_1_2(n=None, m=None, rho=None, v2=None, **kwargs):
    d = _d2(n,m)**0.5

    return v2 * np.exp(-d/rho)


def spatial_covariance_matern_2_3(n=None, m=None, rho=None, v2=None, **kwargs):
    d = _d2(n,m)**0.5

    return v2 * (1+3**0.5 * d/rho) * np.exp(- 3**0.5 * d / rho)


def spatial_covariance_matern_3_5(n=None, m=None, rho=None, v2=None, **kwargs):
    d2 = _d2(n,m)
    d = d2**0.5

    return v2 * (1+5**0.5 * d / rho + 5 * d2 / (3 * rho**2)) * np.exp(- 5**0.5 * d / rho)


def precision_matern(n=None, m=None, rho=None, v2=None, boundary="zero", **kwargs):
    """Build a variance-scaled, Matern-style 5-point precision.

    For the positive discrete Laplacian L, the function constructs

        alpha = 1 / (2 * (cosh(1 / rho) - 1)),
        A = I + alpha * L,
        P0 = A.T @ A.

    The returned precision is P = (s_ref / v2) * P0, where

        s_ref = e_ref.T @ solve(P0, e_ref)

    is the marginal variance of the unscaled prior at the central reference
    cell. Consequently, the central entry of P^{-1} equals v2.

    Parameters
    ----------
    n, m : int
        Number of rows and columns. Vectors use row-major scan order.
    rho : float
        Positive scale parameter of the basis operator A. It is not generally
        the actual 1/e correlation length of the final covariance P^{-1}.
    v2 : float
        Positive target variance at the central reference cell.
    boundary : {"zero", "symmetric"}
        Boundary handling passed to laplacian_2d().

    Returns
    -------
    scipy.sparse matrix
        The scaled prior precision P for an n-by-m latent Gaussian field.
    
    e_ref              markiert die mittlere Zelle
    covariance_column  Kovarianzen aller Zellen mit dieser Zelle
    s_ref              Varianz dieser Referenzzelle   
    """
    laplacian = laplacian_2d(n, m, boundary=boundary)
    alpha = 0.5 / (np.cosh(1/rho) -1)

    A = (sps.eye(n * m, format="csr") + alpha * laplacian).tocsc()
    P0 = A.T @ A
    e_ref = np.zeros(P0.shape[0])
    e_ref[(n//2)*m + m//2] = 1

    covariance_column = sparse_linalg.spsolve(P0, e_ref)   # kernel column of P0^{-1} corresponding to the reference cell

    s_ref = np.sum(covariance_column * e_ref)
    return (s_ref / v2) * P0

def laplacian_9pt_2d(ny, nx, boundary="zero"):
    """Positive 9-point Laplacian for row-major order and unit grid spacing.

    The interior stencil is (20 * center - 4 * cardinal - diagonal) / 6.
    Boundary handling matches laplacian_2d(): zero extension or repetition
    of the boundary cell, including diagonal neighbours at edges and corners.
    """
    Ly = laplacian_1d(ny, boundary=boundary)
    Lx = laplacian_1d(nx, boundary=boundary)
    Iy = sps.eye(ny, format="csr")
    Ix = sps.eye(nx, format="csr")

    By = 2.0 * Iy - Ly
    Bx = 2.0 * Ix - Lx
    # A singleton axis has two ghost neighbours mapped to its only cell.
    if boundary == "symmetric":
        if ny == 1:
            By = 2.0 * Iy
        if nx == 1:
            Bx = 2.0 * Ix

    cardinal_neighbors = (
        sps.kron(By, Ix, format="csr")
        + sps.kron(Iy, Bx, format="csr")
    )
    diagonal_neighbors = sps.kron(By, Bx, format="csr")

    return (
        20.0 * sps.eye(ny * nx, format="csr")
        - 4.0 * cardinal_neighbors
        - diagonal_neighbors
    ) / 6.0


def precision_matern_9pt(n=None, m=None, rho=None, v2=None, boundary="zero", **kwargs):
    """Build a variance-scaled 9-point precision analogously to precision_matern.

    n and m are rows and columns in row-major order. The basis operator is
    A = I + alpha * L9 with alpha = 0.5 / (cosh(1 / rho) - 1).
    The precision P0 = A.T @ A is scaled so that the central reference cell
    has marginal variance v2. Its normalization is computed for L9 itself.

    rho is an operator-scale parameter in grid-cell units, not generally
    the actual 1/e correlation length of the resulting covariance.
    boundary supports "zero" and "symmetric", as in the 5-point builder.
    """
    laplacian = laplacian_9pt_2d(n, m, boundary=boundary)
    alpha = 0.5 / (np.cosh(1/rho) -1)

    A = (sps.eye(n * m, format="csr") + alpha * laplacian).tocsc()
    P0 = A.T @ A
    e_ref = np.zeros(P0.shape[0])
    e_ref[(n//2)*m + m//2] = 1

    covariance_column = sparse_linalg.spsolve(P0, e_ref)

    s_ref = np.sum(covariance_column * e_ref)
    return (s_ref / v2) * P0

    
## --> TODO: move out of this module into research experiments module
