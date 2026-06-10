import numpy as np
from polyagammadensity import Mixin2D 
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


def spatial_covariance_gaussian(n, m, rho, v2):
    """
    Docstring for spatial_covariance_gaussian
    
    :param n: number of gridpoints along "x-axis"
    :param m: number of gridpoints along "y-axis"
    :param rho: the spatial covariance (in units of integer grid)
    :param v: autocorrelation at origin (i.e. the "amplitude")
    """

    d2 = _d2(n, m)

    return v2 * np.exp(-d2/(2 * rho**2))  + 0.000001 * v2 * np.identity(n*m)


def spatial_covariance_matern_1_2(n, m, rho, v2):
    d = _d2(n,m)**0.5

    return v2 * np.exp(-d/rho)


def spatial_covariance_matern_2_3(n, m, rho, v2):
    d = _d2(n,m)**0.5

    return v2 * (1+3**0.5 * d/rho) * np.exp(- 3**0.5 * d / rho)


def spatial_covariance_matern_3_5(n, m, rho, v2):
    d2 = _d2(n,m)
    d = d2**0.5

    return v2 * (1+5**0.5 * d / rho + 5 * d2 / (3 * rho**2)) * np.exp(- 5**0.5 * d / rho)


def precision_matern(n, m, rho, v2, boundary="zero"):
    """
    Sparse Matern-style precision for an n x m grid.

    ``boundary`` is passed to :func:`laplacian_2d`. The default ``"zero"``
    preserves the previous zero-extension behavior. Use ``"symmetric"`` for
    mirrored boundary values, e.g. A_ext[-1, j] = A[0, j].
    """
    laplacian = laplacian_2d(n, m, boundary=boundary)
    ## compute tau and alpha
    #tau, alpha = 1, 1

    alpha = 0.5 / (np.cosh(1/rho) -1)


    Q = (sps.eye(n * m, format="csr") + alpha * laplacian).tocsc()
    Q = Q.T @ Q
    e = np.ones(Q.shape[0])
    e[ (n//2)*m + m//2] = 0

    plt.figure()
    plt.imshow( (Q @ e).reshape( (n, m)))
    plt.show()

    kernel = sparse_linalg.spsolve(Q, e)

    plt.figure()
    plt.imshow( kernel.reshape((n, m)))
    plt.show()

    iv2 = np.sum(kernel * e)
    return (iv2 / v2) * Q

def precision_matern_9pt(ny, mx, tau=1.0, alpha=0.2):
    """
    Sparse precision for an n x n grid using a 9-point Laplacian stencil.

    If rho and v2 are provided, this mirrors precision_matern() with
    Q = I + alpha L and alpha = 0.5 / (cosh(1/rho) - 1), followed by the same
    Q.T @ Q and v2 normalization.

    If rho is omitted, this remains the generic builder
    Q = tau I + alpha L, with alpha defaulting to 0.2.

    Uses a 9-point Laplacian stencil with diagonal neighbors.

    Stencil references:
    https://en.wikipedia.org/wiki/Nine-point_stencil
    https://notebook.community/eramirem/numerical-methods-pdes/05_elliptic
    https://scicomp.stackexchange.com/questions/37656/tensor-product-representation-for-the-9-point-finite-difference-approximations-f
    """
    one_dim = sps.diags(
        [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    neighbor_1d = sps.diags(
        [np.ones(n - 1), np.ones(n - 1)],
        offsets=[-1, 1],
        format="csr",
    )
    identity = sps.eye(n, format="csr")

    cardinal_laplacian = (
        sps.kron(identity, one_dim, format="csr")
        + sps.kron(one_dim, identity, format="csr")
    )
    diagonal_neighbors = sps.kron(neighbor_1d, neighbor_1d, format="csr")

    laplacian = (
        4.0 * cardinal_laplacian
        + 4.0 * sps.eye(n * n, format="csr")
        - diagonal_neighbors
    ) / 6.0

    if rho is None:
        if alpha is None:
            alpha = 0.2
        return (tau * sps.eye(n * n, format="csr") + alpha * laplacian).tocsc()

    if v2 is None:
        raise ValueError("v2 must be provided when rho is provided")
    
    return (tau * sps.eye(n * n, format="csr") + alpha * laplacian).tocsc()

    # alpha = 0.5 / (np.cosh(1/rho) -1)

    # Q = (sps.eye(n * n, format="csr") + alpha * laplacian).tocsc()
    # Q = Q.T @ Q
    # e = np.zeros(Q.shape[0])
    # e[(n//2)*n + n//2] = 1

    # kernel = sparse_linalg.spsolve(Q, e)
    # iv2 = np.sum(kernel * e)
    # return (iv2 / v2) * Q

    
## --> TODO: move out of this module into research experiments module
