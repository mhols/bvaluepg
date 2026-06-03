import numpy as np
from polyagammadensity import Mixin2D 
import scipy.sparse as sps
import scipy.sparse.linalg as sparse_linalg

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


def precision_matern(n, m, rho, v2):
    """
    Sparse precision Q = I + alpha L for an n x m grid.
    """
    one_dim_x = sps.diags(
        [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    one_dim_y = sps.diags(
        [-np.ones(m - 1), 2.0 * np.ones(m), -np.ones(m - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    identity_x = sps.eye(n, format="csr")
    identity_y = sps.eye(m, format="csr")
    laplacian = (
        sps.kron(identity_x, one_dim_y, format="csr")
        + sps.kron(one_dim_x, identity_y, format="csr")
    )
    ## compute tau and alpha
    #tau, alpha = 1, 1
    Q = (sps.eye(n * m, format="csr") + rho * laplacian).tocsc()
    Q = Q.T @ Q
    e = np.zeros(Q.shape[0])
    e[ (n//2)*m + m//2] = 1

    kernel = sparse_linalg.spsolve(Q, e)
    iv2 = np.sum(kernel * e)
    return (iv2 / v2) * Q

def precision_matern_9pt(n, tau=1.0, alpha=0.2):
    """
    Sparse precision Q = tau I + alpha L for an n x n grid.
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

    return (tau * sps.eye(n * n, format="csr") + alpha * laplacian).tocsc()

    
## --> TODO: move out of this module into research experiments module

