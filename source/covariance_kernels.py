import numpy as np
from polyagammadensity import Mixin2D 


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
