__doc__="A collection of simples synthetic data and some helper methods"

import numpy as np
import matplotlib.pyplot as plt

def image_to_scanorder(image):
    assert len(image.shape)==2, 'image must be 2 dimensional'
    return image.ravel()

def scanorder_to_image(linear_image, n, m):
    assert len(linear_image) == n*m, 'number of elements do not correspond'
    return np.reshape(linear_image, (n, m))


def spatial_covariance_gaussian(n, m, rho, v2):
    """
    Docstring for spatial_covariance_gaussian
    
    :param n: number of gridpoints along "x-axis"
    :param m: number of gridpoints along "y-axis"
    :param rho: the spatial covariance (in units of integer grid)
    :param v: autocorrelation at origin (i.e. the "amplitude")
    """

    x, y = np.meshgrid( np.arange(n), np.arange(m))
    x = image_to_scanorder(x)
    y = image_to_scanorder(y)

    d2 = ( (x[:, None] - x[None, :])**2  + (y[:,None] - y[None, :])**2)

    return v2 * np.exp(-d2/(2 * rho**2))


if __name__ == "__main__":

    n, m = 30, 30

    K = spatial_covariance_gaussian(n, m, 5, 1)
    L = np.linalg.choleski(K)

    random_field = scanorder_to_image(
        np.linalg.solve(L, np.random.normal(n * m))
    )


    plt.figure()
    plt.imshow(random_field)
    plt.show()

