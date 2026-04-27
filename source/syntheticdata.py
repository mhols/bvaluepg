__doc__="A collection of simples synthetic data and some helper methods"

import numpy as np
import matplotlib.pyplot as plt
import polyagammadensity as pgd
import covariance_kernels as ck

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

    x, y = np.meshgrid( np.arange(m), np.arange(n))
    x = image_to_scanorder(x)
    y = image_to_scanorder(y)

    d2 = ( (y[:, None] - y[None, :])**2  + (x[:,None] - x[None, :])**2)

    return v2 * np.exp(-d2/(2 * rho**2))  + 0.000001 * v2 * np.identity(n*m)


# here we define some mean value functions for synthetic data

def single_square(n, nn, a, b):

    res = a * np.ones((n,n))
    res[:nn, :nn] = b

    return res

def checkerboard(nn, ncheck, a, b):
    """
    Create a checkerboard where:
    nn     = block size (cells per square)
    ncheck = number of blocks per axis
    Total output size is (nn * ncheck) x (nn * ncheck).
    """
    n = nn * ncheck
    mask = ((np.indices((n, n)) // nn).sum(axis=0) % 2)
    return (a + b) / 2 + 0.5 * (a - b) * mask
    



def experiment_1(
    EstimatorClass=pgd.PolyaGammaDensity2D, 
    n=64, nn=20, a=4.5, b=5.5, rho=16, v2=0.1, lam=10, nmax_mix=60):


    # preparing data
    estim = EstimatorClass(n=n, m=n, lam=lam, nmax_mix=nmax_mix)

    ## from intensity to parameters
    aa = estim.f_from_field(a)
    bb = estim.f_from_field(b)

    # pm = single_square(n, nn, aa, bb)
    ncheck = 4
    assert n % ncheck == 0, 'n must be divisible by ncheck for checkerboard data'
    pm = checkerboard(n // ncheck, ncheck, aa, bb)
    covar = ck.spatial_covariance_matern_2_3(n, n, rho, v2)

    estim.set_prior_Gaussian(pm, covar)

    # Visualize the induced prior density on the Poisson intensity.
    # The Gaussian prior is placed on the latent field f, while the
    # Poisson intensity is obtained through the link function.
    plt.figure()
    plt.title('Induced prior density on Poisson intensity')
    ff = np.linspace(0.001, lam - 0.001, 2000)
    plt.plot(ff, estim.density_under_gaussian(ff, aa, v2), label='low region')
    plt.plot(ff, estim.density_under_gaussian(ff, bb, v2), label='high region')
    plt.xlabel('Poisson intensity')
    plt.ylabel('density')
    plt.legend()
    plt.grid(True)

    data = estim.random_events_from_field(estim.field_from_f(estim.prior_mean))
    estim.set_data(data)
    
    print('artificial data')
    plt.figure()
    plt.title('artificial Poisson rate')
    estim.imshow(estim.field_from_f(estim.prior_mean))
    print('...done')

    print('artificial catalog')
    plt.figure()
    plt.title('artificial Catalog observations')
    plt.gca().set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    plt.gca().margins(0)
    x, y = estim.random_catalog_from_nobs(estim.nobs)
    plt.plot(x, y, '.')
    print('...done')

    #plt.figure()
    #rp = estim.random_prior_parameters()
    #estim.imshow(rp)

    #plt.figure()
    #estim.imshow(estim.field_from_f(estim.prior_mean))

    print('binned observations')
    plt.figure()
    plt.title('Binned observations')
    plt.xticks([])
    plt.yticks([])
    estim.imshow(estim.nobs)
    print('...done')


    ## inversion

    #plt.figure()

    print('first guess gaussian aproximation')
    estim.set_data(data)
    estim.set_prior_Gaussian(estim.f_from_field(data.mean())*np.ones(estim.prior_mean.shape), ck.spatial_covariance_matern_2_3(n, n, rho, v2))

    fge = estim.first_guess_estimator()
    #estim.imshow(fge)

    plt.figure()
    plt.title('Gaussian estimation of field')
    plt.xticks([])
    plt.yticks([])
    estim.imshow(estim.field_from_f(fge))
    print()

    """
    print('Maximum Posterior Estimator')
    plt.figure()
    plt.title('Maximum posterior of field')
    plt.xticks([])
    plt.yticks([])
    mpe = estim.max_logposterior_estimator(method='CG')
    estim.imshow(estim.field_from_f(mpe))
    print('...done')
    """

    plt.figure()


    print('sampling 130 posterior')
    sres=0
    count = 0
    for i, res in enumerate(estim.sample_posterior(initial_f = fge, n_iter=130)):
        field = estim.field_from_f(res)
        sres += field
        if i%10==1 and count<12:
            plt.subplot(3,4,count+1)
            plt.xticks([])
            plt.yticks([])
            estim.imshow(field)
            count +=1

    plt.figure()
    plt.title('posterior mean')
    plt.xticks([])
    plt.yticks([])
    
    estim.imshow(sres/130)
    print('...done')


def experiment_2(nn=5, ncheck=5, a=1, b=2):
    tmp = checkerboard(nn, ncheck, a, b)
    plt.figure()
    plt.imshow(tmp)





if __name__ == "__main__":

    #experiment_1(EstimatorClass=pgd.RampDensity2D, nmax_mix=60 )
    experiment_1(EstimatorClass=pgd.PolyaGammaDensity2D)
    # experiment_2()


    plt.show()
