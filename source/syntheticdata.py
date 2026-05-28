__doc__="A collection of simples synthetic data and some helper methods"

import matplotlib.pyplot as plt
import polyagammadensity as pgd
import covariance_kernels as ck
import numpy as np
import scipy.sparse as sps

#def image_to_scanorder(image):
#    assert len(image.shape)==2, 'image must be 2 dimensional'
#    return image.ravel()

#def scanorder_to_image(linear_image, n, m):
#    assert len(linear_image) == n*m, 'number of elements do not correspond'
#    return np.reshape(linear_image, (n, m))


# def spatial_covariance_gaussian(n, m, rho, v2):
#     """
#     Docstring for spatial_covariance_gaussian
    
#     :param n: number of gridpoints along "x-axis"
#     :param m: number of gridpoints along "y-axis"
#     :param rho: the spatial covariance (in units of integer grid)
#     :param v: autocorrelation at origin (i.e. the "amplitude")
#     """

#     x, y = np.meshgrid( np.arange(m), np.arange(n))
#     x = image_to_scanorder(x)
#     y = image_to_scanorder(y)

#     d2 = ( (y[:, None] - y[None, :])**2  + (x[:,None] - x[None, :])**2)

#     return v2 * np.exp(-d2/(2 * rho**2))  + 0.000001 * v2 * np.identity(n*m)


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
    return a*mask + b*(1-mask)


def grid_precision_laplacian(n, tau=1.0, alpha=0.2):
    """
    Sparse precision Q = tau I + alpha L for an n x n grid.
    """
    one_dim = sps.diags(
        [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    identity = sps.eye(n, format="csr")
    laplacian = (
        sps.kron(identity, one_dim, format="csr")
        + sps.kron(one_dim, identity, format="csr")
    )
    return (tau * sps.eye(n * n, format="csr") + alpha * laplacian).tocsc()
    
## --> TODO: move out of this module into research experiments module

def experiment_1(
    EstimatorClass=pgd.PolyaGammaDensity2D, 
    n=64, nn=20, a=3.5, b=6.5, rho=16, v2=0.1, lam=10, nmax_mix=60):
    # n=64, nn=20, a=3.5, b=6.5, rho=16, v2=0.95, lam=10, nmax_mix=60):


    np.random.seed(0)

    # preparing data
    estim = EstimatorClass(n=n, m=n, lam=lam, nmax_mix=nmax_mix)

    ## from intensity to parameters
    aa = estim.f_from_field(a)
    bb = estim.f_from_field(b)

    # pm = single_square(n, nn, aa, bb)
    ncheck = 4
    assert n % ncheck == 0, 'n must be divisible by ncheck for checkerboard data'
    tm = checkerboard(n // ncheck, ncheck, aa, bb)

    pm = np.mean(tm) * np.ones(n*n)

    covar = ck.spatial_covariance_matern_2_3(n, n, rho, v2)

    estim.set_prior_Gaussian(prior_mean=pm, prior_covariance=covar, sparse=False)

    #print(estim.get_prior_precision() @ estim.prior_covariance)

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

    data = estim.random_events_from_field(estim.field_from_f(tm))
    print(data.shape, pm.shape)
    estim.set_data(data.ravel())

   
    print('artificial data')
    plt.figure()
    estim.imshow(estim.field_from_f(tm.ravel()))
    print('...done')

    print('artificial catalog')
    plt.figure()
    plt.title('artificial Catalog observations')
    plt.gca().set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    plt.gca().margins(0)
    x, y = estim.random_catalog_from_nobs(estim.nobs)
    plt.plot(x, y, '.', markersize=1)
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

    plt.figure()

    print('first guess gaussian aproximation')

    fge = estim.first_guess_estimator()
    plt.title('first guess estimation of field exp1')
    estim.imshow(estim.field_from_f(fge))


    fml = estim.max_logposterior_estimator(niter=1000, method='TNC')

    plt.figure()
    plt.title('MAP estimator of field')
    plt.xticks([])
    plt.yticks([])
    estim.imshow(estim.field_from_f(fml))
    print('...done')

    print('prior Poissonian rate')
    plt.figure()
    plt.title('prior density under Gaussian of lams')
    lams = np.linspace(0, 2*estim.field_from_f(estim.prior_mean.max()), 10000)[1:-1]
    plt.plot(lams, estim.density_under_gaussian(lams,estim.prior_mean.mean(), v2))
    print('...done')
 
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


    np.random.seed(0)
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


def experiment_1_sparse_precision(
    EstimatorClass=pgd.PolyaGammaDensity2D,
    n=64,
    nn=20,
    a=3.5,
    b=6.5,
    tau=1.0,
    alpha=0.2,
    lam=10,
    nmax_mix=60,
    rho=1,
    v2=1
):
    """
    Sparse-precision variant of experiment_1.

    The synthetic data generation stays checkerboard-based, but the Gaussian
    prior is set as f ~ N(mu, Q^{-1}) with sparse grid precision Q.
    """

    np.random.seed(0)

    sparse_precision_estimators = (
        pgd.PolyaGammaDensity,
        pgd.RampDensity,
        pgd.ExponentialDensity,
    )
    if not issubclass(EstimatorClass, sparse_precision_estimators):
        raise ValueError(
            "experiment_1_sparse_precision currently requires "
            "PolyaGammaDensity, RampDensity, ExponentialDensity, or their 2D "
            "variants."
        )
    # preparing data
    estim = EstimatorClass(n=n, m=n, lam=lam, nmax_mix=nmax_mix)

    ## from intensity to parameters
    aa = estim.f_from_field(a)
    bb = estim.f_from_field(b)

    # pm = single_square(n, nn, aa, bb)
    ncheck = 4
    assert n % ncheck == 0, 'n must be divisible by ncheck for checkerboard data'
    tm = checkerboard(n // ncheck, ncheck, aa, bb)
    
    pm = np.mean(tm) * np.ones(n*n)


    precision =  np.linalg.inv(ck.spatial_covariance_matern_2_3(n, n, rho, v2))
    estim.set_prior_Gaussian(prior_mean=pm, prior_precision=precision, sparse=True)


    data = estim.random_events_from_field(estim.field_from_f(tm))
    estim.set_data(data.ravel())

    print('sparse precision prior')
    #print(f'grid: {n} x {n}; N={n*n}; nnz={precision.nnz}; density={precision.nnz / (n*n)**2:.6g}')

    print('artificial data')
    plt.figure()
    estim.imshow(estim.field_from_f(tm.ravel()))
    print('...done')

    print('artificial catalog')
    plt.figure()
    plt.title('artificial Catalog observations')
    plt.gca().set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    plt.gca().margins(0)
    x, y = estim.random_catalog_from_nobs(estim.nobs)
    plt.plot(x, y, '.', markersize=1)
    print('...done')

    print('binned observations')
    plt.figure()
    plt.title('Binned observations')
    plt.xticks([])
    plt.yticks([])
    estim.imshow(estim.nobs)
    print('...done')

    print('first guess gaussian aproximation')
    fge = estim.first_guess_estimator()

    plt.figure()
    plt.title('First guess estimation of field exp1_sparse')
    plt.xticks([])
    plt.yticks([])
    estim.imshow(estim.field_from_f(fge))
    print('...done')


    fml = estim.max_logposterior_estimator(niter=1000, method='TNC')

    plt.figure()
    plt.title('MAP estimator field')
    plt.xticks([])
    plt.yticks([])
    estim.imshow(estim.field_from_f(fml))
    print('...done')


    plt.figure()
    np.random.seed(0)

    print('sampling 130 posterior with sparse precision')
    sres = 0
    count = 0

    for i, res in enumerate(estim.sample_posterior(initial_f=fge, n_iter=130)):
        field = estim.field_from_f(res)
        sres += field
        if i % 10 == 1 and count < 12:
            plt.subplot(3, 4, count + 1)
            plt.xticks([])
            plt.yticks([])
            estim.imshow(field)
            count += 1

    plt.figure()
    plt.title('posterior mean')
    plt.xticks([])
    plt.yticks([])
    estim.imshow(sres / 130)
    print('...done')


def experiment_2(nn=5, ncheck=5, a=1, b=2):
    tmp = checkerboard(nn, ncheck, a, b)
    plt.figure()
    plt.imshow(tmp)


if __name__ == "__main__":

    # experiment_1(EstimatorClass=pgd.ExponentialDensity2D, nmax_mix=60 )
    # experiment_1_sparse_precision(EstimatorClass=pgd.PolyaGammaDensity2D, nmax_mix=60, tau=1.0, alpha=0.2)
    experiment_1_sparse_precision(EstimatorClass=pgd.PolyaGammaDensity2D, n=4, nmax_mix=60, tau=1.0, alpha=0.2, rho=5, v2=1)
    experiment_1(EstimatorClass=pgd.PolyaGammaDensity2D, nmax_mix=60, n=4, rho=5, v2=1)
    
    #experiment_1(EstimatorClass=pgd.PolyaGammaDensity2D, n=64, nn=8, a=1.0, b=1.5, rho=16, v2=0.1, lam=10, nmax_mix=60)
    # experiment_2()


    plt.show()
