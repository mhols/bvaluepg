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

def single_square(n, m, nn, a, b):

    res = a * np.ones((n,m))
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

# def grid_precision_laplacian(n, tau=1.0, alpha=0.2):
#     # Sparse precision Q = tau I + alpha L for an n x n grid.
#     one_dim = sps.diags(
#         [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
#         offsets=[-1, 0, 1],
#         format="csr",
#     )
#     identity = sps.eye(n, format="csr")
#     laplacian = (
#         sps.kron(identity, one_dim, format="csr")
#         + sps.kron(one_dim, identity, format="csr")
#     )
#     return (tau * sps.eye(n * n, format="csr") + alpha * laplacian).tocsc()

# def grid_precision_laplacian_9pt(n, tau=1.0, alpha=0.2):
#     """
#     Sparse precision Q = tau I + alpha L for an n x n grid.
#     Uses a 9-point Laplacian stencil with diagonal neighbors.

#     Stencil references:
#     https://en.wikipedia.org/wiki/Nine-point_stencil
#     https://notebook.community/eramirem/numerical-methods-pdes/05_elliptic
#     https://scicomp.stackexchange.com/questions/37656/tensor-product-representation-for-the-9-point-finite-difference-approximations-f
#     """
#     one_dim = sps.diags(
#         [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
#         offsets=[-1, 0, 1],
#         format="csr",
#     )
#     neighbor_1d = sps.diags(
#         [np.ones(n - 1), np.ones(n - 1)],
#         offsets=[-1, 1],
#         format="csr",
#     )
#     identity = sps.eye(n, format="csr")

#     cardinal_laplacian = (
#         sps.kron(identity, one_dim, format="csr")
#         + sps.kron(one_dim, identity, format="csr")
#     )
#     diagonal_neighbors = sps.kron(neighbor_1d, neighbor_1d, format="csr")

#     laplacian = (
#         4.0 * cardinal_laplacian
#         + 4.0 * sps.eye(n * n, format="csr")
#         - diagonal_neighbors
#     ) / 6.0

#     return (tau * sps.eye(n * n, format="csr") + alpha * laplacian).tocsc()
    
# ## --> TODO: move out of this module into research experiments module
# """

class Experiment:
    """
    direct prescription of observations
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_nobs(self):
        if self.kwargs['type'] == 'A':
            return np.array(self.kwargs['data'], dtype='float')
        elif self.kwargs['type'] == 'B':
            field = self.kwargs['data']
            return np.random.poisson(field)
        elif self.kwargs['type'] == 'C':
            f = self.estim.f_from_field(self.kwargs['data'])
            return self.estim.random_prior_events(f)
        else:
            pass

    @property
    def nobs(self):
        if not hasattr(self, '_nobs'):
            self._nobs= self.get_nobs()
            self.estim.set_data(self._nobs)
        return self._nobs
    
    @property
    def estim(self):
        if not hasattr(self, '_estim'):
            self._estim = self.kwargs['EstimatorClass'](**self.kwargs)
            self._estim.set_data(self.nobs)
            try:
                self.estim.set_prior_Gaussian(**self.kwargs)
            except:
                pass

        return self._estim
    
    @property
    def nm(self):
        return self.kwargs['n'], self.kwargs['m']
    
    def plot_catalog(self, title="artificial catalog"):
        plt.figure()
        plt.title(title)
        plt.gca().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        plt.gca().margins(0)
        x, y = self.estim.random_catalog_from_nobs(self.estim.nobs)
        plt.plot(x, y, '.', markersize=1)

    def plot_data(self, title="artificial data", **kwargs):
        plt.figure()
        plt.title(title)
        plt.gca().set_aspect('equal')
        self.estim.imshow(self.estim.nobs)

    @property
    def map_estimator(self):
        if not hasattr(self, '_map_estimator'):
            self._map_estimator = self.estim.max_logposterior_estimator()
        return self._map_estimator
    
    def plot_map_estimator_f(self, title="MAP estimator"):
        plt.figure()
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        self.estim.imshow(self.map_estimator, vmin=self.kwargs['vmin_f'], vmax=self.kwargs['vmax_f'])

    def plot_map_estimator_field(self, title="MAP estimator"):
        plt.figure()
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        self.estim.imshow(self.estim.field_from_f(self.map_estimator), vmin=self.kwargs['vmin_field'], vmax=self.kwargs['vmax_field'])

    def plot_posterior_f(self, title="posterior"):

        plt.figure()
        plt.title(title)
        np.random.seed(0)
        sres=0
        count = 0

        estim = self.estim

        for i, res in enumerate(estim.sample_posterior(
                initial_f=self.map_estimator, n_iter=130)):
            sres += res

            if i % 10 == 1 and count < 12:
                plt.subplot(3, 4, count + 1)
                plt.xticks([])
                plt.yticks([])
                estim.imshow(res)
                count += 1

        plt.figure()
        plt.title(title + 'mean')
        plt.xticks([])
        plt.yticks([])
    
        estim.imshow(sres/130, vmin=self.kwargs['vmin_f'], vmax=self.kwargs['vmax_f'])
        print('...done')

    def plot_posterior_field(self, title="posterior"):

        plt.figure()
        plt.title(title)
        np.random.seed(0)
        sres=0
        count = 0

        estim = self.estim

        for i, res in enumerate(estim.sample_posterior(
                initial_f=self.map_estimator, n_iter=130)):
            sres += self.estim.field_from_f(res)

            if i % 10 == 1 and count < 12:
                plt.subplot(3, 4, count + 1)
                plt.xticks([])
                plt.yticks([])
                estim.imshow(self.estim.field_from_f(res), vmin=self.kwargs['vmin_field'], 
                             vmax=self.kwargs['vmax_field'] )
                count += 1

        plt.figure()
        plt.title(title + 'mean')
        plt.xticks([])
        plt.yticks([])
    
        estim.imshow(sres/130, vmin=self.kwargs['vmin_field'], vmax=self.kwargs['vmax_field'])
        print('...done')

           



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
    print(rho, v2)
    #covar = np.diag(np.arange(n*n)+1)

    estim.set_prior_Gaussian(prior_mean=pm, prior_covariance=covar, sparse=True)

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

    print('data', estim.nobs)
   
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

    for i, res in enumerate(estim.sample_posterior(initial_f=np.ones(n*n), n_iter=130)):
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
    
    estim.imshow(sres/130)
    print('...done')


def experiment_1_sparse_precision(
    EstimatorClass=pgd.PolyaGammaDensity2D,
    n=264,
    nn=264,
    a=3.5,
    b=4.0,
    tau=1.0,
    alpha=0.2,
    lam=10,
    nmax_mix=60,
    rho=1,
    v2=1,
    stencil="9pt",
    boundary="zero"
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

    tm = 0 * tm 


        

    # pm = np.mean(tm) * np.ones(n*n)
    # covar = ck.spatial_covariance_matern_2_3(n, n, rho, v2)

    # print(rho, v2)

    #covar = np.diag(np.arange(n*n)+1)
    # estim.set_prior_Gaussian(prior_mean=pm, prior_precision=np.linalg.inv(covar), sparse=True)
    # data = estim.random_events_from_field(estim.field_from_f(tm))
    # estim.set_data(data.ravel())

    pm = np.mean(tm) * np.ones(n*n)



    if stencil == "5pt":
      precision = ck.precision_matern(n, n, rho, v2, boundary=boundary)
    elif stencil == "9pt":
      precision = grid_precision_laplacian_9pt(n, tau=tau, alpha=alpha, boundary=boundary)
    else:
      raise ValueError("stencil must be '5pt' or '9pt'")

    estim.set_prior_Gaussian(prior_mean=pm, prior_precision=precision, sparse=True)
    data = 0*estim.random_events_from_field(estim.field_from_f(tm))
    estim.set_data(data.ravel())    


    print('data', estim.nobs)
    print('sparse precision prior')
    #print(f'grid: {n} x {n}; N={n*n}; nnz={precision.nnz}; density={precision.nnz / (n*n)**2:.6g}')

    print('artificial data')
    plt.figure()
    plt.title('field')
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

    print('computing max logposterior estimator')
    fml = estim.max_logposterior_estimator(f0=fge, niter=1000, method='TNC')

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

    for i, res in enumerate(estim.sample_posterior(initial_f=np.ones(n*n), n_iter=26)):
        #field = estim.field_from_f(res)
        sres += res #field
        
        if i % 2 == 1 and count < 12:
            plt.subplot(3, 4, count + 1)
            plt.xticks([])
            plt.yticks([])
            estim.imshow(res)
            count += 1

    plt.figure()
    plt.title('posterior mean')
    plt.xticks([])
    plt.yticks([])
    estim.imshow(sres / 26)
    print('...done')


def experiment_2(nn=5, ncheck=5, a=1, b=2):
    tmp = checkerboard(nn, ncheck, a, b)
    plt.figure()
    plt.imshow(tmp)


if __name__ == "__main__":

    # experiment_1(EstimatorClass=pgd.ExponentialDensity2D, nmax_mix=60 )
    # experiment_1(EstimatorClass=pgd.PolyaGammaDensity2D, nmax_mix=60, n=4, rho=5, v2=1, boundary="symmetric")
    # experiment_1_sparse_precision(EstimatorClass=pgd.PolyaGammaDensity2D, nmax_mix=60, tau=1.0, alpha=0.2)
    # experiment_1_sparse_precision(EstimatorClass=pgd.PolyaGammaDensity2D, n=256, nmax_mix=60, tau=1.0, alpha=0.2, rho=5, v2=1, stencil="9pt")
    # experiment_1_sparse_precision(EstimatorClass=pgd.PolyaGammaDensity2D, n=500, nmax_mix=60, tau=1.0, alpha=0.2, rho=200, v2=1, stencil="5pt")

    # experiment_1_sparse_precision(EstimatorClass=pgd.RampDensity2D, n=500, nmax_mix=60, tau=1.0, alpha=0.2, rho=400, v2=1, stencil="5pt", boundary="symmetric")
    
    # experiment_1(EstimatorClass=pgd.PolyaGammaDensity2D, n=64, nn=8, a=1.0, b=1.5, rho=16, v2=0.1, lam=10, nmax_mix=60)
    # experiment_2()

    # Compare the effect of different boundary conditions on the precision matrix and resulting samples.
    # experiment_1_sparse_precision(EstimatorClass=pgd.RampDensity2D, n=500, nmax_mix=60, tau=1.0, alpha=0.2, rho=5, v2=1, stencil="5pt", boundary="zero")
    # experiment_1_sparse_precision(EstimatorClass=pgd.PolyaGammaDensity2D, n=128, nmax_mix=60, tau=1.0, alpha=0.2, rho=10, v2=1, stencil="5pt", boundary="zero")
    # experiment_1_sparse_precision(EstimatorClass=pgd.ExponentialDensity2D, n=128, nmax_mix=60, tau=1.0, alpha=0.2, rho=10, v2=1, stencil="5pt", boundary="zero")
    

    kwargs = dict(
    n = 67, #232 # 
    m = 59, #229
     rho = 4,
    v2 = 0.5,
    lam = 5,
    vmin_f = -3,
    vmax_f = 1,
    vmin_field = 0,
    vmax_field = 3.5
    )

    EstimatorClass = pgd.PolyaGammaDensity2D  ###gdd.ExponentialDensity2D ###pgd.PolyaGammaDensity2D ###pgd.RampDensity2D
    n, m = kwargs['n'], kwargs['m']
    data_one = np.ones(n * m)
    data_corner_strong = single_square(n, m, n//2, 1, 0.2)

    # choose data
    data = data_one

    # Covariance structures
    def Cov_data_matern_2_3():
        return dict(
            prior_mean=data, 
            prior_covariance=ck.spatial_covariance_matern_2_3(**kwargs),
            sparse=False
        ) 
    
    def Cov_one_matern_2_3():
        return dict(
            prior_mean= np.ones((n,m)), 
            prior_covariance=ck.spatial_covariance_matern_2_3(**kwargs),
            sparse=False
        ) 
    
    def Cov_one_matern_2_sparse():
        return dict(
            prior_mean= np.ones((n,m)), 
            prior_precision=ck.precision_matern(**kwargs),
            sparse=True
        ) 
  
    # choose Covariance Structure
    prior_covar = Cov_one_matern_2_sparse() ###Cov_one_matern_2_3() ###Cov_one_matern_2_sparse()


    A = Experiment(type='A', EstimatorClass=EstimatorClass, 
                     data=data, **prior_covar, **kwargs, random_seed=1)
    
    B = Experiment(type='B', EstimatorClass=EstimatorClass, 
                     data=data, **prior_covar,  **kwargs, random_seed=2)
    
    C = Experiment(type='C',  EstimatorClass=EstimatorClass, 
                     data=data, **prior_covar, **kwargs, random_seed=3)
    


    for E, T in zip([A, B, C], ['A', 'B', 'C']):     
        E.plot_map_estimator_field(f"map estimator exp {T}")
        E.plot_posterior_field(f"posterior {T}")

    plt.show()
