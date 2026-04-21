__doc__="a class to use the Polya-Gamma technique for density estimation"

import numpy as np
import scipy as sp
import scipy.linalg.blas as blas
import scipy.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm


from polyagamma import random_polyagamma

#import syntheticdata as sd
import matplotlib.pyplot as plt

#import polyagamma    ### TODO

def sigmoid(f):
    return 1/(1+np.exp(-f))   #TODO what if f gets big ????

def inv_sigmoid(u):
    return np.log(u/(1-u))

def der_sigmoid(f):
    return sigmoid(f)*sigmoid(-f)

def softplus(t):
    return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0) 

def inv_softplus(t):
    return np.log(np.expm1(t))

def sample_polya_gamma(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ziehen aus PG(b, c) unter Verwendung des Pakets „polyagamma“.

    """
    b = np.asarray(b, dtype=int)

    # Ensure positivity (polyagamma requires h >= 1)
    b = np.clip(b, 1, None)

    c = np.asarray(c, dtype=float)

    return random_polyagamma(h=b, z=c, method="saddle")

# # Idee Sigma0_inv_dot(v) mit L =pdg.Lprior
def sigma0_inv_dot(v, L):
    """Berechne Sigma0^{-1} @ v unter Verwendung der Cholesky-Zerlegung L von Sigma0."""
    # Rechne L @ y = v
    y = spla.solve_triangular(L, v, lower=True, trans=False)
    # Rechne L.T @ x = y
    x = spla.solve_triangular(L, y, lower=True, trans=True)
    return x

class Density:

    def __init__(self, prior_mean=None, prior_covariance=None, **kwargs):

        self.kwargs = kwargs
        self.set_prior_Gaussian(prior_mean, prior_covariance)


    def set_prior_Gaussian(self, prior_mean, prior_covariance):
        """
        """
        try:
            self.prior_mean = prior_mean.ravel()  ## make it one dimensional
        except:
            pass

        self.prior_covariance = prior_covariance
        self._Lprior = None

    def set_data(self, nobs):
        """
        Docstring for set_data
        
        :param self: Description
        :param nobs: array like the observed number of events in the bins
        """
        assert len(nobs) == self.nbins, "wrong dimension for nobs, must be like prior_mean"
        self.nobs = nobs
    
    @property
    def Lprior(self):
        if self._Lprior is None:
            self._Lprior = sp.linalg.cholesky(self.prior_covariance, lower=True)
        return self._Lprior

    @property
    def nbins(self):
        return self.prior_mean.shape[0]
    
    def random_events_from_field(self, field):
        """
        samples events from probability field
        
        :param self: Description
        :param field: Description
        """

        assert np.all( field >= 0 )

        return np.array( [np.random.poisson(l) for l in field] )

    
    def field_from_f(self, f):
        """
        the model that maps f to a Poissonian frequency field
        Must be overwritten by a Mixin

         
        :param self: Description
        :param f: Description
        """
        return f

    def f_from_field(self, field):
        """
        needs to be owerwritten by a mixin.
        Yields the inverse of the field_from_f
        
        :param self: Description
        :param f: Description
        """
        return field
    
    def density_under_gaussian(self, field, mu, gamma2):
        tmp = np.exp( - (self.f_from_field(field) - mu)**2/(2*gamma2))
        tmp /= np.abs(self.derivative_field_from_f(self.f_from_field(field)))

        return tmp / np.sum(tmp)

    def random_events_from_f(self, f):
        return self.random_events_from_field(self.field_from_f(f))

    def random_prior_parameters(self):
        """
        generates a random sample
        """
        f = np.dot(self.Lprior, np.random.normal(size=self.nbins))

        return self.prior_mean + f
    
    def random_prior_field(self):
        """
        a random realization of the underlying poissonian density in each bin
        """
        return self.field_from_f(self.random_prior_parameters())
    
    def random_prior_events(self):
        """
        samples counting data from the prior distribution
        """

        return self.random_events_from_field(self.random_prior_field())
    
    def loglikelihood(self, f):
        """
        returns the non-normalized log loglikelihood
        
        :param f: the paramters
        """
        field = self.field_from_f(f) ## self.lam * sigmoid(f)

        return np.sum(self.nobs * np.log(field)) - np.sum(field)
    
    def neg_logposterior(self, f):
        """
        Docstring for logposterior
        
        :param self: Description
        :param f: Description
        """
        return -self.loglikelihood(f) + np.sum( 
            sp.linalg.solve_triangular(
                self.Lprior, f-self.prior_mean, trans=False, lower=True)**2) / 2
 
    def neg_grad_logposterior(self, f):
        """
        Docstring for grad_logposterior
        1. Feld berechnen
        2. Gradient der Log-Likelihood berechnen
        3. Gradient der Log-Prior berechnen
        4. Beide Gradienten addieren und zurückgeben

        Fuer die Berechnugn des Gradienten der Log-Likelihood wird die Kettenregel angewendet:
        d/d f [ n * log(lam * sigmoid(f)) - lam * sigmoid(f) ]
        = n * (1 / (lam * sigmoid(f))) * lam * sigmoid(f) * (1 - sigmoid(f)) - lam * sigmoid(f) * (1 - sigmoid(f))
        = lam * (n / field - 1) * sigmoid(f) * (1 - sigmoid(f))

        linalg.solve wird verwendet, um den Gradient der Log-Prior zu berechnen:
        d/d f [ -1/2 * f^T * inv(Cov) * f ] = - inv(Cov) * f
        langsam, besser mit Cholesky-Faktorisierung in der naechsten Variante
        zum vergleichen
        """
       
        res = self.nobs * self.derivative_log_field_from_f(f) - self.derivative_field_from_f(f)
        tmp = sp.linalg.solve_triangular(self.Lprior, f-self.prior_mean, trans=False, lower=True) ### besser die schon berechneten Choleski Faktoren benutzen
        tmp = sp.linalg.solve_triangular(self.Lprior, tmp, trans=True, lower=True)

        return -res + tmp
    
    def first_guess_estimator(self, f=None, s2=None):
        """
        Uses a Gaussian approximation of the pixelwise f to obtain a first  
        guess for f


        :param self: Description
        :param fmean: Description
        :param fsigma: Description
        """

        """
        if Gaussian pixelwise proxy is given, use a baseline Poisson based proxy
        """

        if f is None:
            f = self.f_from_field(self.nobs) #np.clip(self.nobs, 1, None))

        if s2 is None:
            s2 = max(0.1, f) / self.derivative_field_from_f(f)**2

        D = np.diag(s2)

        tmp = f - self.prior_mean
        tmp = np.linalg.solve(self.prior_covariance + D, tmp)
        tmp = np.dot(self.prior_covariance, tmp)
        return self.prior_mean + tmp

        """
        D = diag(s2), G = self.prior_covariance, mu = self.priori_mean 

        compute

        tmp = D^{-1} f + G^{-1} mu
        """
        tmp = sp.linalg.solve_triangular(
            self.Lprior, self.prior_mean, lower=True, trans=False)
        tmp = sp.linalg.solve_triangular(self.Lprior, tmp, lower=True, trans=True)

        tmp += f/s2

        """
        compute (D^{-1} + G^{-1})^{-1} tmp
        using equivalent expression

        L ( L^{T} D^{-1} L + Id )^{-1} L^T tmp
        
        """

        ### TODO: replace by blas low lewel methods for matrix multiplication
        T = np.dot( self.Lprior.T , 1/s2[:, None] * self.Lprior)
        np.fill_diagonal(T, np.diagonal(T) + 1)
        tmp = np.dot(self.Lprior.T, tmp)
        tmp = np.linalg.solve(T, tmp)  ## TODO use the fact that T.T = T
        tmp = np.dot(self.Lprior, tmp)

        return tmp
    
    
    def max_logposterior_estimator(self, f0=None, method='Powell', niter=10000, eps=1e-6):
        """
        Docstring for grad_scipy_logposterior
        
        :param self: Description
        :param f: Description
        """
        #f = np.asarray(f, dtype=float)
        #assert f.shape[0] == self.nbins, "falsche dimension for f"

        # approx_fprime in scipy erwartet scalar function 

        ### kann auch direkt self.logposterior sein denke ich
        #def _fun(x):
        #    x = np.asarray(x, dtype=float)
        #    return float(self.neglogposterior(x))
    
        # Verwendet Schrittweiten pro Koordinate (Skalar eps, der an einen Vektor übertragen wird).
        #epsilon = np.full_like(f, float(eps), dtype=float)

        #### wieso nennst Du das Ergebnis grad ???

        #grad = sp.optimize.approx_fprime(f, _fun, epsilon)


        if f0 is None:
            f0 = self.first_guess_estimator()

        res = sp.optimize.minimize(
                self.neg_logposterior, 
                f0, 
                jac=self.neg_grad_logposterior, 
                method=method,
                #bounds = [ (-5, 5) for i in range(self.nbins) ],
                options={'maxiter':niter }) 

        print(res) 
        return res['x']
        #return np.asarray(grad, dtype=float).reshape(-1)  ## what is this ?

class SigmoidMixin:

    def __init__(self, prior_mean, prior_covariance, **kwargs):
        self.last_sample = None
        super().__init__(prior_mean, prior_covariance, **kwargs)

    @property
    def lam(self):
        return self.kwargs['lam']

    #def field_from_f(self, f):
    #
    #    return self.lam * sigmoid(f)
    
    def derivative_field_from_f(self, f):
        return self.lam * sigmoid(f) * sigmoid(-f)
    
    def derivative_log_field_from_f(self, f):
        return sigmoid(-f) 
       
    def sample_polyagamma_cond_f(self):

        field = self.field_from_f(-self.f)
        kk = self.random_events_from_field(field)  ### the random events k given f
    
    def field_from_f(self, f):

        return self.lam * sigmoid(f)
    
    def f_from_field(self, field):
        s = field / self.lam
        return np.log(s / (1-s))
    
    def derivative_field_from_f(self, f):
        return self.lam * sigmoid(f) * sigmoid(-f)
    
    def derivative_log_field_from_f(self, f):
        return sigmoid(-f)
 
        
    def first_guess_estimator(self):

        field = np.clip(self.nobs + 1, 0.5, 0.999 * self.lam)
        
        f = self.f_from_field( field )
        v = self.derivative_field_from_f(f)


        return super().first_guess_estimator( f, field/v**2) 
        

    def sample_posterior(self,
        n_iter: int,
        burn_in: int = 0,
        thin: int = 1,
        initial_f: np.ndarray | None = None,
        random_seed: int | None = None
    ):
        if random_seed is not None:
            np.random.seed(random_seed)

        if initial_f is None and self.last_sample is None:
            initial_f = 0
        elif not self.last_sample is None:
            initial_f = self.last_sample
        else:
            pass

        nbins = self.nbins
        # Precompute the prior precision matrix and its product with the prior mean
        mu0 = self.prior_mean
        # Precompute the inverse once (symmetric, positive definite)
        # Sigma0_inv = np.linalg.inv(Sigma0)
        # L = pgd._Lprior
        L = np.asarray(self.Lprior, dtype=float)
        I = np.eye(self.nbins)

        X = spla.solve_triangular(L, I, lower=True)         # X = L^{-1}
        Sigma0_inv = spla.solve_triangular(L.T, X, lower=False)  # (L^T)^{-1} @ L^{-1}


        Sigma0_inv_mu0 = Sigma0_inv @ mu0

        # Initialiesieren
        if initial_f is None:
            f = mu0.copy()
        else:
            f = np.asarray(initial_f, dtype=float).copy()
            if f.shape != mu0.shape:
                raise ValueError("initial_f must have shape matching prior_mean")

        # Samples speichern
        n_keep = max(0, (n_iter - burn_in) // thin)
        f_samples = np.zeros((n_keep, nbins))

        # Gibbs loop
        # sample_idx = 0
        for it in range(n_iter):
            # --- Step 1: sample k gegenben f ---
            # Die Rate für die latenten „negativen“ Zählungen ist lam * sigmoid(-f)
            rate_neg = self.field_from_f(-f)
            k = np.random.poisson(rate_neg)

            # --- Step 2: sample w gegeben f und counts ---
            b_counts = (self.nobs + k).astype(int)  # Gültigen PG-Formparameter sicherstellen
            w = sample_polya_gamma(b_counts, f)

            # --- Step 3: sample f gegeben w, k ---
            # Kappa berechnen
            kappa = 0.5 * (self.nobs - k)
            # Posterior-Präzision und Mittelwert
            A = Sigma0_inv + np.diag(w)
            # Rechte Seite: Sigma0_inv * mu0 + kappa
            bvec = Sigma0_inv_mu0 + kappa
            # Löse A m = bvec für m (posteriorer Mittelwert)
            # Cholesky-Faktorisierung für numerische Stabilität.
            chol = np.linalg.cholesky(A)
            # Löse den Mittelwert unter Verwendung der Cholesky-Faktoren.
            # Löse zunächst L y = bvec.
            y = spla.solve_triangular(chol, bvec, lower=True, trans=False)
            # Löse L^T m = y
            # m = spla.solve_triangular(chol.T, y, lower=False)
            m = spla.solve_triangular(chol, y, lower=True, trans=True)

            # Ziehe eine zufällig aus N(0, A^{-1})
            z = np.random.normal(size=nbins)
            # Löse L^T x = z für x, sodass x ~ N(0, A^{-1})
            eps = spla.solve_triangular(chol.T, z, lower=False)
            f = m + eps

            # Speichern, wenn burn_in abgeschlossen ist und bei Ausdünnungsintervall
            if it >= burn_in and ((it - burn_in) % thin == 0):
                #f_samples[sample_idx] = f
                # sample_idx += 1
                self.last_sample = f
                yield f
        return


class SmoothRampMixin:


    def field_from_f(self, f):
        return softplus(f)
    
    def f_from_field(self, field):
        return np.log(np.exp(field) - 1) 
                    
    def derivative_field_from_f(self, f):
        return sigmoid(f) 
    
    def derivative_log_field_from_f(self, f):
        return sigmoid(f) / self.field_from_f(f) 
       
    def first_guess_estimator(self):
        f = np.clip(self.nobs+1, 1, None)
        s2 = f * (1-np.exp(-f))**2
        return super().first_guess_estimator(f, s2)
      
    def sample_polyagamma_cond_f(self):

        field = self.field_from_f(-self.f)
        kk = self.random_events_from_field(field)  ### the random events k given f

        # self.polya = [ polyagamma( n + k , f) for n, k, f in zip(self.nobs, kk, self.f)]

        return 
 

    def sample_f_cond_polyagamma(self):
        pass


class PolyaGammaDensity(SigmoidMixin, Density):

    def __init__(self, prior_mean=None, prior_covariance=None, **kwargs):
        super().__init__(prior_mean, prior_covariance, **kwargs)


class RampDensity(SmoothRampMixin, Density):
    def __init__(self,prior_mean=None, prior_covariance=None,  **kwargs):
        super().__init__(prior_mean, prior_covariance, **kwargs)

class Mixin2D:

    #def __init__(self, **kwargs):
    #    self.n = kwargs['n']
    #    self.m = kwargs['m']
    #
    #    assert self.n * self.m == self.prior_mean.shape[0], "wrong dimension n, m"
    #    return super().__init__(**kwargs)
    
    @property
    def n(self):
        return self.kwargs['n']
    
    @property
    def m(self):
        return self.kwargs['m']
    
    @classmethod
    def image_to_scanorder(cls, image):
        assert len(image.shape)==2, 'image must be 2 dimensional'
        return image.ravel()

    def scanorder_to_image(self, linear_image, n=None, m=None):
        if n is None:
            n = self.n
        if m is None:
            m = self.m
        assert len(linear_image) == n*m, 'number of elements do not correspond'
        return np.reshape(linear_image, (n, m))
    
    def random_catalog_from_nobs(self, nobs):

        nobs = self.scanorder_to_image(nobs.ravel())

        x = []
        y = []

        for i in range(self.n):
            for j in range(self.m):
                for k in range(nobs[i,j]):
                    x.append(i+np.random.uniform())
                    y.append(j+np.random.uniform())
        return np.array(x), np.array(y)




    def to_image(self, d):
        self.scanorder_to_image(d, self.n, self.m)
    
    def imshow(self, d):
        plt.imshow( self.scanorder_to_image(d, self.n, self.m))



class PolyaGammaDensity2D(Mixin2D, PolyaGammaDensity):
    """
    Docstring for PolyGammaDensity2D
    specialized in 2Dimensional data
    """
    def __init__(self, prior_mean=None, prior_covariance=None, **kwargs):
        super().__init__(prior_mean, prior_covariance, **kwargs)

   
class RampDensity2D(Mixin2D, RampDensity):
    def __init__(self, prior_mean=None, prior_covariance=None, **kwargs):
        super().__init__(prior_mean, prior_covariance, **kwargs)
    

    


    

if __name__ == '__main__':

    import syntheticdata as sd
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt


    n, m = 20, 20

    pm = 5
    lam = 20   ## then mean(n) = lam/2
    gam = 5 #16 * 20**2 / lam**2
    rho = 3

    kwargs={'lam' : lam, 'n': n, 'm': m}
    
    DensityClass = PolyaGammaDensity2D ##RampDensity

    #DensityClass = RampDensity2D

    pgd = DensityClass(
        prior_mean = pm * np.ones( n * m),
        prior_covariance=sd.spatial_covariance_gaussian(n, m, rho, gam),
        lam=lam,
        n=n,
        m=m
    )

    plt.figure()
    plt.title('density of distribution of field')
    ff = np.linspace(0, lam, 100000)[1:-1]
    plt.plot( ff, DensityClass(**kwargs).density_under_gaussian(ff, pm, gam))
    plt.show()

    prior = pgd.random_prior_parameters()

    plt.figure()
    plt.title('prior parameter f')
    plt.imshow( sd.scanorder_to_image( prior, n, m ).T)



    prior_field = pgd.field_from_f(prior)

    plt.figure()
    plt.title('frequency field')
    plt.imshow( sd.scanorder_to_image(prior_field, n, m).T)


    events = pgd.random_events_from_field(prior_field)
    plt.figure()
    plt.title('random events')
    plt.imshow( sd.scanorder_to_image(events, n, m).T)

    pgd.set_data(events)
    print(np.min(events), np.max(events))

    

    plt.figure()
    plt.title('first guess')
    fg = pgd.first_guess_estimator()
    pgd.imshow(fg)

    grad = pgd.neg_grad_logposterior(prior)

    plt.figure()
    plt.title("Gradient log-posterior von prior sample")
    plt.imshow( sd.scanorder_to_image( grad, n, m ).T)  

    
    #%%
    '''compare with scipy approx
    vorher logposterior anpassen
    '''
    #%%


    #res = pgd.max_logposterior_estimator()

    res = pgd.max_logposterior_estimator(fg, method='CG', niter=8000)
    
    plt.figure()
    plt.title(f"max_posterior estimate of field")
    plt.imshow( sd.scanorder_to_image( pgd.field_from_f(res), n, m ).T)

    #res = pgd.max_logposterior_estimator(res, niter=100, method='CG')

    #plt.figure()
    #plt.title("additional CG step for max_posterior estimate of field")
    #plt.imshow( sd.scanorder_to_image( pgd.field_from_f(res), n, m ).T)




    plt.figure()
    plt.title('relative error params')
    plt.imshow( sd.scanorder_to_image( (res/prior -1)*100, n, m ).T)


    plt.figure()
    plt.title('gradient')
    plt.imshow( sd.scanorder_to_image( pgd.neg_grad_logposterior(res), n, m ).T)

    for res in pgd.sample_posterior(initial_f=res, n_iter=10):
        plt.figure()
        pgd.imshow(res)


    plt.show()
