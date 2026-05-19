__doc__="a class to use the Polya-Gamma technique for density estimation"

import numpy as np
import scipy as sp
from scipy.linalg import cho_solve
import scipy.linalg.blas as blas
import scipy.linalg as spla
import matplotlib.pyplot as plt
import gibbs_softplus_mixture as gsm
import exp_mix_explink as eme
from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
from collections.abc import Iterable


from polyagamma import random_polyagamma

#import syntheticdata as sd
import matplotlib.pyplot as plt

#import polyagamma    ### TODO

def to_numpy(x, dtype=float):
    if np.isscalar(x):
        return np.array([x], dtype=dtype)
    if isinstance(x, np.ndarray):
        return x.ravel()
    if isinstance(x, Iterable):
        return np.fromiter(x, dtype=float)
    return np.array([x], dtype=dtype)

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
## TODO into util.py

def sigma0_inv_dot(v, L):
    """Berechne Sigma0^{-1} @ v unter Verwendung der Cholesky-Zerlegung L von Sigma0."""
    # Rechne L @ y = v
    y = spla.solve_triangular(L, v, lower=True, trans=False)
    # Rechne L.T @ x = y
    x = spla.solve_triangular(L, y, lower=True, trans=True)
    return x

class Density:
    """
    Basis class for density computation
    """

    COVARIANCE=1
    PRECISION=0

    def __init__(self, prior_mean=None, prior_covariance=None, prior_precision=None, sparse=False, **kwargs):
        """
        
        """

        self.kwargs = kwargs
        self.sparse = sparse

        if (prior_covariance is None) and (prior_precision is None):
            return
        
        self.set_prior_Gaussian(prior_mean, prior_covariance, prior_precision, sparse)

    def set_prior_Gaussian(self, prior_mean=None, prior_covariance=None, prior_precision=None, sparse=False, **kwargs):

        if (not prior_covariance is None) and (not prior_precision is None):
            raise Exception("Not both, prior covariance or prior precision can be specified")
        
        if not prior_covariance is None:
            self.prior_covariance = prior_covariance
            n,m = prior_covariance.shape
            self._Lprior = None
            self.mode = Density.COVARIANCE    ### covariance Mome
            if not n==m:
                raise Exception("not a quadratic covariance / precision matrix")
        elif not prior_precision is None:
            self.prior_precision = prior_precision
            n,m = prior_covariance.shape
            self._Vprior = None
            self.mode = Density.PRECSION
            if not n==m:
                raise Exception("not a quadratic covariance / precision matrix")
        else:
            pass

        
       
        self.prior_mean = to_numpy(prior_mean)

        if not self.prior_mean.shape[0] == n:
            raise Exception("dimensions of prior mean and covariance / precision do not match")
        
        if sparse:
            self.cholesky = "sparse_cholesky_method"   ## TODO replace by method
            self.apply_prior_choleski_covar = "TODO"
        

        else:
            self.cholesky = lambda M: sp.linalg.cholesky(M, lower=True)
            
            ### Covar mathods
            self.apply_prior_choleski_covar = lambda f: self.Lprior @ f
            self.apply_prior_choleski_covar_inverse = lambda f: sp.linalg.solve_triangular(self.Lprior, f, trans=False, lower=True)
            self.apply_prior_choleski_covar_T = lambda f: self.Lprior.T @ f
            self.apply_prior_choleski_covar_inverse_T = lambda f: sp.linalg.solve_triangular(self.Lprior, f, trans=True, lower=True)
            #self.apply_prior_inverse_covar = self._apply_prior_inverse_covar

            ### Precision methods
            self.apply_prior_choleski_precision = lambda f: self.Vprior @ f
            self.apply_prior_choleski_precision_inverse = lambda f: sp.linalg.solve_triangular(self.Vprior, f, trans=False, lower=True)
            self.apply_prior_choleski_precision_T = lambda f: self.Vprior.T @ f
            self.apply_prior_choleski_precision_inverse_T = lambda f: sp.linalg.solve_triangular(self.Vprior, f, trans=True, lower=True)
            #self.apply_prior_inverse_precision = self._apply_prior_inverse_precision



    def set_data(self, nobs):
        """
        Docstring for set_data
        
        :param self: Description
        :param nobs: array like the observed number of events in the bins
        """
        assert len(nobs.ravel()) == self.nbins, "wrong dimension for nobs, must be like prior_mean"
        self.nobs = nobs.ravel()
        self.ndata = sum(self.nobs)
    
    @property
    def Lprior(self):
        """
        Lazy evaluation of Chlesky factor of prior covariance
        """
        if self._Lprior is None:
            self._Lprior = self.cholesky(self.prior_covariance)
        return self._Lprior
    
    @property
    def Vprior(self):
        """
        Lazy evaluation of Chlesky factor of prior covariance
        """
        if self._Vprior is None:
            self._Vprior = self.cholesky(self.prior_precision)
        return self._Vprior

    def get_prior_precision(self):
        """
        lazy method that should be avoided all together if dimension is larg
        """
        if not hasattr(self, 'prior_precision'):
            tmp = self.apply_prior_choleski_covar_inverse(np.identity(self.nbins))
            self.prior_precision = self.apply_prior_choleski_covar_inverse_T(tmp)
        
        return self.prior_precision
    
    def laplace_approximation_one_dimension(self, m, v2, n):
        """
        m: prior mean
        v2: prior variance
        n: observed outcome
        """
        m = np.array(np.array([m]))
        v2 = np.array([[v2]])
        calc = self.__class__(m, v2, prior_precision=None, sparse=False,  **self.kwargs)
        calc.set_data(np.array([n]))
        pm = calc.max_logposterior_estimator()
        pv2 = 1./calc.hessian_neg_log_posterior(pm)

        return pm[0], pv2[0,0]
    
    def posterior_f_one_dimension(self, f, pm, pv2, n):
        l = self.field_from_f(f)
        pd = np.exp(-(f-pm)**2/(2*pv2)) * l**n * np.exp(-l)
        return pd
    
    def posterior_field_one_dimension(self, field, pm, pv2, n):
        pd = self.density_under_gaussian(field, pm, pv2)
        pd = field**n * np.exp(-field) * pd
        return pd

    @property
    def nbins(self):
        return self.prior_mean.shape[0]
    
    def get_prior_mean(self):
        """
        shall be redefined in inheriting class for 2D data
        """
        return self.prior_mean
    
    @classmethod
    def random_events_from_field(cls, field):
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
        raise Exception('please implement me befor using me')

    def f_from_field(self, field):
        """
        needs to be owerwritten by a mixin.
        Yields the inverse of the field_from_f
        
        :param self: Description
        :param f: Description
        """
        raise Exception('please implement me befor using me')
    
    def derivative_log_field_from_f(self, f):
        raise Exception('please implement me befor using me')
    
    def second_derivative_log_field_from_f(self, f):
        raise Exception('please implement me befor using me')
    
    def derivative_field_from_f(self, f):
        raise Exception('please implement me befor using me')
    
    def second_derivate_field_from_f(self, f):
        raise Exception('please implement me befor using me')
    
    def density_under_gaussian(self, field, mu, gamma2):
        print('WARNING: depricated, replace by prior field')
        tmp = np.exp( - (self.f_from_field(field) - mu)**2/(2*gamma2))
        tmp /= np.abs(self.derivative_field_from_f(self.f_from_field(field)))

        return tmp / np.sum(tmp)
    
    def prior_single_bin_f(self, f, mu, gamma2):
        tmp = np.exp( - (f-mu)**2 / (2* gamma2))
        tmp /= np.sum(tmp)
        return tmp
    
    def prior_single_bin_field(self, field, mu, gamma2):
        return self.density_under_gaussian(field, mu, gamma2)
    
    def log_likelihood_single_obeservation_f(self, f, n):
        field = self.field_from_f(f)
        return self.log_likelihood_single_observation_field(field, n)

    def log_likelihood_single_observation_field(self, field, n):
        return n*np.log(field) - field

    def posterior_single_obeservation_field(self, field, mu, gamma2, n):
        tmp = self.log_likelihood_single_obeservation_field(field, mu, gamma2, n)
        tmp -= np.max(tmp)
        tmp = np.exp(tmp)
        tmp /= np.sum(tmp)

        return tmp
    
    def posterior_single_obeservation_f(self, f, mu, gamma2, n):
        tmp = self.log_likelihood_single_obeservation_f(f, mu, gamma2, n)
        tmp -= (f-mu)**2/(2*gamma2)
        tmp -= np.max(tmp)
        tmp = np.exp(tmp)
        tmp /= np.sum(tmp)

        return tmp

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
    
    #def apply_prior_choleski_covar_inverse(self, f):
    #
    #    return sp.linalg.solve_triangular(self.Lprior, f, trans=False, lower=True)
    
    #def apply_prior_choleski_covar_inverse_T(self, f):
    #    return sp.linalg.solve_triangular(self.Lprior, f, trans=True, lower=True)
    #
    #def apply_prior_choleski_covar(self, f):
    #    return self.Lprior @ f
    # 
    #def apply_prior_choleski_covar_T(self, f):
    #    return self.Lprior.T @ f
    # 
    def _apply_prior_inverse_covar(self, f):
        tmp = self.apply_prior_choleski_covar_inverse(f)
        tmp = self.apply_prior_choleski_covar_inverse_T(tmp)
        return tmp
    
    def _apply_prior_precision(self, f):
        return self.apply_prior_inverse_covar(f)
 
    def neg_grad_logposterior(self, f):
        """
        """
        res = self.nobs * self.derivative_log_field_from_f(f) - self.derivative_field_from_f(f)

        if self.mode == Density.COVARIANCE:
            tmp = self.apply_prior_choleski_covar_inverse(f-self.prior_mean)
            tmp = self.apply_prior_choleski_covar_inverse_T(tmp)
        else:
            tmp = self.apply_prior_choleski_precision(f-self.prior_mean)
            tmp = self.apply_prior_choleski_precision_T(tmp)
        
        return -res + tmp

    def hessian_neg_log_posterior(self, f):
        D = -np.diag( self.ndata * self.second_derivative_log_field_from_f(f) - self.second_derivate_field_from_f(f))

        if self.mode == Density.PRECISION:
            return D + self.prior_precision
        else:
            return D + self.get_prior_precision()


    def apply_hessian_neg_log_posterior_non_normalized(self, atf, tof):
        """
        applies the Hessian at atf to tof
        TODO: make more efficient
        """
        print('in hess')
        D = -np.diag( self.second_derivative_log_field_from_f(atf))
        tmp = D * self.Lprior
        LDL = self.Lprior.T @ tmp
        LDL = LDL +  np.eye(self.nbins)
        K = sp.linalg.cholesky(LDL, lower=True)
        tmp = self.apply_prior_choleski_covar_T(tof)
        tmp = sp.linalg.solve_triangular(K, tmp, trans=False, lower=True)
        tmp = sp.linalg.solve_triangular(K, tmp, trans=True, lower=True)
        tmp = self.apply_prior_choleski_covar(tmp)
        print('done hess')
        return tmp
    

    
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
        f = f.ravel()

        if s2 is None:
            s2 = self.nobs / self.derivative_field_from_f(f)**2

        D = np.diag(s2)

        tmp = f - self.prior_mean

        if self.mode == Density.COVARIANCE:
            if self.sparse:
                print('WARNING not implemented yet')
                tmp = np.linalg.solve(self.prior_covariance + D, tmp)
            else:
                tmp = np.linalg.solve(self.prior_covariance + D, tmp)
            tmp = self.prior_covariance @ tmp
            return self.prior_mean + tmp
        
        else:
            tmp = self.prior_precision @ tmp
            if self.sparse:
                tmp = np.linalg.solve( self.prior_precision + 1/D , tmp)
            else:
                tmp = np.linalg.solve( self.prior_precision + 1/D , tmp)
            return f - tmp

    
    
    def max_logposterior_estimator(self, f0=None, method='TNC', niter=1000, eps=1e-5, **kwargs):
        """
        Docstring for grad_scipy_logposterior
        
        :param self: Description
        :param f: Description
        """

        if f0 is None:
            f0 = self.first_guess_estimator()

        #bounds = [(m-4*s, m+4*s) for m, s in zip(self.prior_mean, np.sqrt(np.diag(self.prior_covariance)))]

        res = sp.optimize.minimize(
                self.neg_logposterior, 
                f0, 
                jac=self.neg_grad_logposterior, 
                method=method,
                hess=self.hessian_neg_log_posterior,
                # bounds=bounds,
                options={'maxiter':niter, 'maxfun': niter}) 

        print(res)
        return res['x']

class SigmoidMixin(Density):

    def __init__(self, prior_mean, prior_covariance, **kwargs):
        self.last_sample = None
        super().__init__(prior_mean, prior_covariance, **kwargs)

    @property
    def lam(self):
        return self.kwargs['lam']

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
        s = np.clip(s, 0.01, None)
        res = np.log(s / (1-s))
        return res
    
    def derivative_field_from_f(self, f):
        return self.lam * sigmoid(f) * sigmoid(-f)
    
    def second_derivate_field_from_f(self, f):
        return self.lam * sigmoid(f) * sigmoid(-f) * (sigmoid(-f) - sigmoid(f))
    
    def derivative_log_field_from_f(self, f):
        return sigmoid(-f)
 
    def second_derivative_log_field_from_f(self, f):
        return -sigmoid(f)*sigmoid(-f)

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
            initial_f = np.zeros(self.nbins) 
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
        ####
        ###    Du hast immer noch die Inverse von Sigma, brauchst Du aber nicht.
        ###    please eliminate as does Cristina
        ####

        if self.mode == Density.COVARIANCE:
            tmp = self.apply_prior_choleski_covar_inverse(mu0)
            tmp = self.apply_prior_choleski_covar_inverse_T(tmp)
        else:
            tmp = self.apply_prior_choleski_precision_T(mu0)
            tmp = self.apply_prior_choleski_precision(tmp) 
            
        Sigma0_inv_mu0 = tmp

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
            if self.mode==Density.COVARIANCE:
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
            
            else:
                pass
                ### TODO

            # Speichern, wenn burn_in abgeschlossen ist und bei Ausdünnungsintervall
            if it >= burn_in and ((it - burn_in) % thin == 0):
                #f_samples[sample_idx] = f
                # sample_idx += 1
                self.last_sample = f
                yield f
        return


class SmoothRampMixin:

    def __init__(self, prior_mean, prior_covariance, nmax_mix:int=60, cache_dir:Path=Path('.mixture'), softplus_k: float = 1.0, **kwargs) -> dict:
        self._mix = None
        self.nmax_mix = nmax_mix
        self.cache_dir = cache_dir 
        self.softplus_k = float(softplus_k)
        super().__init__(prior_mean, prior_covariance, **kwargs)

    @property
    def mix(self):
        """
        lazy computation of property
        """
        if self._mix is None:
            self._mix = gsm.load_or_build_mix(self.nmax_mix, self.cache_dir, self.softplus_k)
        return self._mix

    def field_from_f(self, f):
        k = self.softplus_k
        return softplus(k * f) / k
    
    def f_from_field(self, field):
        k = self.softplus_k
        return np.log(np.expm1(k * field)) / k 
                    
    def derivative_field_from_f(self, f):
        k = self.softplus_k
        return sigmoid(k * f)
    
    def second_derivate_field_from_f(self, f):
        return sigmoid(f)*sigmoid(-f)
    
    def derivative_log_field_from_f(self, f):
        return sigmoid(f) / softplus(f) 

    def second_derivative_log_field_from_f(self, f):
        tmp = softplus(f)
        return sigmoid(f)*sigmoid(-f) / tmp - sigmoid(f)**2/tmp**2

     
    def first_guess_estimator(self):
        field = np.clip(self.nobs + 1, 1e-8, None)

        f = self.f_from_field(field)
        v = self.derivative_field_from_f(f)

        s2 = field / (v**2 + 1e-15)
        return super().first_guess_estimator(f, s2)
      
 
    def sample_posterior(
        self,
        n_iter: int,
        burn_in: int = 0,
        thin: int = 1,
        initial_f: np.ndarray | None = None,
        random_seed: int | None = None):

        """
        TODO: introduce one more layer of classes to make this a generic mixin
        for Gaussian mixtures
        Generic Gibbs sampler for any density object with attributes:
        - prior_mean
        - Lprior
        - nobs
        - nbins
        - mix

        """
        if not random_seed is None:
            np.random.seed(random_seed)

        N = self.nbins

        if initial_f is None:
            try:
                f = self.first_guess_estimator()
            except Exception:
                f = self.prior_mean.copy()
        else:
            f = np.asarray(initial_f, dtype=float).copy()
            if f.shape != (N,):
                raise ValueError("initial_f must have shape (nbins,)")

        #
        # 
        # n_keep = max(0, (n_iter - burn_in) // thin)
        #f_samples = np.zeros((n_keep, N), dtype=float)

        total_iter = burn_in + n_iter * thin

        # prepare fixed linear algebra once
        fz_cache = gsm.prepare_f_cond_z(
            self.nobs,
            self.prior_mean,
            self.Lprior,
            self.mix,
        )


        idx = 0
        for it in range(total_iter):
            z = gsm.sample_z_cond_f(f, self.nobs, self.mix)
            #f = gsm.sample_f_cond_z(z, self.nobs, self.prior_mean, self.Lprior, self.mix)
            f = gsm.sample_f_cond_z_cache(z, fz_cache, self.mix)

            if it >= burn_in and ((it - burn_in) % thin == 0):
                #f_samples[idx] = f
                idx += 1
                yield f

        return
 
class ExponentialMixin:

    ### TODO:

    def __init__(
        self,
        prior_mean,
        prior_covariance,
        nmax_mix: int = 60,
        cache_dir: Path = Path(".mixture"),
        **kwargs,
    ):
        self._mix = None
        self.nmax_mix = int(nmax_mix)
        self.cache_dir = Path(cache_dir)
        super().__init__(prior_mean, prior_covariance, **kwargs)

    @property
    def mix(self) -> dict:
        """Lazy construction of the exp-link Gaussian mixture approximation."""
        if self._mix is None:
            self._mix = eme.load_or_build_exp_mix(self.nmax_mix, self.cache_dir)
        return self._mix

    def field_from_f(self, f):
        return eme.safe_exp(f)

    def f_from_field(self, field):
        return np.log(np.clip(field, 1e-300, None))

    def derivative_field_from_f(self, f):
        return eme.safe_exp(f)

    def derivative_log_field_from_f(self, f):
        return np.ones_like(np.asarray(f, dtype=float))

    def second_derivative_log_field_from_f(self, f):
        return np.zeros_like(np.asarray(f, dtype=float))

    def second_derivate_field_from_f(self, f):
        return eme.safe_exp(f)
    
    def first_guess_estimator(self):
        field = np.clip(self.nobs + 0.5, 1e-8, None)
        f = self.f_from_field(field)
        v = self.derivative_field_from_f(f)
        s2 = field / (v**2 + 1e-15)
        return super().first_guess_estimator(f, s2)

    def sample_posterior(
        self,
        n_iter: int,
        burn_in: int = 0,
        thin: int = 1,
        initial_f: np.ndarray | None = None,
        random_seed: int | None = None,
    ):
        
        if random_seed is not None:
            np.random.seed(random_seed)

        if not hasattr(self, "nobs"):
            raise ValueError("Call set_data(nobs) before sample_posterior().")

        N = self.nbins

        if initial_f is None:
            try:
                f = self.first_guess_estimator()
            except Exception:
                f = self.prior_mean.copy()
        else:
            f = np.asarray(initial_f, dtype=float).copy()
            if f.shape != (N,):
                raise ValueError("initial_f must have shape (nbins,)")

        total_iter = burn_in + n_iter * thin

        fz_cache = gsm.prepare_f_cond_z(
            self.nobs,
            self.prior_mean,
            self.Lprior,
            self.mix,
        )

        for it in range(total_iter):
            z = gsm.sample_z_cond_f(f, self.nobs, self.mix)
            f = gsm.sample_f_cond_z_cache(z, fz_cache, self.mix)

            if it >= burn_in and ((it - burn_in) % thin == 0):
                yield f
    
class PolyaGammaDensity(SigmoidMixin, Density):

    def __init__(self, prior_mean=None, prior_covariance=None, **kwargs):
        super().__init__(prior_mean, prior_covariance, **kwargs)

class RampDensity(SmoothRampMixin, Density):
    def __init__(self,prior_mean=None, prior_covariance=None,  **kwargs):
        super().__init__(prior_mean, prior_covariance, **kwargs)

class ExponentialDensity(ExponentialMixin, Density):
    def __init__(self,prior_mean=None, prior_covariance=None,  **kwargs):
        super().__init__(prior_mean, prior_covariance, **kwargs)

class Mixin2D:
    
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
    
    def imshow(self, d, **kwargs):
        plt.imshow( self.scanorder_to_image(d, self.n, self.m), **kwargs)


class Density2D(Mixin2D, Density):
    pass


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
    

class ExponentialDensity2D(Mixin2D, ExponentialDensity):
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
