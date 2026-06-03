__doc__="a class to use the Polya-Gamma technique for density estimation"

import numpy as np
import scipy as sp
from scipy.linalg import cho_solve
import scipy.linalg.blas as blas
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as sparse_linalg
from scipy.stats import poisson 
from scipy.special import roots_hermite
import matplotlib.pyplot as plt
import gibbs_softplus_mixture as gsm
from pathlib import Path

import sys

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

import exp_mix_explink as eme
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
from collections.abc import Iterable
from sksparse.cholmod import cholesky


from polyagamma import random_polyagamma

#import syntheticdata as sd
import matplotlib.pyplot as plt


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

def solve_using_chol_factor(L, v):
    """Berechne Sigma0^{-1} @ v unter Verwendung der Cholesky-Zerlegung L von Sigma0."""
    # Rechne L @ y = v
    y = spla.solve_triangular(L, v, lower=True, trans=False)
    # Rechne L.T @ x = y
    x = spla.solve_triangular(L, y, lower=True, trans=True)
    return x

def solve_using_chol_factor_sparse(L, v):
    pass


def _as_cholmod_lower_factor(factor):
    """
    Normalize scikit-sparse/CHOLMOD return variants to (L, permutation).

    Newer local builds can return (L, p) directly. Older scikit-sparse builds
    return a Factor object.
    """
    # Small adapter for different scikit-sparse versions:
    # some return (L, perm) directly, others return a Factor object.
    # In the end we always want the same format: sparse Cholesky L and perm.
    if isinstance(factor, tuple):
        return factor[0].tocsc(), np.asarray(factor[1], dtype=int)

    lower = factor.L().tocsc()
    if hasattr(factor, "P"):
        perm = np.asarray(factor.P(), dtype=int)
    else:
        perm = np.arange(lower.shape[0], dtype=int)
    return lower, perm



def apply_cholesky_sparse(factor, v):
    L, perm = _as_cholmod_lower_factor(factor)
    v = np.asarray(v, dtype=float)

    # CHOLMOD does not factorize A directly, but the permuted matrix:
    # L L.T = P A P.T.
    # perm is "new -> old": perm[j] is the old/original index
    # stored at position j in the new CHOLMOD order.
    #
    # We use C = P.T L as the Cholesky factor in the original variable order.
    # First compute y = L v in CHOLMOD order, then write it back into the
    # old/original order with out[perm].
    y = L @ v
    out = np.empty_like(y, dtype=float)
    # For a vector this is identical to out[perm] = y.
    # For a matrix only the rows are permuted; columns are left untouched.
    out[perm, ...] = y
    return out

def apply_cholesky_sparse_T(factor, v):
    L, perm = _as_cholmod_lower_factor(factor)
    v = np.asarray(v, dtype=float)

    # Transposed factor C.T = L.T P.
    # Therefore first put the input vector into CHOLMOD order:
    # v[perm] is the forward permutation old/original -> new/CHOLMOD.
    return L.T @ v[perm, ...]




class Density:
    """
    Basis class for density computation
    """

    COVARIANCE=1
    PRECISION=0

    def __init__(self, 
                    prior_mean=None, 
                    prior_covariance=None, 
                    prior_precision=None, 
                    sparse=False, 
                    **kwargs):
        """
        
        """

        self.kwargs = kwargs
        self.sparse = sparse

        self.prior_mean=prior_mean
        self.prior_covariance=prior_covariance
        self.prior_precision=prior_precision


        if (prior_covariance is None) and (prior_precision is None):
            return
        
        self.set_prior_Gaussian(
            prior_mean, prior_covariance, prior_precision, sparse)

    def set_prior_Gaussian(self, prior_mean=None, prior_covariance=None, prior_precision=None, sparse=False, **kwargs):

        if not prior_covariance is None:
            self.prior_covariance = np.asarray(prior_covariance, dtype=float)
            n,m = prior_covariance.shape
            self.sparse = False
            self._Lprior = None
            self.prior_precision = None
            self.mode = Density.COVARIANCE    ### covariance Mome
            if not n==m:
                raise Exception("not a quadratic covariance / precision matrix")

        elif not prior_precision is None:
            self.sparse = bool(sparse or sps.issparse(prior_precision))
            if self.sparse:
                self.prior_precision = sps.csc_matrix(prior_precision, dtype=float)
            else:
                self.prior_precision = np.asarray(prior_precision, dtype=float)
            n,m = self.prior_precision.shape
            self.prior_covariance = None
            self._Lprior = None
            self.mode = Density.PRECISION
            if not n==m:
                raise Exception("not a quadratic covariance / precision matrix")
        else:
            raise Exception('at least one vovariance or precision must be speicified')

        self.prior_mean = to_numpy(prior_mean) if not prior_mean is None else np.zeros(n)

        if not self.prior_mean.shape[0] == n:
            raise Exception("dimensions of prior mean and covariance / precision do not match")
        


    def set_data(self, nobs):
        """
        Docstring for set_data
        
        :param self: Description
        :param nobs: array like the observed number of events in the bins
        """
        if not self.prior_mean is None:
            assert len(nobs.ravel()) == self.nbins, "wrong dimension for nobs, must be like prior_mean"
        self.nobs = nobs.ravel()
        self.ndata = sum(self.nobs)

    @staticmethod
    def apply_cholesky_sparse_inverse(factor, v):
        L, perm = _as_cholmod_lower_factor(factor)
        v = np.asarray(v, dtype=float)

        # Solve C x = v with C = P.T L.
        # This becomes L x = P v. In array code, P v is simply v[perm].
        return sparse_linalg.spsolve_triangular(L, v[perm, ...], lower=True)

    @staticmethod
    def apply_cholesky_sparse_inverse_T(factor, v):
        L, perm = _as_cholmod_lower_factor(factor)
        v = np.asarray(v, dtype=float)

        # Solve C.T x = v with C.T = L.T P.
        # First solve L.T y = v. Afterwards y is still in CHOLMOD order.
        y = sparse_linalg.spsolve_triangular(L.T, v, lower=False)
        out = np.empty_like(y, dtype=float)
        # Sort back from CHOLMOD order into the original bin order.
        out[perm, ...] = y
        return out
    
    @property
    def Lprior(self):
        """
        Lazy evaluation of Chlesky factor of prior covariance
        """
        if not self.mode == Density.COVARIANCE:
            raise Exception('you are not in mode COVARIANCE')
        if self._Lprior is None:
            self._Lprior = sp.linalg.cholesky(self.prior_covariance, lower=True)
        return self._Lprior
    
    def get_prior_precision(self):
        """
        lazy method that should be avoided all together if dimension is larg
        """
        if self.mode == Density.PRECISION:
            return self.prior_precision

        if self.mode == Density.COVARIANCE:
            I = np.eye(self.nbins)
            tmp =  sp.linalg.solve_triangular(self.Lprior, I,  trans=False, lower=True)
          
            self.prior_precision = sp.linalg.solve_triangular(self.Lprior, tmp, trans=True, lower=True)
        
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
    
    def prior_n_under_gaussian(self, pm, pv2, n):
        f, w = roots_hermite(20)
        
        Lf = self.field_from_f( pm + np.sqrt(2 * pv2) *f )
        res = 0
        for LLf, ww in zip(Lf, w):
            res += ww * poisson.pmf(n, LLf)
        return res / res.sum()
    
    def posterior_n_single_observation(self, pm, pv2, ncount, n):
        """
        The posterior distribution of number of observed events
        under Gaussian prior (pm, pv2) and a single observation of ncount
        No correlations are taken into account.

        pm: float, prior mean
        pv2: float, prior variance
        ncount: observed number of events
        n: array(dtype=int), the number of predicted events

        returns: array of length n.shape[0] containing the probabilities to observe n events
        """
        
        # using N_HERMITE points for Hermite integration
        N_HERMITE = 20
        f, w = roots_hermite(N_HERMITE)
        
        Lf = self.field_from_f( pm + np.sqrt(2 * pv2) *f)

        res = 0
        tot = 0
        for LLf, ww in zip(Lf, w):
            res += ww * poisson.pmf(n, LLf) * poisson.pmf(ncount, LLf)
            tot += ww * poisson.pmf(ncount, LLf)

        return res / tot

    @property
    def nbins(self):
        return self.prior_mean.shape[0]
    
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
        print('WARNING: depricated, replace by prior single_bin_field')
        tmp = np.exp( - (self.f_from_field(field) - mu)**2/(2*gamma2)) / np.sqrt(2*np.pi*gamma2)
        tmp /= np.abs(self.derivative_field_from_f(self.f_from_field(field)))

        return tmp
    
    def prior_single_bin_f(self, f, mu, gamma2):
        tmp = np.exp( - (f-mu)**2 / (2* gamma2)) / np.sqrt(2*np.pi*gamma2)
        return tmp
    
    def prior_single_bin_field(self, field, mu, gamma2):
        return self.density_under_gaussian(field, mu, gamma2)
    
    def log_likelihood_single_observation_f(self, f, n):
        field = self.field_from_f(f)
        return self.log_likelihood_single_observation_field(field, n)

    def log_likelihood_single_observation_field(self, field, n):
        return n*np.log(field) - field

    def posterior_single_obeservation_field(self, field, mu, gamma2, n):
        tmp = self.log_likelihood_single_obeservation_field(field, mu, gamma2, n)
        tmp -= np.max(tmp)
        tmp = np.exp(tmp)
        tmp /= np.sum(tmp)

        #TODO: compute normalization by Gauss integration instaed

        return tmp
    
    def posterior_single_obeservation_f(self, f, mu, gamma2, n):
        tmp = self.log_likelihood_single_observation_f(f, mu, gamma2, n)
        tmp -= (f-mu)**2/(2*gamma2)
        tmp -= np.max(tmp)
        tmp = np.exp(tmp)
        tmp /= np.sum(tmp)
        #TODO: compute normalization by Gauss integration instaed
        return tmp

    def random_events_from_f(self, f):
        return self.random_events_from_field(self.field_from_f(f))

    def random_prior_parameters(self):
        """
        generates a random sample
        """
        z = np.random.normal(size=self.nbins)
        
        if self.mode == Density.PRECISION:
            if self.sparse:
                factor = cholesky(self.prior_precision, lower=True)
                f = self.apply_cholesky_sparse_inverse_T(factor, z) 
            else:
                chol = np.linalg.cholesky(self.prior_precision)
                f = spla.solve_triangular(chol.T, z, lower=False)
        elif self.mode == Density.COVARIANCE and not self.sparse:
            f = np.dot(self.Lprior, np.random.normal(size=self.nbins))
        
        else:
            raise Exception('not yet implemented')

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
        centered = f - self.prior_mean
        if self.mode == Density.PRECISION:
            prior_quad = sum(centered * (self.prior_precision @ centered))
        
        elif self.mode == Density.COVARIANCE:
            prior_quad = np.sum(
                sp.linalg.solve_triangular(
                    self.Lprior, centered, trans=False, lower=True)**2)
        else:
            raise Exception('not implemented yet')

        return -self.loglikelihood(f) + prior_quad / 2
    
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

 
    def neg_grad_logposterior(self, f):
        """
        """
        res = self.nobs * self.derivative_log_field_from_f(f) - self.derivative_field_from_f(f)

        if self.mode == Density.COVARIANCE:

            tmp = sp.linalg.solve_triangular(self.Lprior, f-self.prior_mean, trans=False, lower=True)
            tmp = sp.linalg.solve_triangular(self.Lprior, tmp, trans=True, lower=True)
            
        elif self.mode == Density.PRECISION:
            tmp = self.prior_precision @ (f-self.prior_mean)
        
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
        guess for f. Take this as a noise measurement of f and then do the Bayesian
        inversion using the prior mean and covariance for f.

        $f^ = \mu + \Sigma( \Sigma + D)^{-1} (f -\mu)$

        or with Q = \Sigma^{-1} ( use the Woodbury identity...)

        f^ = f - (Q + D^{-1})^{-1} Q (f - \mu)

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

        tmp = f - self.prior_mean
        
        if self.mode == Density.COVARIANCE and self.sparse:
            raise Exception('not implemented yet')
        
        elif self.mode == Density.COVARIANCE and not self.sparse:
            dinv = 1.0 / np.asarray(s2, dtype=float)
            tmp = self.Lprior.T @ (dinv * f) + self.Lprior @ self.prior_mean

            DiL = dinv[:,None] * self.Lprior

            tmp = np.linalg.solve(np.eye(self.nbins) + self.Lprior.T @ DiL, tmp)
            tmp = self.Lprior @ tmp

            return tmp
        
        elif self.mode == Density.PRECISION and not self.sparse:
            dinv = 1.0 / np.asarray(s2, dtype=float)
            tmp = self.prior_precision @ self.prior_mean + dinv * f
            tmp = np.linalg.solve(self.prior_precision + np.diag(dinv), tmp )
            return tmp
        
        elif self.mode == Density.PRECISION and self.sparse:
            dinv = 1.0 / np.asarray(s2, dtype=float)
            tmp = self.prior_precision @ self.prior_mean + dinv * f
            system = self.prior_precision + sps.diags(dinv, format="csc")
            tmp = sparse_linalg.spsolve(system, tmp)

            return tmp
        
        else:
            raise Exception('not implemented yet')
    
    
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
        mu0 = self.prior_mean

        # if self.mode == Density.COVARIANCE:
        #     # Avoid forming Sigma0^{-1} explicitly.  We only need products
        #     # Sigma0^{-1} @ v, which can be computed through the Cholesky
        #     # factor L of Sigma0 by solving L y = v and L.T x = y.
        #     #L = np.asarray(self.Lprior, dtype=float)
        #     #apply_Sigma0_inv = lambda v: sigma0_inv_dot(v, L)
        #     #tmp = apply_Sigma0_inv(mu0)
        #     Sigma0_inv_mu0 = self.apply_prior_inverse_covar(mu0)
        # elif self.mode == Density.PRECISION and self.sparse:
        #     #apply_Sigma0_inv = lambda v: self.prior_precision @ v
        #     #tmp = apply_Sigma0_inv(mu0)

        #Sigma0_inv_mu0 = self.apply_prior_precision(mu0)


            

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


        if self.mode == Density.COVARIANCE and not self.sparse:
            print('case COVARIANCE')
            Sigma0_inv_mu0 = spla.solve_triangular(self.Lprior, mu0, lower=True, trans=False)
            Sigma0_inv_mu0 = spla.solve_triangular(self.Lprior, Sigma0_inv_mu0, lower=True, trans=True)

        elif self.mode == Density.PRECISION and not self.sparse:
            print('case PRECISION not sparse')
            Sigma0_inv_mu0 = self.prior_precision @ mu0

        elif self.mode == Density.PRECISION and self.sparse:
            print('case PRECISION sparse')
            Sigma0_inv_mu0 = self.prior_precision @ mu0

        else:
            raise Exception('not implemented yet')
        

        
        # Gibbs loop
        for it in range(n_iter):
            print('iteration ', it)
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


            # same for all cases
            b = Sigma0_inv_mu0 + kappa
            
            
            if self.mode == Density.COVARIANCE and not self.sparse:

                
                weighted_L = w[:, None] * self.Lprior
                M = np.eye(nbins) + self.Lprior.T @ weighted_L

                chol = sp.linalg.cholesky(M, lower=True)

                rhs = self.Lprior.T @ b  

                rhs = spla.solve_triangular(chol, rhs, lower=True, trans=False)
                rhs = spla.solve_triangular(chol, rhs, lower=True, trans=True)
                m = self.Lprior @ rhs

                print('m', m)

                # sampling of new f 
                z = np.random.normal(size=nbins)

                print('z', z[:5])
                eps = spla.solve_triangular(chol, z, lower=True, trans=True)
                eps = self.Lprior @ eps


                f = m + eps

            elif self.mode == Density.PRECISION and self.sparse:
                A = (self.prior_precision + sps.diags(w, format="csc")).tocsc()

                factor = cholesky(A, lower=True)
                tmp = self.apply_cholesky_sparse_inverse(factor, b)
                m = self.apply_cholesky_sparse_inverse_T(factor, tmp)

                z = np.random.normal(size=nbins)
                eps = self.apply_cholesky_sparse_inverse_T(factor, z)

                f = m + eps


            elif self.mode == Density.PRECISION and not self.sparse:   
                A = self.prior_precision + np.diag(w)

                chol = sp.linalg.cholesky(A, lower=True)

                rhs = spla.solve_triangular(chol, b, lower=True, trans=False)
                m = spla.solve_triangular(chol, rhs, lower=True, trans=True)


                z = np.random.normal(size=nbins)
                eps = spla.solve_triangular(chol, z, lower=True, trans=True)

                f = m + eps
            else:
                raise Exception('not implemented yet')

        # Speichern, wenn burn_in abgeschlossen ist und bei Ausdünnungsintervall
            if it >= burn_in and ((it - burn_in) % thin == 0):
                self.last_sample = f
                yield f
        return


class SmoothRampMixin:

    def __init__(
        self,
        prior_mean=None,
        prior_covariance=None,
        prior_precision=None,
        sparse=False,
        nmax_mix:int=60,
        cache_dir:Path=Path('.mixture'),
        softplus_k: float = 1.0,
        **kwargs,
    ) -> dict:
        self._mix = None
        self.nmax_mix = nmax_mix
        self.cache_dir = cache_dir 
        self.softplus_k = float(softplus_k)
        super().__init__(
            prior_mean,
            prior_covariance,
            prior_precision=prior_precision,
            sparse=sparse,
            **kwargs,
        )

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
        fz_cache = gsm.prepare_f_cond_z(self)
        


        idx = 0
        for it in range(total_iter):
            print('samplilng z | f', it)
            z = gsm.sample_z_cond_f(f, self.nobs, self.mix)
            #f = gsm.sample_f_cond_z(z, self.nobs, self.prior_mean, self.Lprior, self.mix)
            print('samplilng f | z', it)
            f = gsm.sample_f_cond_z_cache(z, self, fz_cache)
            if it >= burn_in and ((it - burn_in) % thin == 0):
                #f_samples[idx] = f
                idx += 1
                yield f

        return
 
class ExponentialMixin:

    ### TODO:

    def __init__(
        self,
        prior_mean=None,
        prior_covariance=None,
        prior_precision=None,
        sparse=False,
        nmax_mix: int = 60,
        cache_dir: Path = Path(".mixture"),
        **kwargs,
    ):
        self._mix = None
        self.nmax_mix = int(nmax_mix)
        self.cache_dir = Path(cache_dir)
        super().__init__(
            prior_mean,
            prior_covariance,
            prior_precision=prior_precision,
            sparse=sparse,
            **kwargs,
        )

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

        fz_cache = gsm.prepare_f_cond_z(self)

        for it in range(total_iter):
            z = gsm.sample_z_cond_f(f, self.nobs, self.mix)
            f = gsm.sample_f_cond_z_cache(z, self, fz_cache)

            if it >= burn_in and ((it - burn_in) % thin == 0):
                yield f
    
class PolyaGammaDensity(SigmoidMixin, Density):

    def __init__(self, prior_mean=None, prior_covariance=None, prior_precision=None, sparse=False, **kwargs):
        super().__init__(
            prior_mean,
            prior_covariance,
            prior_precision=prior_precision,
            sparse=sparse,
            **kwargs,
        )

class RampDensity(SmoothRampMixin, Density):
    def __init__(self, prior_mean=None, prior_covariance=None, prior_precision=None, sparse=False, **kwargs):
        super().__init__(
            prior_mean,
            prior_covariance,
            prior_precision=prior_precision,
            sparse=sparse,
            **kwargs,
        )

class ExponentialDensity(ExponentialMixin, Density):
    def __init__(self, prior_mean=None, prior_covariance=None, prior_precision=None, sparse=False, **kwargs):
        super().__init__(
            prior_mean,
            prior_covariance,
            prior_precision=prior_precision,
            sparse=sparse,
            **kwargs,
        )

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
    def __init__(self, prior_mean=None, prior_covariance=None, prior_precision=None, sparse=False, **kwargs):
        super().__init__(
            prior_mean,
            prior_covariance,
            prior_precision=prior_precision,
            sparse=sparse,
            **kwargs,
        )

   
class RampDensity2D(Mixin2D, RampDensity):
    def __init__(self, prior_mean=None, prior_covariance=None, prior_precision=None, sparse=False, **kwargs):
        super().__init__(
            prior_mean,
            prior_covariance,
            prior_precision=prior_precision,
            sparse=sparse,
            **kwargs,
        )
    

class ExponentialDensity2D(Mixin2D, ExponentialDensity):
    def __init__(self, prior_mean=None, prior_covariance=None, prior_precision=None, sparse=False, **kwargs):
        super().__init__(
            prior_mean,
            prior_covariance,
            prior_precision=prior_precision,
            sparse=sparse,
            **kwargs,
        )


    

if __name__ == '__main__':
    pass
