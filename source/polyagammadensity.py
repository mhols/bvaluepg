__doc__="a class to use the Polya-Gamma technique for density estimation"

import numpy as np
import scipy as sp
#import polyagamma    ### TODO

def sigmoid(f):
    return 1/(1+np.exp(-f))   #TODO what if f gets big ????

def der_sigmoid(f):
    return sigmoid(f)*sigmoid(-f)

class PolyaGammaDensity:

    def __init__(self, prior_mean, prior_covariance, lam, **kwargs):

        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.lam = lam

        self.kwargs = kwargs

        self._Lprior = None # for lazy evaluation


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
        if not self._Lprior:
            self._Lprior = np.linalg.cholesky(self.prior_covariance)
        return self._Lprior
    
    @property
    def nbins(self):
        return self.prior_mean.shape[0]
    
    def field_from_f(self, f):
        return self.lam * sigmoid(f)

    def random_events_from_field(self, field):
        """
        samples events from probability field
        
        :param self: Description
        :param field: Description
        """

        assert np.all( field >= 0 )

        return np.array( [np.random.poisson(l) for l in field] )
                    
    def random_events_from_f(self, f):
        return self.random_events_from_field(self.field_from_f(f))

    def random_prior_prameters(self):
        """
        generates a random sample
        """
        f = np.dot(self.Lprior, np.random.normal(size=self.nbins))

        return self.prior_mean + f
    
    def random_prior_field(self):
        """
        a random realization of the underlying poissonian density in each bin
        """
        return self.field_from_f(self.random_prior_prameters())
    
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
        field = self.field_from_f(f)

        return np.sum(self.nobs * np.log(field)) - np.sum(field)
    
    def logposterior(self, f):
        """
        Docstring for logposterior
        
        :param self: Description
        :param f: Description
        """
        return self.loglikelihood(f) - np.sum( (np.linalg.solve(self.Lprior, f)**2 / 2))
    
    def grad_logposterior(self, f):
        res = self.nobs * sigmoid(-f)
        res -= 0 ## TODO
        return res

    def maxposterior_estimate(self):
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
        field = self.field_from_f(f)

        grad_loglikelihood = self.lam * (self.nobs / field - 1) * sigmoid(f) * (1 - sigmoid(f))

        grad_logprior = - np.linalg.solve(self.prior_covariance, f) ### besser die schon berechneten Choleski Faktoren benutzen

        return grad_loglikelihood + grad_logprior
    
    def grad_scipy_logposterior(self, f, eps=1e-6):
        """
        Docstring for grad_scipy_logposterior
        
        :param self: Description
        :param f: Description
        """
        f = np.asarray(f, dtype=float)
        assert f.shape[0] == self.nbins, "falsche dimension for f"

        # approx_fprime in scipy erwartet scalar function
        def _fun(x):
            x = np.asarray(x, dtype=float)
            return float(self.logposterior(x))
    
        # Verwendet Schrittweiten pro Koordinate (Skalar eps, der an einen Vektor übertragen wird).
        epsilon = np.full_like(f, float(eps), dtype=float)
        grad = sp.optimize.approx_fprime(f, _fun, epsilon)
        return np.asarray(grad, dtype=float).reshape(-1)
        
    def sample_polyagamma_cond_f(self):

        field = self.field_from_f(-self.f)
        kk = self.random_events_from_field(field)  ### the random events k given f

        # self.polya = [ polyagamma( n + k , f) for n, k, f in zip(self.nobs, kk, self.f)]

        return 

    def maxposterior_estimate(self):
        pass      ### TODO implement maximum posterior estimate
        # """
        # finds the maximum posterior estimate using optimization
        # """
        # result = minimize(
        #     lambda f: -self.logposterior(f),
        #     x0=self.prior_mean,
        #     method='BFGS'
        #     )
        # return result.x
 

    def sample_f_cond_polyagamma(self):
        pass



    

if __name__ == '__main__':

    import syntheticdata as sd
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    n, m = 30, 30

    pgd = PolyaGammaDensity(
        prior_mean=np.zeros( n * m ),
        prior_covariance=sd.spatial_covariance_gaussian(n, m, 3, 1),
        lam=10
    )

    prior = pgd.random_prior_prameters()

    plt.figure()
    plt.imshow( sd.scanorder_to_image( prior, n, m ).T)


    prior_field = pgd.field_from_f(prior)

    plt.figure()
    plt.imshow( sd.scanorder_to_image(prior_field, n, m).T)


    prior_events = pgd.random_events_from_field(prior_field)
    plt.figure()
    plt.imshow( sd.scanorder_to_image(prior_events, n, m).T)

    pgd.set_data(prior_events)

    grad = pgd.grad_logposterior(prior)

    plt.figure()
    plt.title("Gradient log-posterior von prior sample")
    plt.imshow( sd.scanorder_to_image( grad, n, m ).T)

    
    #%%
    '''compare with scipy approx
    vorher logposterior anpassen
    '''

    #%%




    # grad = pgd.grad_scipy_logposterior(prior)

    # plt.figure()
    # plt.title("Gradient_scipy1 log-posterior von prior sample")
    # plt.imshow( sd.scanorder_to_image( grad, n, m ).T)

    plt.show()
    

