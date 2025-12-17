__doc__="a class to use the Polya-Gamma technique for density estimation"

import numpy as np

def sigmoid(f):
    return 1/(1+np.exp(-f))   #TODO what if f gets big ????


class PolyaGammaDensity:

    def __init__(self, prior_mean, prior_covariance, **kwargs):

        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance

        self.kwargs = kwargs

        self._Lprior = None # for lazy evaluation

    @property
    def Lprior(self):
        if not self._Lprior:
            self._Lprior = np.linalg.cholesky(self.prior_covariance)
        return self._Lprior
    
    @property
    def nbins(self):
        return self.prior_mean.shape[0]
        
            

    def random_prior_prameters(self):
        """
        generates a random sample
        """
        f = np.solve(self.Lprior, np.random.normal(size=self.nbins))

        return self.prior_mean + f
    
    def random_prior(self):
        f = self.random_prior_prameters()

        return sigmoid(f)
