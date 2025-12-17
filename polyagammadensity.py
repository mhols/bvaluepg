__doc__="a class to use the Polya-Gamma technique for density estimation"

import numpy as np

def sigmoid(f):
    return 1/(1+np.exp(-f))   #TODO what if f gets big ????


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
        return self.lam * sigmoid(self.random_prior_prameters())
    
    def random_prior(self):
        """
        samples counting data from the prior distribution
        """
        pf = self.random_prior_field()

        return [
            np.random.poisson(lam=pf[i]) for i in range(self.nbins)
        ]
    

if __name__ == '__main__':

    import syntheticdata as sd
    import matplotlib.pyplot as plt

    n, m = 30, 30

    pgd = PolyaGammaDensity(
        np.zeros( n * m ),
        sd.spatial_covariance_gaussian(n, m, 4, 1),
        10
    )

    plt.figure()
    plt.imshow( sd.scanorder_to_image(pgd.random_prior(), n, m))
    plt.show()
    

