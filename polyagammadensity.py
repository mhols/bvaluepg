__doc__="a class to use the Polya-Gamma technique for density estimation"

import numpy as np
#import polyagamma    ### TODO

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
    
    def field_from_f(self, f):
        return self.lam * sigmoid(f)

    def random_events_from_field(self, field):
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
        return self.lam * sigmoid(self.random_prior_prameters())
    
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
        field = self.lam * sigmoid(f)

        return np.sum(self.nobs * np.log(field)) - np.sum(field)
    
    def logposterior(self, f):
        """
        Docstring for logposterior
        
        :param self: Description
        :param f: Description
        """
        return self.loglikelihood(f) - np.sum( (np.solve(self.Lprior, f)**2 / 2))
    
    def maxposterior_estimate(self):
        pass      ### TODO implement maximum posterior estimate

    def sample_polyagamma_cond_f(self):

        field = self.field_from_f(-self.f)
        kk = self.random_events_from_field(field)  ### the random events k given f

        # self.polya = [ polyagamma( n + k , f) for n, k, f in zip(self.nobs, kk, self.f)]

        return 

    def sample_f_cond_polyagamma(self):
        pass



    

if __name__ == '__main__':

    import syntheticdata as sd
    import matplotlib.pyplot as plt

    n, m = 50, 50

    pgd = PolyaGammaDensity(
        prior_mean=np.zeros( n * m ),
        prior_covariance=sd.spatial_covariance_gaussian(n, m, 3, 1),
        lam=10
    )

    plt.figure()
    plt.imshow( sd.scanorder_to_image(pgd.random_prior_events(), n, m).T)
    plt.show()
    

