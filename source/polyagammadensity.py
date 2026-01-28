__doc__="a class to use the Polya-Gamma technique for density estimation"

import numpy as np
import scipy as sp
import syntheticdata as sd
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
        if self._Lprior is None:
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
        return -self.loglikelihood(f) + np.sum( sp.linalg.solve_triangular(self.Lprior, f-self.prior_mean, lower=True)**2) / 2
    
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
       
        sigf = sigmoid(f)
        sigm = sigmoid(-f)

        res = self.nobs * sigm - self.lam * sigm * sigf
        tmp = sp.linalg.solve_triangular(self.Lprior, f, lower=False) ### besser die schon berechneten Choleski Faktoren benutzen
        tmp = sp.linalg.solve_triangular(self.Lprior.T, tmp, lower=True)

        return -res + tmp
    
    def max_logposterior_estimator(self, f0=None, niter=10, eps=1e-6):
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


        s = (self.nobs-np.sqrt(self.nobs)) / self.lam
        s = np.where(s>=1, 0.9, s)
        s = np.where(s<=0, 0.1, s)

        if f0 is None:
            f0 = np.log(s / (1-s)) #### TODO use something more reasonable


        res = sp.optimize.minimize(
                self.neg_logposterior, 
                f0, 
                jac=self.neg_grad_logposterior, 
                method='Powell', 
                #bounds = [ (-5, 5) for i in range(self.nbins) ],
                options={'maxiter':niter }) ##computing the minimization using conjugated gradients
         
        return res['x']
        #return np.asarray(grad, dtype=float).reshape(-1)  ## what is this ?
        
    def sample_polyagamma_cond_f(self):

        field = self.field_from_f(-self.f)
        kk = self.random_events_from_field(field)  ### the random events k given f

        # self.polya = [ polyagamma( n + k , f) for n, k, f in zip(self.nobs, kk, self.f)]

        return 
 

    def sample_f_cond_polyagamma(self):
        pass



class PolyGammaDensity2D(PolyaGammaDensity):
    """
    Docstring for PolyGammaDensity2D
    specialized in 2Dimensional data
    """
    def __init__(self, prior_mean, prior_covariance, n, m, lam, **kwargs):
        super(prior_mean, prior_covariance, lam, **kwargs)
        self.n, self.m = n, m

    def to_image(self, d):
        sd.scanorder_to_image(d, self.n, self.m)
    
    def imshow(self, d):
        plt.imshow( sd.scanorder_to_image(d, self.n, self.m).T)

    

    

    


    

if __name__ == '__main__':

    import syntheticdata as sd
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt


    n, m = 20, 20

    pgd = PolyaGammaDensity(
        prior_mean=np.zeros( n * m ),
        prior_covariance=sd.spatial_covariance_gaussian(n, m, 3, 1),
        lam=10
    )

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

    ##grad = pgd.neg_grad_logposterior(prior)

    #plt.figure()
    #plt.title("Gradient log-posterior von prior sample")
    #plt.imshow( sd.scanorder_to_image( grad, n, m ).T)   ### Gradient ist ein Vektor-feld... imshow????

    
    #%%
    '''compare with scipy approx
    vorher logposterior anpassen
    '''

    #%%


    res = pgd.max_logposterior_estimator()

    for i in range(10):
        plt.figure()
        plt.title(f"{i}-th max_posterior estimate of field")
        plt.imshow( sd.scanorder_to_image( pgd.field_from_f(res), n, m ).T)

        res = pgd.max_logposterior_estimator(res, 50)



    plt.figure()
    plt.title('difference params')
    plt.imshow( sd.scanorder_to_image( res - prior, n, m ).T)


    plt.figure()
    plt.title('gradient')
    plt.imshow( sd.scanorder_to_image( pgd.neg_grad_logposterior(prior), n, m ).T)

    plt.show()
