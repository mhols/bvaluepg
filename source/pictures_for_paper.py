import polyagammadensity as pg
import syntheticdata as sd
import matplotlib.pyplot as plt
import covariance_kernels as ck
import numpy as np
import scipy.sparse.linalg as sparse_linalg

class PicturesForPaper:

    def __init__(self, **kwargs):
        #prepare all data
        self.kwargs = kwargs
        self.n = 64 # size of square
        self.m = 64
        self.nn = 32
        self.a_high=0.5
        self.b_high=5
        self.a_low=0.5
        self.b_low=1
        self.lam = 15

        ## covariance settings
        self.prior_covariance_class = ck.spatial_covariance_gaussian
        self.r = 3    # correlation length
        self.prior_mean_PG_high_value = 0
        self.prior_mean_PG_low_value = -2
        self.prior_mean_EXP_low_value = 0.5
        self.prior_mean_EXP_high_value = 2
        
        self.v2_PG_low = 4
        self.v2_PG_high = 9
        self.v2_EXP_low = 4
        self.v2_EXP_high = 9

        self.prior_mean_PG_low = self.prior_mean_PG_low_value * np.ones(self.n * self.m)
        self.prior_mean_PG_high = self.prior_mean_PG_high_value * np.ones(self.n * self.m)


        self.truth_high_contrast_square = sd.single_square(n=self.n, nn=self.nn, a=self.a_high, b=self.b_high)
        self.truth_low_contrast_square = sd.single_square(n=self.n, nn=self.nn, a=self.a_low, b=self.b_low)
        self.sample_data_high = pg.Density.random_events_from_field( self.truth_high_contrast_square)
        self.sample_data_low = pg.Density.random_events_from_field( self.truth_low_contrast_square)


        self.PG_low  = pg.PolyaGammaDensity2D(n=self.n, m=self.m, lam=self.lam)
        self.PG_high = pg.PolyaGammaDensity2D(n=self.n, m=self.m, lam=self.lam)
        self.RD_low = pg.RampDensity2D(n=self.n, m=self.m)
        self.RD_high= pg.RampDensity2D(n=self.n, m=self.m)
        self.EXP_low = pg.ExponentialDensity(n=self.n, m=self.m)  ## for the moment no sampling et
        self.EXP_high= pg.ExponentialDensity(n=self.n, m=self.m)  ## for the moment no sampling et


 
        covar_low = self.prior_covariance_class(self.n, self.m, self.r, self.v2_PG_low)
        self.PG_low.set_prior_Gaussian(self.prior_mean_PG_low, covar_low)

        covar_high= self.prior_covariance_class(self.n, self.m, self.r, self.v2_PG_high)
        self.PG_high.set_prior_Gaussian(self.prior_mean_PG_high, covar_high)

       

    @property
    def maximum_logposterior_PG_high(self):
        if not hasattr(self, '_maximum_logposterior_PG_high'):
            self.PG_high.set_data(self.sample_data_high)
            self._maximum_logposterior_PG_high = self.PG_high.max_logposterior_estimator()
        return self._maximum_logposterior_PG_high

    @property
    def maximum_logposterior_PG_low(self):
        if not hasattr(self, '_maximum_logposterior_PG_low'):
            self.PG_low.set_data(self.sample_data_low)
            self._maximum_logposterior_PG_low= self.PG_low.max_logposterior_estimator()
        return self._maximum_logposterior_PG_low


    @property
    def posterior_samples_PG_high(self):
        if not hasattr(self, '_posterior_samples_PG_high'):
            self._posterior_samples_PG_high = list( self.PG_high.sample_posterior(n_iter=150))

        return self._posterior_samples_PG_high
    
    @property
    def posterior_samples_PG_low(self):
        if not hasattr(self, '_posterior_samples_PG_low'):
            self._posterior_samples_PG_low= list( self.PG_low.sample_posterior(n_iter=150))

        return self._posterior_samples_PG_low


    def figure_01(self):
        # upper square example high-contrast

        plt.figure()
        plt.imshow(self.truth_high_contrast_square, vmin=0, vmax=6)


    def figure_02(self):
        # upper square example low-contrast


        plt.figure()
        plt.imshow(self.truth_low_contrast_square, vmin=0, vmax=6)

   
    def figure_03(self):
        # upper square example high-contrast

        plt.figure()
        plt.imshow(self.sample_data_high, vmin=0, vmax=6)

    def figure_04(self):
        # upper square example low-contrast

        plt.figure()
        plt.imshow(self.sample_data_low, vmin=0, vmax=6)


    def figure_05(self):
        #maximum posterior estimate

        plt.figure()
        self.PG_high.imshow(self.PG_high.field_from_f(self.maximum_logposterior_PG_high), vmin=0, vmax=6)

    def figure_06(self):
        #maximum posterior estimate

        plt.figure()
        self.PG_low.imshow(self.PG_low.field_from_f(self.maximum_logposterior_PG_low), vmin=0, vmax=6)


    def figure_07(self):
        ## posterior samplinghj

        plt.figure()
        self.PG_high.imshow(
            sum( [ self.PG_high.field_from_f(f) for f in self.posterior_samples_PG_high] ) / len(self.posterior_samples_PG_high), vmin=0, vmax=6)

    def figure_08(self):
        ## posterior samplinghj

        plt.figure()
        self.PG_low.imshow(
            sum( [ self.PG_low.field_from_f(f) for f in self.posterior_samples_PG_low] ) / len(self.posterior_samples_PG_low), vmin=0, vmax=6)


    def figure_09(self):
        lams = np.linspace(0.01, self.lam*0.9, 10000)
        plt.figure()
        plt.plot(lams, self.PG_high.density_under_gaussian(lams, self.prior_mean_PG_high_value, self.v2_PG_high))

    def figure_10(self):
        lams = np.linspace(0.01, self.lam*0.9, 10000)
        plt.figure()
        plt.plot(lams, self.PG_low.density_under_gaussian(lams, self.prior_mean_PG_low_value, self.v2_PG_low))

    def figure_11(self):
        lams = np.linspace(0.01, self.lam*0.9, 10000)
        plt.figure()
        plt.plot(lams, self.EXP_low.density_under_gaussian(lams, self.prior_mean_EXP_low_value, self.v2_EXP_low))

    def figure_12(self, title):
        lams = np.linspace(0.01, self.lam*0.9, 10000)
        plt.figure()
        plt.title(title)
        plt.plot(lams, self.PG_low.posterior_field_one_dimension(lams, self.prior_mean_PG_low_value, self.v2_PG_low, 1))

    def figure_13(self, title):
        """
        
        """
        f = np.linspace(-3, 10,  10000)

        plt.figure()
        plt.title(title)
        lam = self.EXP_low.field_from_f(f)
        pm = self.prior_mean_EXP_low_value
        n = int(self.EXP_low.field_from_f(pm)/2)
        pv2 = self.v2_EXP_low
        plm, vl2 = self.EXP_low.laplace_approximation_one_dimension(pm, pv2, n)

        posterior = n*np.log(lam) - lam - (f-pm)**2/(2*pv2)
        posterior -= posterior.max()
        posterior = np.exp(posterior)
        plt.plot(f, posterior)
        plt.plot(f, np.exp(-(f-plm)**2/(2*vl2)))

    def figure_14(self, title):
        """
        
        """
        f = np.linspace(-3, 10,  10000)

        plt.figure()
        plt.title(title)
        lam = self.EXP_high.field_from_f(f)
        pm = self.prior_mean_EXP_high_value
        n = 0 #int(self.EXP_high.field_from_f(pm) / 2)
        pv2 = self.v2_EXP_high*4
        plm, vl2 = self.EXP_high.laplace_approximation_one_dimension(pm, pv2, n)

        posterior = n*np.log(lam) - lam - (f-pm)**2/(2*pv2)
        posterior -= posterior.max()
        posterior = np.exp(posterior)
        plt.plot(f, posterior)
        plt.plot(f, np.exp(-(f-plm)**2/(2*vl2)))

    def figure_15(self, title, pm, n, pv2, CALC):
        """
        
        """
        CALC = self.EXP_high

        f = np.linspace(-7, 10,  10000)
        lam = np.linspace(0, 15, 10000)[1:]

        ## toy parameters
        pm = 0.5
        n = 1 
        pv2 = 4 

        
        plm, vl2 = CALC.laplace_approximation_one_dimension(pm, pv2, n)

        plt.figure()
        plt.title(title)
        plt.plot()
        lam = self.EXP_high.field_from_f(f)
        posterior = n*np.log(lam) - lam - (f-pm)**2/(2*pv2)
        posterior -= posterior.max()
        posterior = np.exp(posterior)
        
        plt.plot(f, posterior)
        plt.plot(f, np.exp(-(f-plm)**2/(2*vl2)))

        plt.figure()
        llam = np.linspace(0, 15, 10000)
        postlam = self.EXP_high.density_under_gaussian(llam, pm, pv2) \
            * llam**n * np.exp(-llam)
        postlam /= postlam.max()
        print(np.sum(postlam*llam)/np.sum(postlam))
        plt.plot(llam, postlam)
        postlaplace = self.EXP_high.density_under_gaussian(llam, plm, vl2)
        postlaplace /= postlaplace.max()

        print(np.sum(postlaplace*llam)/np.sum(postlaplace))
        plt.plot(llam, postlaplace)

        plt.figure()
        plt.plot(llam, self.EXP_high.density_under_gaussian(llam, pm, pv2))

        plt.figure()
        plt.plot(f, lam**n * np.exp(-lam))

    def figure_16(self, title, pm, pv2, n, CALC=None):
        if CALC is None:
            CALC = self.PG_high 

        plt.figure()
        plt.title(title)

        plt.plot( n, CALC.prior_n_under_gaussian(pm, pv2, n), 'o')


    def figure_17(self, title, pm, pv2, ncount, n, CALC=None):
        if CALC is None:
            CALC = self.PG_high 

        plt.figure()
        plt.title(title)

        plt.plot( n, CALC.posterior_n_single_observation(pm, pv2, ncount, n), 'o')

    def figure_18(self, title):
        """
        Sparse Matern Covariance
        """

        n = 256
        L = ck.precision_matern_9pt(n, 0.01, 1)
        Q = L.T @ L

        e = np.zeros(Q.shape[0])
        e[ n*n//2+n//2] = 1

        kernel = sparse_linalg.spsolve(Q, e)
        #kernel = Q @ e

        plt.figure()
        plt.imshow(pg.Mixin2D().scanorder_to_image(kernel, n, n))





if __name__ == '__main__':
    P = PicturesForPaper()
    # P.figure_01()
    # P.figure_02()
    # P.figure_03()
    # P.figure_04()
    # P.figure_05()
    # P.figure_06()
    # P.figure_07()
    # P.figure_08()
    # P.figure_09()
    # P.figure_10()
    # P.figure_11()
    # P.figure_12('posterior density for 1-observation')
    # P.figure_13('posterior f for 1-observation')
    # P.figure_14('posterior f for 1-observation')
    # P.figure_15('posterior f for 1-observation', pm=1, n=1, pv2=1, CALC=pg.ExponentialDensity)
    #P.figure_16('prior distribtuion events exponential low', pm=1, pv2=4, n=np.arange(20), CALC=P.EXP_low)
    #P.figure_16('prior distribtuion events exponential high', pm=1, pv2=4, n=np.arange(20), CALC=P.EXP_high)
    
    #P.figure_17('posterior single observation sigmoid', 
    #            pm=1, pv2=4, ncount=0, n=np.arange(20), CALC=P.PG_low)
    #P.figure_17('posterior single observation exponential low', 
    #            pm=1, pv2=4, ncount=0, n=np.arange(20), CALC=P.EXP_low)
    P.figure_18('Matern precision')
    plt.show()