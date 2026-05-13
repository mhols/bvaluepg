import polyagammadensity as pg
import syntheticdata as sd
import matplotlib.pyplot as plt
import covariance_kernels as ck
import numpy as np


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
        
        self.v2_PG_low = 1
        self.v2_PG_high = 2
        self.v2_EXP_low = 2

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
        self.EXP_low = pg.ExponentialDensity()  ## for the moment no sampling et

 
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
        f = np.linspace(-3, 2,  10000)

        plt.figure()
        plt.title(title)
        lam = self.RD_low.field_from_f(f)
        pm = self.prior_mean_PG_low_value
        pv2 = self.v2_PG_low
        pm, v2 = self.RD_low.laplace_approximation_one_dimension()
        plt.plot(f, np.exp(-(f-self.prior_mean_PG_low_value)**2/(2*self.v2_EXP_low**2))*lam*np.exp(-lam))

if __name__ == '__main__':
    P = PicturesForPaper()
    P.figure_01()
    P.figure_02()
    P.figure_03()
    P.figure_04()
    P.figure_05()
    P.figure_06()
    P.figure_07()
    P.figure_08()
    P.figure_09()
    P.figure_10()
    P.figure_11()
    P.figure_12('posterior density for 1-observation')
    P.figure_13('posterior f for 1-observation')
    plt.show()