import pandas as pd
import numpy as np
import copy
import sys
from pathlib import Path
from . import declustering as dc
from .. import covariance_kernels as ck
import scipy
import matplotlib.pyplot as plt
import pickle
import zipfile
import geopandas as gpd
from shapely.geometry import box
from .. import polyagammadensity as pgd
from . import coordinates as coord
from scipy.stats import truncexpon
import os
from functools import cached_property

def bin_indices(x, y, xedges, yedges):
    nx, ny = len(xedges) - 1, len(yedges) - 1

    # bin index along each axis: 0 .. n-1
    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1

    # put points exactly on the last edge into the last bin
    ix[x == xedges[-1]] = nx - 1
    iy[y == yedges[-1]] = ny - 1

    # keep only points inside the grid
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    idx_ok = np.flatnonzero(ok)

    flat = iy[ok] * nx + ix[ok]                 # row-major: x is the row, y the column

    order = np.argsort(flat, kind="stable")     # stable keeps the original order within a bin
    sorted_idx = idx_ok[order]
    counts = np.bincount(flat, minlength=nx * ny)
    splits = np.cumsum(counts)[:-1]
    return np.split(sorted_idx, splits)

def bin_indices_1d(x, edges):
    n = len(edges) - 1
    ix = np.digitize(x, edges) - 1
    ix[x == edges[-1]] = n - 1                 # last edge belongs to the last bin (like np.histogram)

    ok = (ix >= 0) & (ix < n)
    idx_ok = np.flatnonzero(ok)

    order = np.argsort(ix[ok], kind="stable")
    counts = np.bincount(ix[ok], minlength=n)
    return np.split(idx_ok[order], np.cumsum(counts)[:-1])

class CatalogMixin(pgd.Mixin2D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._coordinates = None

        if 'catalog' in kwargs:
            self.load_catalog()

        else:
            with open(kwargs['catpickle'], 'rb') as f:
                self._catDataFrame = pickle.load(f)

    def load_catalog(self, **kwargs):

        catalog = self.kwargs.get('catalog')

        path, kwargs, trans = catalog
        cata = pd.read_csv(path, **kwargs)

        self.original_catalog = cata

        self._catDataFrame = pd.DataFrame()

        if 'type_to_keep' in trans  and not trans['type_to_keep'] is None:
            k, v = trans['type_to_keep']
            I = cata[k] == v
            cata = cata[I]

        # make homogeneous catalog
        for k in ['Lon', 'Lat', 'MAG', 'Time', 'Depth', 'Id']:
            if type(trans[k]) is str:
                self._catDataFrame[k] = cata[trans[k]]
            else: # use callable
                self._catDataFrame[k] = trans[k](cata)

        #: selected elements after filtering
        self.I = np.array(len(self._catDataFrame) * [True])

        if self.f_eta_0:
            self.filter_decluster()
            self.downsample_catalog()



    @property
    def catDataFrame(self):
        """
        the filtered datalog
        """
        return self._catDataFrame[self.I]

    @property
    def lon(self):
        return self._catDataFrame.loc[self.I, 'Lon'].to_numpy()

    @property
    def _lon(self):
        return self._catDataFrame['Lon']

    @property
    def lat(self):
        return self._catDataFrame.loc[self.I, 'Lat'].to_numpy()

    @property
    def _lat(self):
        return self._catDataFrame['Lat']


    @property
    def depth(self):
        return self._catDataFrame.loc[self.I, 'Depth'].to_numpy()

    @property
    def _depth(self):
        return self._catDataFrame['Depth']

    @property
    def time(self):
        return self._catDataFrame.loc[self.I, 'Time'].to_numpy()

    @property
    def _time(self):
        return self._catDataFrame['Time']

    @property
    def mag(self):
        return self._catDataFrame.loc[self.I, 'MAG'].to_numpy()


    @property
    def _mag(self):
        return self._catDataFrame['MAG']


    @cached_property
    def coordinates(self):
        """
        lazy and one time only evalutaion is important to avoid
        shrinking down on filtering
        """
        if not hasattr(self, '_coordinates') or self._coordinates is None:
            clat = self.kwargs.get('CLAT', np.mean(self.lat))
            clon = self.kwargs.get('CLON', np.mean(self.lon))
            angle = self.kwargs.get('ROTATION_ANGLE', 0)
            self._coordinates = coord.Coordinates(
                center_lon=clon, center_lat=clat, rotation_angle=angle)
        return self._coordinates

    @cached_property
    def xy(self):
        """
        returns a cartesion mapping of the data
        """
        return self.coordinates.lonlat_to_rotated_xy(self.lon, self.lat)

    @property
    def _xy(self):
        return self.coordinates.lonlat_to_rotated_xy(self._lon, self._lat)

    def filter_time(self, tmin, tmax):
        I = (self._catDataFrame['Time'] >= tmin & self._catDataFrame['Time'] <= tmax)
        self.I = self.I & I
        return I

    def filter_lon(self, lonmin, lonmax):
        I =  (self._catDataFrame['Lon'] >= lonmin & self._catDataFrame['Lon'] <= lonmax)
        self.I = self.I & I
        return I

    def filter_lon(self, latmin, latmax):
        self.latmin, self.latmax = latmin, latmax
        I =  (self._catDataFrame['Lat'] >= latmin & self._catDataFrame['Lat'] <= latmax)
        self.I = self.I & I
        return I

    def filter_xy(self, xmin, xmax, ymin, ymax):
        #self.xmin, self.xmax, self.ymin, self.ymax =  xmin, xmax, ymin, ymax

        x, y = self._xy
        I =  (x >= xmin) & (x <= xmax) &\
               (y >= ymin) & (y <=  ymax)
        self.I = self.I & I
        return I

    def filter_mag(self, magmin, magmax=None):
        I =  (self.self._catDataFrame['MAG'] >= magmin)
        self.I = self.I & I
        if magmax:
            I = I &  (self.self._catDataFrame['MAG'] <= magmax)
            self.I = self.I & I
        return I

    def filter_decluster(self, Mc = None, f_eta_0=None):
        MMc = Mc if not self.Mc else self.Mc
        ff_eta_0 = f_eta_0 if not self.f_eta_0 else self.f_eta_0

        I = dc.decluster(self._catDataFrame, MMc, ff_eta_0)
        self.I = I & self.I
        return I

    def clear_selection(self):
        self.I = True & self.I

    def downsample_catalog(self, I=None):
        I = I if I else self.I
        self._catDataFrame = self._catDataFrame.copy()[I]
        self.I = np.full(len(self._catDataFrame), True)

        return self._catDataFrame


    @classmethod
    def binning(cls, x=None, y=None, xbins=None, ybins=None, binfunc=None):

        """
        I = bin_indices(x, y, xbins, ybins)
        nbinx = len(xbins) - 1
        nbiny = len(ybins) - 1
        """


        i = np.digitize(x, xbins)
        j = np.digitize(y, ybins)

        print('ijshape', i.shape, j.shape)

        nbinx = len(xbins) - 1
        nbiny = len(ybins) - 1

        print(nbinx, nbiny)

        ## count the number of events in each bin
        resc = np.zeros((nbiny, nbinx), dtype=float)
        resi = []
        for jj in range(nbiny):
            for ii in range(nbinx):
                II = np.logical_and( jj+1 == j, ii+1 == i)
                resi.append(II)
                resc[ nbiny - jj -1, ii] = binfunc( II )
        '''
        n = np.arange(nbinx*nbiny)
        resi=[n[I[i]]==i for i in range(nbinx*nbiny) ]
        resc=np.array([binfunc(resi[i]) for i in range(nbinx*nbiny) ])
        '''
        return {'counts' : resc, 'I': resi}


    '''

    @property
    def binningI(self, size=None):
        if self._binningI is None:
            dkm = size if size else self.BINSIZE

            x_rot, y_rot = self.xy
            xmin = self.xmin if self.xmin else x_rot.min()
            xmax  = self.xmax if self.xmax else x_rot.max()
            ymin = self.ymin if self.ymin else y_rot.min()
            ymax = self.ymax if self.ymax else y_rot.max()

            nbinx = int(np.ceil((xmax - xmin) / dkm))
            nbiny = int(np.ceil((ymax - ymin) / dkm))

            xbins = xmin + dkm * np.arange(nbinx+1)
            ybins = ymin + dkm * np.arange(nbiny+1)

            i = np.digitize(x_rot, xbins)
            j = np.digitize(y_rot, ybins)

            self._binningI = nbiny * i + j, nbinx, nbiny, i, j, xmin, xmax, ymin, ymax

        return self._binningI

    '''

    def binning_of(self, size=None, binfunc=None):
        dkm = size if size else self.BINSIZE

        x_rot, y_rot = self.xy

        #if self.xmin is None:
        #    raise Exception('expected other ')

        xmin = self.xmin if self.xmin else x_rot.min()
        xmax  = self.xmax if self.xmax else x_rot.max()
        ymin = self.ymin if self.ymin else y_rot.min()
        ymax = self.ymax if self.ymax else y_rot.max()

        nbinx = int(np.ceil((xmax - xmin) / dkm))+1
        nbiny = int(np.ceil((ymax - ymin) / dkm))+1



        xbins = xmin + dkm * np.arange(nbinx+1)
        ybins = ymin + dkm * np.arange(nbiny+1)

        res = CatalogMixin.binning(
            x=x_rot, y=y_rot,
            xbins = xbins, ybins = ybins,
            binfunc = binfunc
        )

        return {
            'nbinx': nbinx,
            'nbiny': nbiny,
            'counts': res['counts'],
            'I': res['I'],
            'xylimits':  (xmin, xmax, ymin, ymax),
            'extent': (xmin, xmin + nbinx * dkm, ymin, ymin+nbiny*dkm)
        }

    def binning_count(self, size=None):
        dkm = size if size else self.BINSIZE
        tmp = self.binning_of(size=dkm, binfunc=lambda I: sum(I))
        self.n = tmp['nbiny']
        self.m = tmp['nbinx']
        return tmp

    def binning_mag(self, size=None):

        dkm = size if size else self.BINSIZE
        return self.binning_of(size=dkm, binfunc=lambda I: sum(self.mag[I]))

    @property
    def extent(self):
        """
        for plotting a 2D image
        """
        return self.binning_count()['extent']




class MapMixin(pgd.Mixin2D):



    def plot_coastlines(self, axis=None, color='red', linewidth=1, **kwargs):

        ax = axis if axis else plt.gca()

        for poly in self.coastlines:
            lon, lat = poly[:, 0], poly[:, 1]
            x, y = self.coordinates.lonlat_to_rotated_xy(lon, lat)
            poly = np.column_stack((x, y))
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=linewidth, **kwargs)

        ax.set_aspect("equal")

        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

    def plot_event_locations(self, axis=None, **kwargs):
        ax = axis if axis else plt.gca()
        ax.scatter(*self.xy, **kwargs)
        ax.set_aspect("equal")
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

    def imshow(self, d, xylimext=None, **kwargs):
        ax = plt.gca()
        ax.set_aspect("equal")
        if xylimext:
            xmin, xmax, ymin, ymax = xylimext['xylimits']
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            extent = xylimext['extent']
        else:
            extent = None
        return super().imshow(d,  extent=extent, **kwargs)


class CatalogAnalyserMixin:
    """
    Shared base for spatial field analysers that work with a catalog.
    Assumes self.catalog is provided — either by a catalog mixin in the MRO
    or created explicitly in a subclass __init__.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.catalog = kwargs['catalogclass'](**kwargs)

    @cached_property
    def binning_count(self):
        return self.catalog.binning_count()


    @property
    def ny(self):
        return self.binning_count['nbiny']

    @property
    def nx(self):
        return self.binning_count['nbinx']




    def imshow(self, pic, **kwargs):
        return self.catalog.imshow(pic, xylimext=self.binning_count, **kwargs)

    def plot_event_locations(self, **kwargs):
        self.catalog.plot_event_locations(**kwargs)

    def plot_coastlines(self, **kwargs):
        self.catalog.plot_coastlines(**kwargs)

    def _setup_grid(self):
        """Set m, n, rho from the current catalog binning."""
        self.m = self.binning_count['nbinx']
        self.n = self.binning_count['nbiny']
        self.rho = self.rhokm / self.BINSIZE

    def _apply_prior(self, calc, pmean, **kwargs):
        """
        Set Gaussian prior on calc via COVARIANCECLASS or PRECISIONCLASS.
        Returns the covariance matrix (covariance path) or cholprec (precision path)
        so the caller can derive submatrix priors for subsets of bins.
        Extra kwargs (e.g. sparse=True) are forwarded to set_prior_Gaussian.
        """
        if self.COVARIANCECLASS:
            if self.PRECISIONCLASS:
                raise Exception('too many covariance structures')
            covar = self.COVARIANCECLASS(**self.kwargs)
            calc.set_prior_Gaussian(prior_mean=pmean, prior_covariance=covar, **kwargs)
            return covar
        else:
            prec, cholprec = self.PRECISIONCLASS(cholesky=True, **self.kwargs)
            calc.set_prior_Gaussian(prior_mean=pmean, prior_precision=prec, **kwargs)
            calc.cholprec = cholprec
            return cholprec


class AValueMixin(CatalogAnalyserMixin):
    """
    scaling with binsize h = BiNSIZE
    lam  h^2
    f    1   # represents the value of the latent field at grid points
    a    1   # is independent on the bin size

    gamma_i poisson rate is (lamda/surface) h^2 * 10^a
    Covariance of f's is point evaluation at the grid cells
    rho[pixel] = rho[km]/h
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_calc()

    def update_calc(self):
        self._setup_grid()
        self.calc = self.PGCLASS(**self.kwargs)
        self.calc.alpha = 1
        self.calc.beta = 1
        self.calc.gamma = 1
        self.calc.lam = self.lamps * self.BINSIZE**2

        pmean = self.prior_mean_value_f
        self._apply_prior(self.calc, pmean)
        self.calc.set_nobs(self.binning_count['counts'])

    def max_logposterior_estimator(self, f0=None, **kwargs):
        self.calc.alpha = 1
        self.calc.beta = 1
        self.calc.gamma = 1
        self.calc.lam = self.lamps * self.BINSIZE**2
        return self.calc.max_logposterior_estimator(f0, **kwargs)

    def sample_posterior(self, **kwargs):
        self.calc.alpha = 1
        self.calc.beta = 1
        self.calc.gamma = 1
        self.calc.lam = self.lamps * self.BINSIZE**2
        return self.calc.sample_posterior(**kwargs)

    @property
    def prior_mean_value_f(self):
        return np.full(self.n * self.m, self.f_from_a(self.prior_avalue))

    def update_data(self):
        """
        ensures recomputation of binin from the catalog
        This allows modifications of the catalog
        """
        self.calc.set_data(self.catalog.binning_count['counts'].ravel())

    def f_from_a(self, a):
        return self.calc.f_from_field(10**a*self.BINSIZE**2)

    def a_from_f(self, f):
        return np.log( self.calc.field_from_f(f)/self.BINSIZE**2) / np.log(10)

    def density_under_gaussian(self, a, m, v2):
        return self.calc.prior_single_bin_field(
            10**a * self.BINSIZE**2, m, v2
        ) * 10**a * np.log(10)*self.BINSIZE**2

class AValueAnalyser(pgd.KwargsUpdater, AValueMixin, pgd.KwargsSink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BetaPolyaGammaDensity(pgd.KwargsUpdater, pgd.BetaSigmoidMixin, pgd.Density, pgd.KwargsSink):
    """
    Docstring for PolyGammaDensity
    """
    def __init__(self,  **kwargs):
        super().__init__(
            **kwargs
        )

class BetaPolyaGammaDensity2D(pgd.KwargsUpdater, pgd.Mixin2D, pgd.BetaSigmoidMixin, pgd.Density, pgd.KwargsSink):
    """
    Docstring for PolyGammaDensity2D
    specialized in 2Dimensional data
    """
    def __init__(self,  **kwargs):
        super().__init__(
            **kwargs
        )
rng = np.random.default_rng()

def trunc_exp(m, D, lam):
    m, D, lam = np.broadcast_arrays(m, D, lam)       # m, D, lam may be scalars or arrays
    a = m - D/2
    u = rng.random(m.shape)                          # one independent draw per element
    with np.errstate(divide="ignore", invalid="ignore"):
        x = a - np.log1p(u * np.expm1(-lam*D)) / lam
    return np.where(lam == 0, a + D*u, x)            # uniform limit where lam == 0

class BValueMixin(CatalogAnalyserMixin):

    def __init__(self,  **kwargs):
        self.kwargs['PGCLASS'] = BetaPolyaGammaDensity2D

        super().__init__(**kwargs)

        self.lam = 2*self.beta_from_b(self.prior_bvalue)
        self.catalog.BINSIZE= kwargs['BINSIZE']

        self.update_data()
        self.update_calc()

        self.magnitudes = [ self.catalog.mag[I] for I in self.binning_count['I']]

    @cached_property
    def Ibool(self):
        return self.calc.nobs > 0

    @cached_property
    def I(self):
        n = np.arange(self.calc.nbins)
        return n[self.Ibool]

    def resample_magnitudes(self, f):
        beta = self.calc.field_from_f(f)
        D = self.magbinsize
        w = np.zeros(self.calcI.nbins)

        for i, j in enumerate(self.I):
            I = self.binning_count['I'][j]

            #m = trunc_exp(self.catalog.mag[I], D, beta[i] )
            m = self.catalog.mag[I]
            w[i] = np.sum(m - self.Mc + D/2)
        self.calcI.weight = w
        return f

    def sample_posterior(self, **kwargs):
        return self.calcI.sample_posterior(after_cycle_method=self.resample_magnitudes, **kwargs)

    def binning_mag(self):
        return self.catalog.binning_mag

    def update_data(self):
        """
        ensures recomputation of binin from the catalog
        This allows modifications of the catalog
        """
        pass
        #self.I = self.catalog.binning_count['counts'].ravel() > 0  # the set of active boxes

    def update_calc(self):
        self._setup_grid()

        self.calc  = self.PGCLASS(**self.kwargs)
        self.calcI = self.PGCLASS(**self.kwargs)

        counts = self.binning_count['counts'].ravel()
        self.calc.set_data(counts)
        self.calcI.set_data(counts[self.Ibool])

        pmean = self.prior_mean_value_f

        result = self._apply_prior(self.calc, pmean, sparse=self.SPARSE)
        if self.COVARIANCECLASS:
            sub = result[np.ix_(self.Ibool, self.Ibool)]
            self.calcI.set_prior_Gaussian(prior_mean=pmean[self.I],
                                         prior_covariance=sub, sparse=self.SPARSE)
        else:
            covar = ck.inverse_sparse(result)
            self.calcI.set_prior_Gaussian(prior_mean=pmean[self.Ibool],
                                          prior_covariance=covar[np.ix_(self.Ibool, self.Ibool)], sparse=False)

        self.calc.weight = self.calcI.weight = 1
        self.calc.lam = self.calcI.lam = self.lam

    @property
    def prior_mean_value_f(self):
        return np.zeros(self.n * self.m)

    def f_from_b(self, b):
        return self.calc.f_from_field(self.beta_from_b(b))

    def b_from_f(self, f):
        return self.b_from_beta(self.calc.field_from_f(f))

    def b_from_beta(self, beta):
        return beta / np.log(10)

    def beta_from_b(self, b):
        return np.log(10) * b


class BValueAnalyser(pgd.KwargsUpdater, BValueMixin, pgd.KwargsSink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

if __name__=='__main__':
    pass
