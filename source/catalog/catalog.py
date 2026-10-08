import pandas as pd
import numpy as np
import sys
from pathlib import Path
from  polyagammapoisson.catalog import declustering as dc
import scipy
import matplotlib.pyplot as plt
import pickle
import zipfile
import geopandas as gpd
from shapely.geometry import box
import polyagammapoisson.polyagammadensity as pgd
import polyagammapoisson.catalog.coordinates as coord
import os



class Catalog:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._coordinates = None

        if 'catalog' in kwargs and self.catpickle is None:
            self.load_catalog()

        else:
            with open(kwargs['catpickle'], 'rb') as f:
                self._catDataFrame = pickle.load(f)

    def load_catalog(self, **kwargs):
        
        self.catalog = kwargs.get('catalog', self.catalog)

        path, kwargs, trans = self.catalog
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


    @property
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

    @property
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
        i = np.digitize(x, xbins)
        j = np.digitize(y, ybins)

        nbinx = len(xbins) - 1
        nbiny = len(ybins) - 1

        ## count the number of events in each bin
        res = np.zeros((nbiny, nbinx), dtype=float)
        for jj in range(nbiny):
            for ii in range(nbinx):
                II = np.logical_and( jj+1 == j, ii+1 == i)
                res[ nbiny - jj -1, ii] = binfunc( II )

        return res

 
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
        x_rot = x_rot
        y_rot = y_rot

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

        res = Catalog.binning(
            x=x_rot, y=y_rot,
            xbins = xbins, ybins = ybins,
            binfunc = binfunc
        )

        return {
            'nbinx': nbinx,
            'nbiny': nbiny,
            'counts': res,
            'xylimits':  (xmin, xmax, ymin, ymax),
            'extent': (xmin, xmin + nbinx * dkm, ymin, ymin+nbiny*dkm)
        }
    
    @property
    def binning_count(self, size=None):
        if hasattr(self, '_binning_count') and not self._binning_count is None:
            return self._binning_count

        dkm = size if size else self.BINSIZE
        self._binning_count = self.binning_of(size=dkm, binfunc=lambda I: sum(I))
        return self._binning_count

    @property
    def binning_mag(self, size=None):
        if hasattr(self, '_binning_mag') and not self._binning_mag is None:
            return self._binning_mag
        
        dkm = size if size else self.BINSIZE
        self._binning_mag= self.binning_of(size=dkm, binfunc=lambda I: sum(self.mag[I]))
        return self._binning_mag

    @property
    def extent(self):
        """
        for plotting a 2D image
        """
        return self.binning_count['extent']



class MapMixin(pgd.Mixin2D):


    @property
    def coastlines(self):
        pass


    def plot_coastlines(self, color='red', linewidth=1, **kwargs):
        ax = plt.gca()

        for poly in self.coastlines:
            lon, lat = poly[:, 0], poly[:, 1]
            x, y = self.coordinates.lonlat_to_rotated_xy(lon, lat)
            poly = np.column_stack((x, y))
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=linewidth, **kwargs)

        #ax.set_aspect("equal")
        #ax.set_xlim(self.xmin, self.xmax)
        #ax.set_ylim(self.xmin, self.xmax)

    def plot_event_locations(self, **kwargs):
        ax = plt.gca()
        ax.scatter(*self.xy, **kwargs)
        #ax.set_aspect("equal")
        #ax.set_xlim(self.xmin, self.xmax)
        #ax.set_ylim(self.xmin, self.xmax)

    def imshow(self, d, xylimext=None, **kwargs):
        ax = plt.gca()
        ax.set_aspect("equal")
        if xylimext:
            xmin, xmax, ymin, ymax = xylimext['xylimits']
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            extent = xylimext['extent']
            print (xylimext)
            print( xylimext['xylimits'])
        else:
            extent = None 
        super().imshow(d, extent=extent, **kwargs)




class AValueMixin:

    def __init__(self, 
                #COVARCLASS=None, 
                #PGCLASS=pgd.PolyaGammaDensity2D,
                #PRECISIONCLASS=None,
                #prior_avalue=None,
                #lam=None,
                #v2=None,
                #rho=None,
                #sparse=False,
                #boundary='zero',
                #magbinsize=0.5
                
                 **kwargs):
        super().__init__(**kwargs)

        #self.prior_avalue = self.kwargs.get(
        #self.magbinsize = magbinsize


        self.m = self.binning_count['nbinx']
        self.n = self.binning_count['nbiny']

        self.calc = self.PGCLASS(**self.kwargs)
        self.rho = self.rho / self.BINSIZE
        self.h = self.BINSIZE

        if self.prior_avalue:
            pmean = np.full( self.n * self.m, self.f_from_a(self.prior_avalue))
            self.prior_mean_value_f = pmean
        else:
            pmean = self.prior_mean_value_f

        if self.COVARCLASS:
            covar = self.COVARCLASS(**self.kwargs)
            self.calc.set_prior_Gaussian(prior_mean=pmean, prior_covariance=covar)
            if self.PRECISIONCLASS:
                raise Exception('too many covariance structures')
        else: 
            prec, cholprec = self.PRECISIONCLASS(cholesky=True, **self.kwargs)
            self.calc.set_prior_Gaussian(prior_mean=pmean, prior_precision=prec)
            self.calc.cholprec = cholprec


        self.calc.alpha = 1
        self.calc.beta = 1 #self.BINSIZE**2
        self.calc.gamma = 0.00001 * self.BINSIZE**2 
        self.calc.lam *= self.h**2
        
        self.set_data()

    @property
    def prior_mean_value_f(self):
        return np.full(self.n * self.m, self.f_from_a(self.prior_avalue))

    def set_data(self):
        """
        ensures recomputation of binin from the catalog
        This allows modifications of the catalog
        """
        self._binning_count = None       
        self.calc.set_data(self.binning_count['counts'].ravel())


    def f_from_a(self, a):
        return self.calc.f_from_field(10**a)
    
    def a_from_f(self, f):
        return np.log( self.calc.field_from_f(f)/self.BINSIZE**2) / np.log(10)

    def density_under_gaussian(self, a, m, v2):
        return self.calc.prior_single_bin_field(10**a * self.BINSIZE**2, m, v2) * 10**a * np.log(10)*self.BINSIZE**2
    
    
class BetaPolyaGammaDensity2D(pgd.KwargsUpdater, pgd.Mixin2D, pgd.BetaSigmoidMixin, pgd.KwargsSink):
    """
    Docstring for PolyGammaDensity2D
    specialized in 2Dimensional data
    """
    def __init__(self,  **kwargs):
        super().__init__(
            **kwargs
        )


class BValueMixin(AValueMixin):

    def __init__(self,  **kwargs):
        self.kwargs['PGCLASS'] = BetaPolyaGammaDensity2D
        self.lam = 2*self.beta_from_b(self.prior_bvalue)

        super().__init__(**kwargs)


        f = self.prior_mean_value_f
        b = self.b_from_f(f)
        b1, b2 = self.b_from_f(f-0.5), self.b_from_f(f+0.5)
        print(f, b, b1, b2)
        
        
        magsum = self.binning_mag['counts'].ravel()
        nobs = self.calc.nobs.ravel()

        #weight = np.where(nobs > 0, magsum + nobs * self.magbinsize/2 - nobs * self.Mc, 0)

        self.calc.weight = magsum

    
    @property
    def prior_mean_value_f(self):
        return 0 * self.f_from_b(self.prior_bvalue)


    def f_from_b(self, b):
        return self.calc.f_from_field(self.beta_from_b(b))

    def b_from_f(self, f):
        return self.b_from_beta(self.calc.field_from_f(f))

    def b_from_beta(self, beta):
        return beta / np.log(10)

    def beta_from_b(self, b):
        return np.log(10) * b



if __name__=='__main__':
    pass
