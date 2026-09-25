import pandas as pd
import numpy as np
import coordinates
import sys
from pathlib import Path
import declustering as dc
import scipy
import matplotlib.pyplot as plt
import pickle
import zipfile
import geopandas as gpd
from shapely.geometry import box
import polyagammapoisson.polyagammadensity as pgd
import polyagammapoisson.covariance_kernels as ck


SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent

#: describe your catalog data
CATALOGS = {
    'HORUS':  (
        SCRIPT_DIR / 'data/HORUS/HORUS_Ita_Catalog.txt',
        {'sep': '|'},   # kwargs for pd.read_csv
        {}              # mapping for column names
    ),
    'INGV':   (
        SCRIPT_DIR / 'data/italy_ingv_m2point5_2015-2026.txt',
        {'sep': '|', 'parse_dates': ['Time']},    # kwargs for pd.read_csv
        {'Lon': 'Longitude',
          'Lat': 'Latitude',
          'MAG': 'Magnitude',
          'Time': 'Time',
          'Depth': 'Depth/Km',
          'Id': '#EventID',
          'type_to_keep': ('EventType', 'earthquake') # used to filter from other than earthquakes
        }
    )
}

class Catalog:

    def __init__(self, catname='', BINSIZE=1, **kwargs):
        print(kwargs)
        super().__init__(**kwargs)

        if not catname in CATALOGS:
            raise Exception(f'no such catalog: {catname}')

        #self.kwargs = kwargs
        self.BINSIZE = BINSIZE

        path, kwargs, trans = CATALOGS[catname]
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
        

        self._coordinates = None

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
        return self._catDataFrame['Lon'].to_numpy()
    
    @property
    def lat(self):
        return self._catDataFrame.loc[self.I, 'Lat'].to_numpy()

    @property
    def _lat(self):
        return self._catDataFrame['Lat'].to_numpy()

    
    @property
    def time(self):
        return self._catDataFrame.loc[self.I, 'Time']

    @property
    def _time(self):
        return self._catDataFrame['Time']

    @property
    def mag(self):
        return self._catDataFrame.loc[self.I, 'MAG']


    @property
    def _mag(self):
        return self._catDataFrame['MAG']


    @property
    def coordinates(self):
        if not hasattr(self, '_coordinates') or self._coordinates is None:
            clat = self.kwargs.get('CLAT', np.mean(self.lat))
            clon = self.kwargs.get('CLON', np.mean(self.lon))
            angle = self.kwargs.get('ROTATION_ANGLE', 0)

            self._coordinates = coordinates.Coordinates(
                center_lon=clon, center_lat=clat, rotation_angle=angle
            )
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
        I =  (self._catDataFrame['Lat'] >= latmin & self._catDataFrame['Lat'] <= latmax)
        self.I = self.I & I
        return I

    def filter_xy(self, xmin, xmax, ymin, ymax):

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
        Mc = Mc if Mc else self.kwargs.get('Mc', self.mag.min())
        f_eta_0 = f_eta_0 if f_eta_0 else -4.6  # TODO

        I = dc.decluster(self._catDataFrame, Mc, f_eta_0)
        self.I = I & self.I
        return I

    def clear_selection(self):
        self.I = True & self.I

    def downsample_catalog(self, I=None):
        I = I if I else self.I
        self._catDataFrame = self._catDataFrame.copy()[I]
        self.I = np.full(len(self._catDataFrame), True)
        return self._catDataFrame

    @property
    def binning_count(self, size=None):
        if hasattr(self, '_binning_count') and not self._binning_count is None:
            return self._binning_count
        
        dkm = size if size else self.BINSIZE

        x_rot, y_rot = self.xy
        nbinx = int(np.ceil((x_rot.max() - x_rot.min()) / dkm))
        nbiny = int(np.ceil((y_rot.max() - y_rot.min()) / dkm)) 

        i = np.digitize(x_rot, x_rot.min() + (np.arange(nbinx + 1)) * dkm) - 1
        j = np.digitize(y_rot, y_rot.min() + (np.arange(nbiny + 1)) * dkm) - 1

        I = nbiny * i + j

        ## count the number of events in each bin
        counts = np.zeros((nbinx, nbiny), dtype=int)
        for ii in range(nbinx):
            for jj in range(nbiny):
                counts[ii, jj] = np.sum((i == ii) & (j == jj))


        self._binning_count = {
            'i' : i,
            'j' : j,
            'nbinx': nbinx,
            'nbiny': nbiny,
            'counts': counts,
            'extent': (x_rot.min(), x_rot.min() + nbinx * dkm, y_rot.min(), y_rot.min() + nbiny * dkm)
        }

        return self._binning_count


    @property
    def binning_mag(self, size=None):
        if hasattr(self, '_binning_mag') and not self._binning_mag is None:
            return self._binning_mag
        
        dkm = size if size else self.BINSIZE

        x_rot, y_rot = self.xy
        nbinx = int(np.ceil((x_rot.max() - x_rot.min()) / dkm))
        nbiny = int(np.ceil((y_rot.max() - y_rot.min()) / dkm)) 

        i = np.digitize(x_rot, x_rot.min() + (np.arange(nbinx + 1)) * dkm) - 1
        j = np.digitize(y_rot, y_rot.min() + (np.arange(nbiny + 1)) * dkm) - 1

        I = nbiny * i + j

        ## count the number of events in each bin
        magsum = np.zeros((nbinx, nbiny), dtype=float)
        for ii in range(nbinx):
            for jj in range(nbiny):
                magsum[ii, jj] = np.sum(self.mag[I == ii*nbiny + jj])


        self._binning_mag= {
            'i' : i,
            'j' : j,
            'nbinx': nbinx,
            'nbiny': nbiny,
            'magsum': magsum,
            'extent': (x_rot.min(), x_rot.min() + nbinx * dkm, y_rot.min(), y_rot.min() + nbiny * dkm)
        }

        return self._binning_mag


    @property
    def extent(self):
        """
        for plotting a 2D image
        """
        return self.binning_count['extent']



class MapMixin:


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

        ax.set_aspect("equal")
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
 

class AValueMixin:

    def __init__(self, 
                COVARCLASS=None, 
                PGCLASS=pgd.PolyaGammaDensity2D,
                PRECISIONCLASS=None,
                prior_avalue=1,
                lam=None,
                v2=None,
                rho=None,
                sparse=False,
                boundary='zero',
                magbinsize=0.5,
                 **kwargs):
        super().__init__(**kwargs)

        self.prior_avalue = prior_avalue
        self.magbinsize = magbinsize


        n = self.binning_count['nbinx']
        m = self.binning_count['nbiny']
        
        self.calc = PGCLASS(lam=lam, n=n, m=m, sparse=sparse, **kwargs)


        pmean = np.full( n*m, self.prior_mean_value_f)

        if COVARCLASS:
            covar = COVARCLASS(n=n, m=m, v2=v2, rho=rho, boundary=boundary, **kwargs)
            self.calc.set_prior_Gaussian(prior_mean=pmean, prior_covariance=covar)
            if PRECISIONCLASS:
                raise Exception('too many covariance structures')
        else: 
            prec = PRECISIONCLASS(n=n, m=m, v2=v2, rho=rho, boundary=boundary, **kwargs)
            self.calc.set_prior_Gaussian(prior_mean=pmean, prior_precision=prec)


        self.calc.set_data(self.binning_count['counts'])


    @property
    def prior_mean_value_f(self):
        return self.f_from_a(self.prior_avalue)

    def f_from_a(self, a):
        return self.calc.f_from_field(10**a)
    
    def a_from_f(self, f):
        return np.log( self.calc.field_from_f(f)) / np.log(10)
    
    @property
    def completenes_mag(self):
        return self.kwargs.get('M0', np.min(self.mag))



class BValueMixin(AValueMixin):


    
    def __init__(self,  M0=None, prior_bvalue=1, lam=None, **kwargs):

        self.prior_bvalue = prior_bvalue
        if lam is None:
            lam = 2*self.beta_from_b(prior_bvalue)

        super().__init__(lam=lam, **kwargs)

        f = self.prior_mean_value_f
        b = self.b_from_f(f)
        print(f, b)
        
        magsum = self.binning_mag['magsum']
        print(magsum)

        self.calc.set_weight(magsum.ravel() - 
                self.binning_count['counts'].ravel() * (self.completenes_mag-self.magbinsize/2))

    
    @property
    def prior_mean_value_f(self):
        return self.f_from_b(self.prior_bvalue)


    def f_from_b(self, b):
        return self.calc.f_from_field(self.beta_from_b(b))

    def b_from_f(self, f):
        return self.b_from_beta(self.calc.field_from_f(f))

    def b_from_beta(self, beta):
        return beta / np.log(10)

    def beta_from_b(self, b):
        return np.log(10) * b


class Italy(MapMixin, Catalog):

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    ITALYCOASTLINE =  REPO_ROOT / "data" /  "coastlines/ne_10m_coastline.zip"
    EXTRACTED_COASTLINE_DIR = REPO_ROOT / "experiments" / "naturalearth" 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._coordinates = coordinates.Italy_Coordinates
        #self.cache_file = self.REPO_ROOT / "experiments" / "cache" / "italy_coastlines.pkl"

    @property
    def coastlines(self):
        if not hasattr(self, 'polygons') or self.polygons is None:

            if hasattr(self, 'cache_file') and self.cache_file.exists():
                print(f"Loading cached coastlines from {self.cache_file}")
                with open(self.cache_file, "rb") as file:
                    cache = pickle.load(file)

                self.polygons = cache.get("polygons", None)
                if self.polygons is not None:
                    return self.polygons

            with zipfile.ZipFile(open(Italy.ITALYCOASTLINE, "rb")) as z:
                z.extractall(Italy.EXTRACTED_COASTLINE_DIR)

            coast = gpd.read_file(
                Italy.EXTRACTED_COASTLINE_DIR / "ne_10m_coastline.shp"
            )


            bbox = box(6, 35, 19, 48)

            italy_coast = coast


            polygons = []
            for geom in italy_coast.geometry:
                if geom.geom_type == "LineString":
                    polygons.append(np.asarray(geom.coords))

                elif geom.geom_type == "MultiLineString":
                    for line in geom.geoms:
                        polygons.append(np.asarray(line.coords))

            self.polygons = polygons

            #with open(self.cache_file, "rb") as file:
            #    cache = pickle.load(file)

            #cache["polygons"] = self.polygons

            #with open(self.cache_file, "wb") as file:
            #    pickle.dump(cache, file)



        return self.polygons

class SicilyCalabria(Italy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filter_xy( -450, 250, -650, -190)
        self.downsample_catalog()


class StrettoDiMessina(Italy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filter_xy( -275, -69, -590, -420)
        self.downsample_catalog()

class ItalyBValue(BValueMixin, Italy):
    def __init__(self, **kwargs):
        self.kwargs=kwargs
        super().__init__(**kwargs)

class SicilyCalabriaBValue(BValueMixin,  SicilyCalabria ):
    def __init__(self, **kwargs):
        self.kwargs=kwargs
        super().__init__(**kwargs)

class StrettoDiMessinaBValue(BValueMixin, StrettoDiMessina):

    def __init__(self, **kwargs):
        self.kwargs=kwargs
        super().__init__(**kwargs)

class StrettoDiMessinaAValue(AValueMixin, StrettoDiMessina):

    def __init__(self, **kwargs):
        self.kwargs=kwargs
        super().__init__(**kwargs)



if __name__=='__main__':


    #Region = SicilyCalabriaBValue
    #Region = ItalyBValue
    Region = StrettoDiMessinaBValue

    C = Region( catname='INGV', BINSIZE=1,
                        PRECISIONCLASS=ck.precision_matern,
                        prior_bvalue=1, 
                        sparse=True,
                        boundary="symmetric",
                        v2=0.1, rho=20, M0=2.45)

    RegionA = StrettoDiMessinaAValue

    A = RegionA( catname='INGV', BINSIZE=1,
                        PGCLASS=pgd.RampDensity2D,
                        PRECISIONCLASS=ck.precision_matern,
                        prior_avalue= 0, 
                        sparse=True,
                        boundary="symmetric",
                        v2=1, rho=20, lam=10)

    print(SicilyCalabriaBValue.__mro__)


    


    #b = C.binning_mag['magsum']  / \
    #    np.where(C.binning_count['counts']>0, C.binning_count['counts'], 0.001)    

    plt.figure(figsize=(10, 8))
    plt.title('B-value')
    b = C.b_from_f(C.calc.max_logposterior_estimator())
    C.calc.imshow(b, 
               origin='lower', 
               extent=C.extent, cmap='jet')
    C.plot_coastlines(color='white', linewidth=4)
    plt.colorbar()

    plt.plot(*C.xy, '.g', markersize=2)

    plt.figure(figsize=(10, 8))

    plt.title('A-value at M0')
    a = A.a_from_f(A.calc.max_logposterior_estimator()) 

    A.calc.imshow(a, 
               origin='lower', 
               extent=C.extent, cmap='jet')
    C.plot_coastlines(color='white', linewidth=4)
    plt.colorbar()

    plt.plot(*C.xy, '.g', markersize=2)


    plt.figure(figsize=(10, 8))

    plt.title('A-value at 0')

    A.calc.imshow((a + 2.5*b)/np.log(10) + np.log(b), 
               origin='lower', 
               extent=C.extent, cmap='jet')
    C.plot_coastlines(color='white', linewidth=4)
    plt.colorbar()

    plt.plot(*C.xy, '.g', markersize=2)




    
    plt.show()