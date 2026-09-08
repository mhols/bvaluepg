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

    def __init__(self, catname='', **kwargs):
        if not catname in CATALOGS:
            raise Exception(f'no such catalog: {catname}')

        self.kwargs = kwargs

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
        
        dkm = size if size else self.kwargs.get('BINSIZE', 2)

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
        
        dkm = size if size else self.kwargs.get('BINSIZE', 2)

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


        self._binning_count = {
            'i' : i,
            'j' : j,
            'nbinx': nbinx,
            'nbiny': nbiny,
            'magsum': magsum,
            'extent': (x_rot.min(), x_rot.min() + nbinx * dkm, y_rot.min(), y_rot.min() + nbiny * dkm)
        }

        return self._binning_count


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


    def plot_coastlines(self):
        ax = plt.gca()

        for poly in self.coastlines:
            lon, lat = poly[:, 0], poly[:, 1]
            x, y = self.coordinates.lonlat_to_rotated_xy(lon, lat)
            poly = np.column_stack((x, y))
            ax.plot(poly[:, 0], poly[:, 1], color='red', linewidth=0.8)

        ax.set_aspect("equal")
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
 

class Italy(Catalog, MapMixin):

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    ITALYCOASTLINE =  REPO_ROOT / "data" /  "coastlines/ne_10m_coastline.zip"
    EXTRACTED_COASTLINE_DIR = REPO_ROOT / "experiments" / "naturalearth" 

    def __init__(self, *args, **kwargs):
        super(Italy, self).__init__(*args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super(SicilyCalabria, self).__init__(*args, **kwargs)

        self.filter_xy( -450, 250, -650, -190)
        self.downsample_catalog()


if __name__=='__main__':

    C = SicilyCalabria('INGV', BINSIZE=10)

    for t in set(C.original_catalog['EventType']):
        print( t, sum(C.original_catalog['EventType']==t))

    C.filter_decluster(f_eta_0=-6.2)
    print (f'old count: {len(C._catDataFrame.index)}')
    print (f'new count: {sum(C.I)}')

    print(C._catDataFrame)

    print(C.binning_count) 

    plt.figure(figsize=(10, 8))
    plt.imshow(C.binning_count['counts'].T, vmin=0, vmax=5, origin='lower', extent=C.extent, cmap='viridis')
    C.plot_coastlines()

    plt.show()