from polyagammapoisson.catalog.catalog import Catalog, MapMixin, BValueMixin, AValueMixin
from polyagammapoisson.catalog import coordinates
from pathlib import Path
import zipfile
import pickle
from shapely.geometry import box
import geopandas as gpd
import polyagammapoisson.polyagammadensity as pgd
import polyagammapoisson.covariance_kernels as ck
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

ITALYCOASTLINE =  REPO_ROOT / "data" /  "coastlines/ne_10m_coastline.zip"
EXTRACTED_COASTLINE_DIR = REPO_ROOT / "experiments" / "naturalearth" 

CATALOGS = {
    'HORUS':  (
            REPO_ROOT / "data" / 'HORUS/HORUS_Ita_Catalog.txt',
            {'sep': '|'},   # kwargs for pd.read_csv
            {}              # mapping for column names
        ),

    'INGV':   (
            REPO_ROOT / "data" /'italy_ingv_m2point5_2015-2026.txt',
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





class Italy(MapMixin, Catalog):

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

            with zipfile.ZipFile(open(ITALYCOASTLINE, "rb")) as z:
                z.extractall(EXTRACTED_COASTLINE_DIR)

            coast = gpd.read_file(
                EXTRACTED_COASTLINE_DIR / "ne_10m_coastline.shp"
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

    import matplotlib.pyplot as plt


    #: describe your catalog data
    catalog = CATALOGS["INGV"]

    S = SicilyCalabria(catalog=CATALOGS["INGV"], BINSIZE=10)


    #Region = SicilyCalabriaBValue
    #Region = ItalyBValue
    Region = StrettoDiMessinaBValue

    C = Region( catalog=CATALOGS["INGV"],
                        PRECISIONCLASS=ck.precision_matern,
                        prior_bvalue=1, 
                        sparse=True,
                        boundary="symmetric",
                        v2=0.1, rho=20, M0=2.45)

    RegionA = StrettoDiMessinaAValue

    A = RegionA( catalog=CATALOGS["INGV"], BINSIZE=1, 
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