import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd

import matplotlib.pyplot as plt
from geodatasets import get_path
import pickle

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "source"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))

from coordinates import Italy_Coordinates as IC
import polyagammadensity as pgd
import covariance_kernels as ck
from preprocess_nnd_rot_cut_bin import *

# the coastline files are assumed to be in the "data" directory at the root of the repository, they can be downloaded from the following link:
# https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DATA = REPO_ROOT / "data" / "preprocess_nnd_rot_cut_bin_Mc_2.5_eta_-4.60_dkm_2_events.csv"
ITALYCOASTLINE =  REPO_ROOT / "data" /  "coastlines/ne_10m_coastline.zip"
EXTRACTED_COASTLINE_DIR = REPO_ROOT / "experiments" / "naturalearth" 
PLOTS_DIR = REPO_ROOT / "talks" / "2026_summer_yehuda" / "figures"


import io
import zipfile
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

### helper methods (copied from /data)
def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, sep="|", skiprows=0)
    df.columns = [str(column).strip() for column in df.columns]
    df = df.rename(
        columns={
            "#EventID": "event_id",
            "EventID": "event_id",
            "Time": "datetime",
            "Latitude": "lat",
            "Longitude": "lon",
            "Depth/Km": "depth",
            "Magnitude": "mag",
        }
    )
    required = ["datetime", "lat", "lon", "depth", "mag"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; available columns: {df.columns.tolist()}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for column in ["lat", "lon", "depth", "mag"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "event_id" in df.columns:
        df["event_id_num"] = pd.to_numeric(df["event_id"], errors="coerce")
    else:
        df["event_id"] = np.arange(1, len(df) + 1)
        df["event_id_num"] = df["event_id"].astype(float)

    df = df.dropna(subset=["datetime", "lat", "lon", "mag"]).copy()
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["second"] = df["datetime"].dt.second + df["datetime"].dt.microsecond / 1_000_000
    df = add_time_fields(df)
    return df.sort_values("datetime").reset_index(drop=True)


def create_synthetic_catalog() -> pd.DataFrame:
    """Placeholder for later synthetic background-only catalogues.

    Intended output columns:
    datetime, lat, lon, depth, mag, event_id, and optionally f_true/lambda_true.
    The returned dataframe can then be passed through the same pipeline below.

synthetic catalogues generieren
    ganzen Katalog fuer die pipeline generieren oder nur background events generieren :/
    vielleicht gleich mit dummy zeit und so
    1. Erzeuge ein wahres Feld, (Block, Balken oder Checkerboard)
    2. Ziehe daraus Poisson-Counts
    3. Wandle Counts in zufällige Eventpunkte pro Bin um
    4. Skaliere diese Punkte auf ein kuenstilches x_proj_km/y_proj_km-Gebiet
    5. Rechne daraus passende lon/lat zurück oder besser direkt synthetische lon/lat setzen
    6. Ergänze Dummy-Zeit, Tiefe, Magnitude (und optional lambda_true, f_true, bin_ix, bin_iy)

ok das wird zuviel fuer hier, besser in eigenem skript. ich muss nur aufpassen, dass die Spaltennamen und Formate passen

    """
    raise NotImplementedError("Synthetic catalogue generation will be added later.")


def filter_catalog(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if MIN_MAGNITUDE is not None:
        result = result[result["mag"] >= float(MIN_MAGNITUDE)].copy()
    if MAX_MAGNITUDE is not None:
        result = result[result["mag"] <= float(MAX_MAGNITUDE)].copy()
    if YEAR_MIN is not None:
        result = result[result["decimal_year"] >= float(YEAR_MIN)].copy()
    if YEAR_MAX is not None:
        result = result[result["decimal_year"] <= float(YEAR_MAX)].copy()
    return result.sort_values("datetime").reset_index(drop=True)


def make_eqcat(df: pd.DataFrame) -> EqCat:
    eqcat = EqCat()
    eqcat.data = {
        "N": df["N"].to_numpy(float),
        "Time": df["decimal_year"].to_numpy(float),
        "Mag": df["mag"].to_numpy(float),
        "Lat": df["lat"].to_numpy(float),
        "Lon": df["lon"].to_numpy(float),
        "Depth": df["depth"].fillna(0.0).to_numpy(float),
        "X": df["x_proj_km"].to_numpy(float),
        "Y": df["y_proj_km"].to_numpy(float),
    }
    return eqcat


def run_nnd_declustering(eqcat: EqCat) -> dict[str, np.ndarray]:
    dpar = {"D": NND_D, "b": NND_B, "Mc": MIN_MAGNITUDE}
    np.random.seed(RANDOM_SEED)
    eqcat.data["Z"] = eqcat.data["Depth"]
    return clustering.NND_eta(eqcat, dpar, correct_co_located=True, verbose=False)


def add_nnd_status(df: pd.DataFrame, nnd: dict[str, np.ndarray]) -> pd.DataFrame:
    result = df.copy()
    result["nnd_parent_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["nnd_eta"] = np.nan
    result["nnd_log10_eta"] = np.nan

    child_to_row = pd.Series(result.index.to_numpy(), index=result["N"].astype(float)).to_dict()
    for child, parent, eta in zip(nnd["aEqID_c"], nnd["aEqID_p"], nnd["aNND"]):
        row = child_to_row.get(float(child))
        if row is None:
            continue
        result.at[row, "nnd_parent_id"] = int(parent)
        result.at[row, "nnd_eta"] = float(eta)
        result.at[row, "nnd_log10_eta"] = float(np.log10(eta))

    result["nnd_is_triggered"] = result["nnd_log10_eta"].lt(ETA_THRESHOLD_LOG10).fillna(False)
    result["decluster_kept"] = ~result["nnd_is_triggered"]
    return result



class ItalyData: 

    def __init__(self, BIN_SIZE_KM=2, Declusterd=False, rho=5.0, lam=5, var=1.0, prior_mean=0.0, saturation = 1e6):
        self.BIN_SIZE_KM = BIN_SIZE_KM
        self.Declusterd = Declusterd
        self.rho = rho
        self.lam = lam
        self.var = var
        self.prior_mean = prior_mean
        self.saturation = saturation
        self.cache_file = REPO_ROOT / "data" / (
            f"italy_preprocessed_bin_{BIN_SIZE_KM}_declustered_{Declusterd}.pkl"
        )
        self.data = None
        self.polygons = None

        self._binn_data_in_rotated_coordinates()
        self._prepare_prior_kernels()
        self._prpare_sampler()


    @property
    def coastlines(self):
        if self.polygons is None:

            if self.cache_file.exists():
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

            with open(self.cache_file, "rb") as file:
                cache = pickle.load(file)

            cache["polygons"] = self.polygons

            with open(self.cache_file, "wb") as file:
                pickle.dump(cache, file)



        return self.polygons


    #def _binn_data_in_rotated_coordinates(self):
    #    data = pd.read_csv(PREPROCESSED_DATA, sep='|')
    #
        # if self.Declusterd:
        #     I = data['decluster_kept'] & data['inside_final_cut']
        # else:
        #     I = data['inside_final_cut']   

        # self.data = data.loc[I]


        # self.i, self.j, self.nbinx, self.nbiny, self.x_rot, self.y_rot, self.counts, self.extent =  \
        # IC.get_binned_data_in_rotated_coordinates(data['lon'].values, data['lat'].values, self.BIN_SIZE_KM)

        # return self.i, self.j, self.nbinx, self.nbiny, self.x_rot, self.y_rot, self.counts, self.extent


    def _binn_data_in_rotated_coordinates(self):

        if self.cache_file.exists():
            print(f"Loading cached data from {self.cache_file}")
            with open(self.cache_file, "rb") as file:
                cache = pickle.load(file)

            self.data = cache["data"]
            self.i = cache["i"]
            self.j = cache["j"]
            self.nbinx = cache["nbinx"]
            self.nbiny = cache["nbiny"]
            self.x_rot = cache["x_rot"]
            self.y_rot = cache["y_rot"]
            self.counts = cache["counts"]
            self.extent = cache["extent"]

            return self.i, self.j, self.nbinx, self.nbiny, self.x_rot, self.y_rot, self.counts, self.extent

        print(f"Cache file {self.cache_file} not found. Processing data from {PREPROCESSED_DATA}")

        data = pd.read_csv(PREPROCESSED_DATA, sep='|')

        if self.Declusterd:
            I = data['decluster_kept'] & data['inside_final_cut']
        else:
            I = data['inside_final_cut']

        self.data = data.loc[I].copy()

        self.i, self.j, self.nbinx, self.nbiny, self.x_rot, self.y_rot, self.counts, self.extent = \
            IC.get_binned_data_in_rotated_coordinates(
                self.data['lon'].values,
                self.data['lat'].values,
                self.BIN_SIZE_KM
        )

        cache = {
            "data": self.data,
            "i": self.i,
            "j": self.j,
            "nbinx": self.nbinx,
            "nbiny": self.nbiny,
            "x_rot": self.x_rot,
            "y_rot": self.y_rot,
            "counts": self.counts,
            "extent": self.extent,
        }

        with open(self.cache_file, "wb") as file:
            pickle.dump(cache, file)

        return self.i, self.j, self.nbinx, self.nbiny, self.x_rot, self.y_rot, self.counts, self.extent   


    def _prepare_prior_kernels(self):
        self.prior_kernel = ck.precision_matern(n=self.nbinx, m=self.nbiny, rho=self.rho/self.BIN_SIZE_KM, v2=self.var, boundary="symmetric")
        self.prior_mean = np.ones(self.nbinx * self.nbiny) * self.prior_mean

    def _prpare_sampler(self):
        self.sampler = pgd.PolyaGammaDensity2D(
            prior_precision=self.prior_kernel, 
            prior_mean=self.prior_mean, lam=self.lam,  n=self.nbinx, m=self.nbiny, seed=42
        )
        if self.saturation is not None:
            if self.saturation <= 0:
                counts_flat = np.clip(self.counts.flatten(), -self.saturation, None) + self.saturation
            
            else: ## if self.saturation > 0:
                counts_flat = np.clip(self.counts.flatten(), 0, self.saturation)
        else:
            counts_flat = self.counts.flatten()
        self.sampler.set_data(counts_flat)


    def plot_coastlines(self):
        ax = plt.gca()

        for poly in self.coastlines:
            lon, lat = poly[:, 0], poly[:, 1]
            x, y = IC.lonlat_to_rotated_xy(lon, lat)
            poly = np.column_stack((x, y))
            ax.plot(poly[:, 0], poly[:, 1], color='white', linewidth=0.8)

        ax.set_aspect("equal")
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])


    def plot(self):
        ax = plt.gca()

        self.plot_coastlines()

        plt.imshow(self.counts.T, extent=self.extent, origin="lower", cmap="viridis")

        evet_rot_x, event_rot_y = IC.lonlat_to_rotated_xy(self.data['lon'].values, self.data['lat'].values)
        plt.plot(evet_rot_x, event_rot_y, 'ro', markersize=2, alpha=0.3, label='Events')

        ax.set_aspect("equal")
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

    def save_plot(self, plot_name):

        catalog = "declustered" if self.Declusterd else "all"

        filename = (
            f"italy_{plot_name}"
            f"_bin_{self.BIN_SIZE_KM}km"
            f"_{catalog}.png"
        )

        path = PLOTS_DIR / filename

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved plot to {path}")

    def posterior_summary(
        self,
        initial_f,
        n_samples=200,
        burn_in=50,
        thin=1,
        n_plot_samples=6,
        random_seed=0,
    ):
    
        f_mean = np.zeros_like(initial_f)
        f_M2 = np.zeros_like(initial_f)
        rate_mean = np.zeros_like(initial_f)    
        rate_M2 = np.zeros_like(initial_f)

        count = 0
        samples_to_plot = []
        n_kept = (n_samples - burn_in) // thin
        plot_every = max(1, n_kept // n_plot_samples)

        for res in self.sampler.sample_posterior(
            n_iter=n_samples,
            burn_in=burn_in,
            thin=thin,
            initial_f=initial_f,
            random_seed=random_seed,
        ):
            count += 1

            if count % plot_every == 0 and len(samples_to_plot) < n_plot_samples:
                samples_to_plot.append(res.copy())

            delta = res - f_mean
            f_mean += delta / count
            f_M2 += delta * (res - f_mean)

            rate = self.sampler.field_from_f(res)
            rate_mean += (rate - rate_mean) / count
            rate_M2 += (rate - rate_mean) ** 2

        f_sd = np.sqrt(f_M2 / (count - 1))
        rate_sd = np.sqrt(rate_M2 / (count - 1))



        return f_mean, f_sd, rate_mean, rate_sd, samples_to_plot

    def plot_posterior_samples(self, samples):

        plt.figure(figsize=(15, 8))
        plt.suptitle("Posterior Samples of f")
        print("Number of samples to plot:", len(samples))   

        for i, sample in enumerate(samples):
            plt.subplot(2, 3, i + 1)
            self.plot_coastlines()
            self.sampler.imshow( sample, extent=self.extent, origin="lower", cmap="viridis"
            )
        
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()




    def plot_posterior_summary(self, f_mean, f_sd, rate_mean, rate_sd, f_map = None, rate_map = None, f_vmin=None, f_vmax=None, rate_vmin=None, rate_vmax=None):

        plt.figure(figsize=(10, 8))
        plt.title("Posterior mean of f")

        self.sampler.imshow(f_mean, extent=self.extent, vmin=f_vmin, vmax=f_vmax, origin="lower", cmap="viridis")

        self.plot_coastlines()
        plt.colorbar()
        self.save_plot("posterior_mean_f")


        plt.figure(figsize=(10, 8))
        plt.title("Posterior standard deviation of f")

        self.sampler.imshow(f_sd, extent=self.extent, origin="lower", cmap="viridis")

        self.plot_coastlines()
        plt.colorbar()
        self.save_plot("posterior_sd_f")


        plt.figure(figsize=(10, 8))
        plt.title("Posterior mean of rate")

        self.sampler.imshow(rate_mean, extent=self.extent, vmin=rate_vmin, vmax=rate_vmax, origin="lower", cmap="viridis")

        self.plot_coastlines()
        plt.colorbar()
        self.save_plot("posterior_mean_rate")

        plt.figure(figsize=(10, 8))
        plt.title("Posterior standard deviation of rate")

        self.sampler.imshow(rate_sd, extent=self.extent, origin="lower", cmap="viridis")

        self.plot_coastlines()
        plt.colorbar()
        self.save_plot("posterior_sd_rate")

        if f_map is not None:

            f_diff = f_mean - f_map

            f_diff_absmax = np.max(
                np.abs(f_diff)
            )

            plt.figure(figsize=(10, 8))
            plt.title("Difference in f: posterior mean - MAP")

            im = self.sampler.imshow( f_diff, extent=self.extent, origin="lower", cmap="RdBu_r", vmin=-f_diff_absmax,  vmax=f_diff_absmax)

            self.plot_coastlines()
            plt.colorbar(im)
            self.save_plot("difference_f")

        if rate_map is not None:

            rate_diff = rate_mean - rate_map

            rate_diff_absmax = np.max(np.abs(rate_diff))
            

            plt.figure(figsize=(10, 8))
            plt.title("Difference in rate: posterior mean - MAP")

            im = self.sampler.imshow( rate_diff, extent=self.extent, origin="lower", cmap="RdBu_r", vmin=-rate_diff_absmax,  vmax=rate_diff_absmax)

            self.plot_coastlines()
            plt.colorbar(im)
            self.save_plot("difference_rate")    



if __name__ == "__main__":

    italy_data = ItalyData(BIN_SIZE_KM=5, rho=20, lam=25, var=4, prior_mean = -7.0 ,Declusterd= True)
    

    f = italy_data.sampler.max_logposterior_estimator()

    rate_map = italy_data.sampler.field_from_f(f)

    f_mean, f_sd, rate_mean, rate_sd, samples_to_plot = italy_data.posterior_summary(initial_f=f, n_samples=110, burn_in=10, thin=10)


    f_vmin = min(np.min(f), np.min(f_mean))

    f_vmax = max( np.max(f), np.max(f_mean))



    rate_vmin = min(np.min(rate_map), np.min(rate_mean))

    rate_vmax = max(np.max(rate_map), np.max(rate_mean))


    plt.figure(figsize=(10, 8))
    


    plt.title("Italy Earthquake Data (Declustered)" if italy_data.Declusterd else "Italy Earthquake Data (All Events)")
    italy_data.plot()
    italy_data.save_plot("italy_earthquake_data")

    counts = italy_data.counts.flatten()


    plt.figure(figsize=(10, 8))
    plt.title("MAP estimate of f")
    italy_data.plot_coastlines()
    italy_data.sampler.imshow(f, extent=italy_data.extent, vmin=f_vmin, vmax=f_vmax, origin="lower", cmap="viridis")
    italy_data.save_plot("map_f")


    plt.figure(figsize=(10, 8))
    plt.title("MAP estimate of rate")
    italy_data.plot_coastlines()
    italy_data.sampler.imshow(rate_map, extent=italy_data.extent,  origin="lower", cmap="viridis")
    italy_data.save_plot("map_rate")

    italy_data.plot_posterior_samples(samples_to_plot)
    italy_data.save_plot("posterior_samples")

    italy_data.plot_posterior_summary(f_mean, f_sd, rate_mean, rate_sd, f_map=f, rate_map=rate_map, f_vmin=-14, f_vmax=7, rate_vmin=0, rate_vmax=25)

  

    italy_data = ItalyData(BIN_SIZE_KM=5, rho=20, lam=25, var=4, prior_mean = -7.0 ,Declusterd= False)
    

    f = italy_data.sampler.max_logposterior_estimator()

    rate_map = italy_data.sampler.field_from_f(f)

    f_mean, f_sd, rate_mean, rate_sd, samples_to_plot = italy_data.posterior_summary(initial_f=f, n_samples=110, burn_in=10, thin=10)


    f_vmin = min(np.min(f), np.min(f_mean))

    f_vmax = max( np.max(f), np.max(f_mean))



    rate_vmin = min(np.min(rate_map), np.min(rate_mean))

    rate_vmax = max(np.max(rate_map), np.max(rate_mean))


    plt.figure(figsize=(10, 8))
    


    plt.title("Italy Earthquake Data (Declustered)" if italy_data.Declusterd else "Italy Earthquake Data (All Events)")
    italy_data.plot()
    italy_data.save_plot("italy_earthquake_data")

    counts = italy_data.counts.flatten()


    plt.figure(figsize=(10, 8))
    plt.title("MAP estimate of f")
    italy_data.plot_coastlines()
    italy_data.sampler.imshow(f, extent=italy_data.extent, vmin=f_vmin, vmax=f_vmax, origin="lower", cmap="viridis")
    italy_data.save_plot("map_f")


    plt.figure(figsize=(10, 8))
    plt.title("MAP estimate of rate")
    italy_data.plot_coastlines()
    italy_data.sampler.imshow(rate_map, extent=italy_data.extent,  origin="lower", cmap="viridis")
    italy_data.save_plot("map_rate")

    italy_data.plot_posterior_samples(samples_to_plot)
    italy_data.save_plot("posterior_samples")

    italy_data.plot_posterior_summary(f_mean, f_sd, rate_mean, rate_sd, f_map=f, rate_map=rate_map, f_vmin=-14, f_vmax=7, rate_vmin=0, rate_vmax=25)

    plt.show()

