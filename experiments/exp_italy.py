import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd

import matplotlib.pyplot as plt
from geodatasets import get_path
import pickle

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "source"))

from coordinates import Italy_Coordinates as IC
import polyagammadensity as pgd
import covariance_kernels as ck

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DATA = REPO_ROOT / "data" / "preprocess_nnd_rot_cut_bin_Mc_2.5_eta_-4.60_dkm_2_events.csv"
ITALYCOASTLINE =  REPO_ROOT / "data" /  "coastlines/ne_10m_coastline.zip"
EXTRACTED_COASTLINE_DIR = REPO_ROOT / "experiments" / "naturalearth" 

DKM = 20
import io
import zipfile
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box


class ItalyData: 

    def __init__(self, BIN_SIZE_KM=2, Declusterd=True, rho=5.0, lam=5, var=1.0, prior_mean=0.0):
        self.BIN_SIZE_KM = BIN_SIZE_KM
        self.Declusterd = Declusterd
        self.rho = rho
        self.lam = lam
        self.var = var
        self.prior_mean = prior_mean
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

            italy_coast = coast.clip(bbox)


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
        self.sampler.set_data(self.counts.flatten())


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

        count = 0
        samples_to_plot = []
        plot_every = max(1, n_samples // n_plot_samples)

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

        f_sd = np.sqrt(f_M2 / (count - 1))

        return f_mean, f_sd, rate_mean, samples_to_plot

    def plot_posterior_samples(self, samples):

        plt.figure(figsize=(15, 8))

        for i, sample in enumerate(samples):
            plt.subplot(2, 3, i + 1)

            self.sampler.imshow( self.sampler.field_from_f(sample), extent=self.extent, origin="lower", cmap="viridis"
            )

        self.plot_coastlines()

        plt.title(f"Posterior rate sample {i + 1}")
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()


    def plot_posterior_summary(self, f_mean, f_sd, rate_mean):

        plt.figure(figsize=(10, 8))
        plt.title("Posterior mean of f")

        self.sampler.imshow(f_mean, extent=self.extent, origin="lower", cmap="viridis")

        self.plot_coastlines()
        plt.colorbar()


        plt.figure(figsize=(10, 8))
        plt.title("Posterior standard deviation of f")

        self.sampler.imshow(f_sd, extent=self.extent, origin="lower", cmap="viridis")

        self.plot_coastlines()
        plt.colorbar()


        plt.figure(figsize=(10, 8))
        plt.title("Posterior mean of rate")

        self.sampler.imshow(rate_mean, extent=self.extent, origin="lower", cmap="viridis")

        self.plot_coastlines()
        plt.colorbar()



if __name__ == "__main__":

    plt.figure(figsize=(10, 8))

    italy_data = ItalyData(BIN_SIZE_KM=5, rho=20, lam=1, var=1, Declusterd=True)
    italy_data.plot()


    f = italy_data.sampler.max_logposterior_estimator()

    plt.figure(figsize=(10, 8))
    italy_data.plot_coastlines()
    italy_data.sampler.imshow(f, extent=italy_data.extent, origin="lower", cmap="viridis")


    plt.figure(figsize=(10, 8))
    italy_data.plot_coastlines()
    italy_data.sampler.imshow(italy_data.sampler.field_from_f(f), extent=italy_data.extent, origin="lower", cmap="viridis")

    f_mean, f_sd, rate_mean, samples_to_plot = italy_data.posterior_summary(initial_f=f, n_samples=200, burn_in=50, thin=1)

    italy_data.plot_posterior_samples(samples_to_plot)

    italy_data.plot_posterior_summary(f_mean, f_sd, rate_mean)

    plt.show()