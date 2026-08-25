

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import brentq
from scipy.special import roots_hermite, expit



REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import covariance_kernels as ck
import polyagammadensity as pgd



PLOTS_DIR = REPO_ROOT / "talks" / "2026_summer_yehuda" / "figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TYPE = "bars"  # "block", "bars", "checkerboard"

TRUTH_FILE = REPO_ROOT / "data" / "synthetic" / f"{TYPE}_synthetic_catalog_truth.npz"

LAM = 12.0

RHO = 3.0

PRIOR_MEAN = -3.7

PRIOR_VARIANCE = 4.0

BOUNDARY = "symmetric"

MAP_NITER = 300

PG_N_ITER = 500
PG_BURN_IN = 100
PG_THIN = 1

N_PLOT_SAMPLES = 6

RANDOM_SEED = 0



def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))




class SyntheticData:

    def __init__(self, truth_file=TRUTH_FILE, rho=RHO, lam=LAM, var=PRIOR_VARIANCE,
                 prior_mean=PRIOR_MEAN, prior_target="truth", boundary=BOUNDARY):

        self.truth_file = Path(truth_file)
        self.rho = rho
        self.lam = lam
        self.var = var
        self.boundary = boundary
        self.prior_target = prior_target

        self._load_truth()

        self.prior_mean_scalar = float(prior_mean)

        self._prepare_prior_kernel()
        self._prepare_sampler()




    def _load_truth(self):
        print(f"Loading synthetic data from {self.truth_file}")

        with np.load(self.truth_file) as data:
            self.lambda_true = data["lambda_true"].astype(float)
            self.f_true = data["f_true"].astype(float)
            self.counts = data["counts"].astype(int)

        if self.counts.ndim != 2:
            raise ValueError("counts must be a two-dimensional array")

        self.n, self.m = self.counts.shape
        self.nbins = self.n * self.m

        print("grid:", self.n, "x", self.m)
        print("number of bins:", self.nbins)
        print("events:", self.counts.sum())
        print("mean true rate:", self.lambda_true.mean())
        print("mean observed count:", self.counts.mean())



    def _prepare_prior_kernel(self):
        self.prior_kernel = ck.precision_matern(
            n=self.n, m=self.m, rho=self.rho, v2=self.var, boundary=self.boundary
        )

        self.prior_mean = np.ones(self.nbins) * self.prior_mean_scalar




    def _prepare_sampler(self):
        self.sampler = pgd.PolyaGammaDensity2D(
            prior_precision=self.prior_kernel,
            prior_mean=self.prior_mean,
            sparse=True,
            lam=self.lam,
            n=self.n,
            m=self.m,
            seed=42,
        )

        self.sampler.set_data(self.counts.ravel(order="C"))




    def image(self, values):
        return self.sampler.scanorder_to_image(values)


    def save_plot(self, plot_name):
        filename = (
            f"synthetic_{TYPE}_{plot_name}.png")

        path = PLOTS_DIR / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {path}")


    def plot_single(self, values, title, plot_name, cmap="viridis",
                    vmin=None, vmax=None, colorbar_label=None):

        plt.figure(figsize=(8, 7))

        im = plt.imshow(
            self.image(values),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        plt.title(title)
        plt.xticks([])
        plt.yticks([])

        if colorbar_label is None:
            plt.colorbar(im)
        else:
            plt.colorbar(im, label=colorbar_label)

        plt.tight_layout()
        self.save_plot(plot_name)




    def plot_data(self):
        self.plot_single(
            self.lambda_true,
            title="True Poisson rate",
            plot_name="true_rate",
            vmin=0,
            vmax=self.lam,
            colorbar_label="Poisson rate",
        )

        self.plot_single(
            self.counts,
            title="Observed counts",
            plot_name="observed_counts",
            colorbar_label="Count",
        )

        self.plot_single(
            self.f_true,
            title="True latent field",
            plot_name="true_f",
            colorbar_label="f",
        )


 

    def fit_map(self):
        print("\n" + "=" * 70)
        print("COMPUTING MAP")
        print("=" * 70)

        f0 = self.sampler.first_guess_estimator()
        f_map = self.sampler.max_logposterior_estimator(f0=f0, method="TNC", niter=MAP_NITER)
        rate_map = self.sampler.field_from_f(f_map)

        return f_map, rate_map




    def posterior_summary(self, initial_f, n_samples=PG_N_ITER, burn_in=PG_BURN_IN,
                          thin=PG_THIN, n_plot_samples=N_PLOT_SAMPLES,
                          random_seed=RANDOM_SEED + 20_000):

        f_mean = np.zeros_like(initial_f)
        f_M2 = np.zeros_like(initial_f)
        rate_mean = np.zeros_like(initial_f)
        rate_M2 = np.zeros_like(initial_f)

        count = 0
        samples_to_plot = []

        n_kept = max(0, (n_samples - burn_in + thin - 1) // thin)
        plot_every = max(1, n_kept // n_plot_samples)

        print("\n" + "=" * 70)
        print("PÓLYA-GAMMA POSTERIOR")
        print("=" * 70)

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

            # Latent field moments
            delta = res - f_mean
            f_mean += delta / count
            f_M2 += delta * (res - f_mean)

            # Rate moments
            rate = self.sampler.field_from_f(res)
            delta = rate - rate_mean
            rate_mean += delta / count
            rate_M2 += delta * (rate - rate_mean)

        if count < 2:
            raise ValueError("Not enough posterior samples.")

        f_sd = np.sqrt(f_M2 / (count - 1))
        rate_sd = np.sqrt(rate_M2 / (count - 1))

        print("retained samples:", count)

        return f_mean, f_sd, rate_mean, rate_sd, samples_to_plot



    def plot_posterior_samples(self, samples):
        plt.figure(figsize=(15, 8))
        plt.suptitle("Pólya-Gamma posterior samples of rate")

        for i, sample in enumerate(samples[:6]):
            plt.subplot(2, 3, i + 1)

            rate = self.sampler.field_from_f(sample)

            plt.imshow(
                self.image(rate),
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=self.lam,
            )

            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        self.save_plot("posterior_samples")



    def plot_results(self, f_map, rate_map, f_pg, f_pg_sd, rate_pg, rate_pg_sd):
        true_f = self.f_true.ravel(order="C")
        true_rate = self.lambda_true.ravel(order="C")



        f_vmin = min(true_f.min(), f_map.min(), f_pg.min())
        f_vmax = max(true_f.max(), f_map.max(), f_pg.max())

        rate_vmin = 0
        rate_vmax = max(true_rate.max(), rate_map.max(), rate_pg.max())


   

        self.plot_single(
            rate_map,
            title="MAP estimate of rate",
            plot_name="map_rate",
            vmin=rate_vmin,
            vmax=rate_vmax,
            colorbar_label="Poisson rate",
        )




        self.plot_single(
            rate_pg,
            title="Pólya-Gamma posterior mean of rate",
            plot_name="posterior_mean_rate",
            vmin=rate_vmin,
            vmax=rate_vmax,
            colorbar_label="Poisson rate",
        )


   

        self.plot_single(
            f_map,
            title="MAP estimate of f",
            plot_name="map_f",
            vmin=f_vmin,
            vmax=f_vmax,
            colorbar_label="f",
        )




        self.plot_single(
            f_pg,
            title="Pólya-Gamma posterior mean of f",
            plot_name="posterior_mean_f",
            vmin=f_vmin,
            vmax=f_vmax,
            colorbar_label="f",
        )



        self.plot_single(
            rate_pg_sd,
            title="Pólya-Gamma posterior standard deviation of rate",
            plot_name="posterior_sd_rate",
            vmin=0,
            colorbar_label="Posterior SD",
        )




        self.plot_single(
            f_pg_sd,
            title="Pólya-Gamma posterior standard deviation of f",
            plot_name="posterior_sd_f",
            vmin=0,
            colorbar_label="Posterior SD",
        )




        map_diff = rate_map - true_rate
        pg_diff = rate_pg - true_rate

        error_absmax = max(np.abs(map_diff).max(), np.abs(pg_diff).max())

        self.plot_single(
            map_diff,
            title="MAP - truth",
            plot_name="map_minus_truth",
            cmap="RdBu_r",
            vmin=-error_absmax,
            vmax=error_absmax,
            colorbar_label="Rate difference",
        )




        self.plot_single(
            pg_diff,
            title="Posterior mean - truth",
            plot_name="posterior_mean_minus_truth",
            cmap="RdBu_r",
            vmin=-error_absmax,
            vmax=error_absmax,
            colorbar_label="Rate difference",
        )



        pg_map_diff = rate_pg - rate_map
        pg_map_absmax = np.abs(pg_map_diff).max()

        self.plot_single(
            pg_map_diff,
            title="Posterior mean - MAP",
            plot_name="posterior_mean_minus_map",
            cmap="RdBu_r",
            vmin=-pg_map_absmax,
            vmax=pg_map_absmax,
            colorbar_label="Rate difference",
        )




    def print_metrics(self, f_map, rate_map, f_pg, f_pg_sd, rate_pg, rate_pg_sd):
        true_f = self.f_true.ravel(order="C")
        true_rate = self.lambda_true.ravel(order="C")
        counts = self.counts.ravel(order="C")

        print("\n" + "=" * 70)
        print("SYNTHETIC MODEL DIAGNOSTICS")
        print("=" * 70)

        print("\nModel parameters")
        print("rho:", self.rho)
        print("lam:", self.lam)
        print("prior variance:", self.var)
        print("prior mean:", self.prior_mean_scalar)

        print("\nRate RMSE")
        print("raw counts:", rmse(counts, true_rate))
        print("MAP:", rmse(rate_map, true_rate))
        print("PG posterior mean:", rmse(rate_pg, true_rate))

        print("\nLatent field RMSE")
        print("MAP:", rmse(f_map, true_f))
        print("PG posterior mean:", rmse(f_pg, true_f))

        print("\nPosterior uncertainty")
        print("mean PG f SD:", f_pg_sd.mean())
        print("mean PG rate SD:", rate_pg_sd.mean())




if __name__ == "__main__":

    synthetic = SyntheticData(
        truth_file=TRUTH_FILE,
        rho=RHO,
        lam=LAM,
        var=PRIOR_VARIANCE,
        prior_mean=PRIOR_MEAN,
        prior_target="truth",
        boundary=BOUNDARY,
    )



    synthetic.plot_data()



    f_map, rate_map = synthetic.fit_map()



    f_pg, f_pg_sd, rate_pg, rate_pg_sd, samples_to_plot = synthetic.posterior_summary(
        initial_f=f_map,
        n_samples=PG_N_ITER,
        burn_in=PG_BURN_IN,
        thin=PG_THIN,
    )




    synthetic.plot_posterior_samples(samples_to_plot)




    synthetic.plot_results(
        f_map=f_map,
        rate_map=rate_map,
        f_pg=f_pg,
        f_pg_sd=f_pg_sd,
        rate_pg=rate_pg,
        rate_pg_sd=rate_pg_sd,
    )



    synthetic.print_metrics(
        f_map=f_map,
        rate_map=rate_map,
        f_pg=f_pg,
        f_pg_sd=f_pg_sd,
        rate_pg=rate_pg,
        rate_pg_sd=rate_pg_sd,
    )

    plt.show()