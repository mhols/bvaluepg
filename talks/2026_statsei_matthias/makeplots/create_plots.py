"""
Simple South Italy earthquake experiment with fixed lambda.

Model
-----
    n_i | f_i ~ Poisson(lambda * sigmoid(f_i))
    f ~ N(m, Q^{-1})

This is intentionally a small real-data example:
- lambda is fixed;
- one MCMC chain;
- no global-shift move;
- no multiple-chain diagnostics;
- no sensitivity or posterior-predictive suite.

The catalogue selection, declustering, binning, coordinates, and coastlines are
taken from source/catalog/catalog.py.
"""
from __future__ import annotations
import sys
import os
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'experiments'))


from contextlib import redirect_stdout
import sys
import warnings
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.special import expit
from polyagammapoisson.catalog.catalog import AValueAnalyser, BValueAnalyser


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

#import polyagammapoisson.catalog.catalog as catalog
import polyagammapoisson.covariance_kernels as ck
import polyagammapoisson.polyagammadensity as pgd
import italy_data as it


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
LAMBDA_PER_SURFACE = 10/25 ## peek intensity per pixel
PICKLE = Path(__file__).resolve().parent / 'tmp.pickle'
BIN_SIZE_KM = 5
BIN_SIZE_KM_SMALL = 2 
RHO_KM = 50
PRIOR_VARIANCE = 4
PRIOR_VARIANCE_PER_SURFACE = 4/25
PRIOR_AVALUE = -2   ### will simply set prior_f

B_BIN_SIZE_KM = 5 
B_BIN_SIZE_KM_SMALL = 2
B_RHO_KM = 50
B_PRIOR_VARIANCE = 0.01 
B_PRIOR_BVALUE= 1

MAG_MIN = 2.5
DECLUSTER_ETA0 = -6.2


BOUNDARY = "symmetric" #"zero"

#COVARCLASS=ck.precision_matern_9pt,
PGCLASS=pgd.PolyaGammaDensity2D
COVARIANCECLASS=ck.spatial_covariance_matern_2_3
PRECISIONCLASS=ck.precision_matern
SPARSE=False
MAGBINSIZE=0.1
MAP_NITER = 3000
N_ITER = 10
BURN_IN = 3
THIN = 1
RANDOM_SEED = 101
N_POSTERIOR_SAMPLES_TO_PLOT = 4
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "south_italy_simple_fixed_lambda"
FIGURES_DIR = REPO_ROOT / "experiments" / "figures" / "south_italy_simple_fixed_lambda"
SAMPLER_LOG = REPO_ROOT / "experiments" / "results" / "south_italy_simple_fixed_lambda" / "sampler.log"

LOAD_PICKLE = False

CATALOG = it.StrettoDiMessina(
    catalog=it.CATALOGS["INGV"],
    f_eta_0=DECLUSTER_ETA0
)

CATALOG2 = copy.deepcopy(CATALOG)
CATALOG3 = copy.deepcopy(CATALOG)
CATALOG4 = copy.deepcopy(CATALOG)



if LOAD_PICKLE:
    with open(PICKLE, 'rb') as f:
        AAS = pickle.load(f)
        A, AS = AAS
else:
    A = AValueAnalyser( 
            catalogclass=it.StrettoDiMessina,
            catalog=it.CATALOGS["INGV"],
            f_eta_0=DECLUSTER_ETA0,
            BINSIZE=BIN_SIZE_KM, 
            PGCLASS=PGCLASS,
            PRECISIONCLASS=PRECISIONCLASS,
            #COVARIANZCLASS=COVARIANZCLASS,
            prior_avalue = PRIOR_AVALUE, 
            sparse=SPARSE,
            boundary=BOUNDARY,
            v2=PRIOR_VARIANCE,
            v2ps=PRIOR_VARIANCE_PER_SURFACE, 
            rhokm=RHO_KM, 
            lamps=LAMBDA_PER_SURFACE,
            Mc=MAG_MIN,
        )
    AS = AValueAnalyser( 
            catalogclass=it.StrettoDiMessina,
            catalog=it.CATALOGS["INGV"],
            f_eta_0=DECLUSTER_ETA0,
            BINSIZE=BIN_SIZE_KM_SMALL, 
            PGCLASS=PGCLASS,
            PRECISIONCLASS=PRECISIONCLASS,
            #COVARIANZCLASS=COVARIANZCLASS,
            prior_avalue = PRIOR_AVALUE, 
            sparse=SPARSE,
            boundary=BOUNDARY,
            v2ps=PRIOR_VARIANCE_PER_SURFACE,
            v2=PRIOR_VARIANCE, 
            rhokm=RHO_KM, 
            lamps=LAMBDA_PER_SURFACE,
            Mc=MAG_MIN,
        )
    with open(PICKLE, 'wb') as f:
        pickle.dump((A, AS ), f)
B = BValueAnalyser( 
        catalogclass=it.StrettoDiMessina,
        catalog=it.CATALOGS["INGV"],
        f_eta_0=None,
        BINSIZE=B_BIN_SIZE_KM, 
        COVARIANCECLASS=COVARIANCECLASS,
        prior_bvalue = B_PRIOR_BVALUE, 
        sparse=SPARSE,
        boundary=BOUNDARY,
        v2=B_PRIOR_VARIANCE,
        rhokm=B_RHO_KM, 
        Mc=MAG_MIN,
        magbinsize=0.1,
    )


BS = BValueAnalyser( 
        catalogclass=it.StrettoDiMessina,
        catalog=it.CATALOGS["INGV"],
        f_eta_0=None,
        BINSIZE=B_BIN_SIZE_KM_SMALL, 
        COVARIANCECLASS=COVARIANCECLASS,
        prior_bvalue = B_PRIOR_BVALUE, 
        sparse=SPARSE,
        boundary=BOUNDARY,
        v2=B_PRIOR_VARIANCE,
        rhokm=B_RHO_KM, 
        Mc=MAG_MIN,
        magbinsize=MAGBINSIZE
    )

def fig1_0():
    import scipy.sparse.linalg as sparse_linalg

    for AAA, title in zip ( (A, AS), ('A', 'AS')):
        m, n = AAA.calc.m, AAA.calc.n
        e_ref=np.zeros(AAA.calc.prior_precision.shape[0])
        e_ref[(n//2)*m + m//2] = 1


        oneline = sparse_linalg.spsolve(AAA.calc.prior_precision, e_ref)
        plt.figure()
        plt.title(title)

        AAA.imshow(oneline)
        plt.figure()
        plt.title(title)
        AAA.imshow(e_ref)
        plt.figure()
        plt.title(title)
        i = np.argmax(np.abs(oneline))
        jj = i // m
        ii = i % n
        oneline = AAA.calc.scanorder_to_image(oneline)
        print(oneline.shape)
        mm = np.arange(m)
        x = (mm-ii) * AAA.BINSIZE
        plt.plot(x, oneline[jj,:])

        


def fig1():
    """
    a value plot
    """

    plt.figure()
    plt.title('historgram of bins')
    #A.lam = LAMBDA_PER_SURFACE * BIN_SIZE_KM**2 
    nobs= A.calc.nobs.ravel()
    #nobs = nobs[nobs>0].5
    #nobs = Acount['counts'].ravel()
    #A.prior_avalue = PRIOR_AVALUE * np.ones(nobs.shape)
    res = []
    n = np.arange(16)
    for i in n:
        res.append(sum(nobs==i))

    #plt.plot(n[1:], res[1:]/sum(res), 'o', label='counts')
    bins = np.arange(16)-0.5
    plt.hist(nobs, bins=bins, label='observd count')
    #plt.yscale('log')

    p = A.calc.prior_n_under_gaussian(pm=A.f_from_a(PRIOR_AVALUE), pv2=PRIOR_VARIANCE, n=n)

    plt.plot(n, p*np.sum(nobs), 'ro')
    plt.plot(n, p*np.sum(nobs), 'r-', label='prior dist')
    plt.legend()
    plt.xticks(np.arange(0,16))
    plt.xlim(-0.5,16)
    #plt.ylim(0,250)
    plt.yscale('log')
    plt.savefig('fig1_A_nobs.png',bbox_inches='tight' )

def fig1_1():
    plt.figure()
    plt.title('prior distribution of binned a values')
    a = np.linspace(-3, 2, 1000)
    plt.plot(a, A.density_under_gaussian(a, -3, 2)*200, label="prior a")
    nobs = A.calc.nobs.ravel()
    #nobs = Acount['counts'].ravel()
    nobs=nobs[nobs>0]
    plt.hist(np.log(nobs/A.BINSIZE**2)/np.log(10), bins=np.arange(-3,0,0.2), 
             label='a per pixel')
    plt.legend()
    plt.savefig('fig1_1_hista.png',bbox_inches='tight' )

def fig2():
    plt.figure()
    plt.title('random prior')
    n = np.arange(20)
    p = A.calc.prior_n_under_gaussian(pm=A.f_from_a(PRIOR_AVALUE), pv2=PRIOR_VARIANCE, n=n)
    plt.plot(n[1:], p[1:], 'o')
    plt.xticks(n)

def fig2_1():
    plt.figure()
    plt.title('binned catalog 5 x 5 km grid')
    nobs = A.calc.nobs.ravel()
    I = nobs>0
    A.imshow( np.log((nobs+1) / A.BINSIZE**2)/np.log(10) , xylimext=A.binning_count, 
             alpha=0.3, cmap='jet')
    #plt.colorbar()
    A.plot_coastlines(color='black', linewidth=2)
    A.plot_event_locations(c='black', s=4)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('fig2_1_binned_data.pdf',bbox_inches='tight' )

def fig2_11():
    plt.figure()
    plt.title('binned catalog 2 x 2 km grid')
    nobs = AS.calc.nobs.ravel()
    I = nobs>0
    AS.imshow( np.log((nobs+1) / AS.BINSIZE**2)/np.log(10) , xylimext=AS.binning_count, 
             alpha=0.3, cmap='jet')
    #plt.colorbar()
    AS.plot_coastlines(color='black', linewidth=2)
    AS.plot_event_locations(c='black', s=4)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('fig2_11_binned_data.pdf',bbox_inches='tight' )


def fig2_3():
    PRIOR_VARIANCE=5
    n=2
    prior_mean=2
    calc = pgd.ExponentialDensity2D()
    # Laplace vs exact
    plt.figure()
    lam = np.linspace(0, 10, 10000)[1:]
    f = calc.f_from_field(lam)
    pr = calc.prior_single_observation_field(lam, 
        mu = prior_mean,
        #A.f_from_a(PRIOR_AVALUE), 
        gamma2=PRIOR_VARIANCE)
    plt.plot([0]+list(lam), [0]+list(pr), label='prior')

    pf = calc.posterior_single_observation_field(
        lam,
        prior_mean,
        #A.f_from_a(PRIOR_AVALUE), 
        PRIOR_VARIANCE, n)

    plt.plot(lam, pf, label='exact posterior', color='black', linewidth=2)

    #lh = calc.log_likelihood_single_observation_field(lam, n=n)
    #plt.plot(lam, pgd.normalize_log(lh))


    def laplace(f, pm, pv, n):
        # n*f - exp( f  -mu + mu) -(f-mu)^2/(2 pv)
        # n*f - e^mu (1 + (f-mu)) - (f-mu)^2 / (2 pv)
        # n*f - e^mu f
        pl = calc.posterior_single_obeservation_f(f, pm, pv, n)
        i = np.argmax(pl)
        mest = f[i]

        a = -n + np.exp(pm) - pm*np.exp(pm)
        b = np.exp(pm) + pv

        ppml = - a/b #    -pv*n + pv*np.exp(mest) - mest
        ppm = mest
        ppv = 1/(np.exp(ppm) + 1/pv)
        ppvl = 1/(np.exp(ppml) + 1/pv)

        return ppm, ppv, ppml, ppvl 

    ppm, ppv, ppml, ppvl  = laplace(f,  
            pm = prior_mean,
            #A.f_from_a(PRIOR_AVALUE), 
            pv=PRIOR_VARIANCE, n=n)

    print(ppm, ppv, ppml, ppvl)

    pl = calc.prior_single_observation_field(lam, ppm, ppv)    

    plt.plot(lam, pl, label='Laplace approximation')

    pl = calc.prior_single_observation_field(lam, ppml, ppvl)    

    plt.plot(lam, pl, label='Laplace approximation (second order)')

    plt.legend()

    plt.xlabel('Poisson intensity')
    plt.yticks([])
    plt.xlim(-0.5, 6)

    plt.savefig('laplaceapprox.pdf',bbox_inches='tight' )

def fig2_4():
    lam=10
    pm = -1
    pv = 0.5

    lams = np.linspace(0, lam, 1000, endpoint=False)
    f = np.linspace(-4*np.sqrt(pv), 4*np.sqrt(pv), 1000)
    calc = pgd.PolyaGammaDensity2D(lam=lam)

    prior_f = calc.prior_single_observation_f(f, pm, pv)
    prior_f /= np.max(prior_f)

    prior2_f = calc.prior_single_observation_f(f, -pm, pv)
    prior2_f /= np.max(prior2_f)

    fig = plt.figure(figsize=(14,5))
    axes = fig.subplots(1,2)
    axes[0].plot(f, calc.field_from_f(f), color='black', label='sigmoid link')
    axes[0].plot(f, lam*0.8*prior_f, color='red')
    axes[0].plot(f, lam*0.8*prior2_f, color='blue')
    axes[0].set_yticks([])
    axes[0].legend()
    axes[0].set_title('prior in f')

    axes[1].plot(lams, calc.prior_single_observation_field(lams, pm, pv), color='red')
    axes[1].plot(lams, calc.prior_single_observation_field(lams, -pm, pv), color='blue')
    axes[1].set_yticks([])
    axes[1].set_title('prior in poisson intensity')

    plt.savefig('./priorillustration.png',bbox_inches='tight' )






# def fig3():

#     fig, axes = plt.subplots(2,2)
#     plt.suptitle('random prior')
#     for i in range(4):
#         plt.subplot(2,2,i+1)
#         A.imshow(A.a_from_f(A.calc.random_prior_parameters()), xylimext=A.binning_count, 
#                  vmin=-2.6, vmax=0, cmap='jet')
#         A.plot_coastlines(color='white', linewidth=2)
    
#     fig.colorbar()
#     plt.savefig('fig3_random_prior.png')

def fig3():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Random prior')

    for ax in axes.flat:
        plt.sca(ax)
        im = A.imshow(
            A.a_from_f(A.calc.random_prior_parameters()),
            xylimext=A.binning_count,
            vmin=-2.5,
            vmax=-0.5,
            cmap='jet'
        )
        A.plot_coastlines(color='white', linewidth=2)

        ax.set_xticks([])
        ax.set_yticks([])

    # Reserve space on the right for the colorbar
    fig.subplots_adjust(right=0.87, wspace=0.15, hspace=0.15)

    # Add one shared colorbar
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, cmap='jet')

    cbar.set_ticks(np.linspace(0, 1, 5))
    cbar.set_ticklabels(['-2.5', '-2.0', '-1.5', '-1.0', '-0.5'])
    cbar.ax.tick_params(labelsize=10)

    fig.savefig('fig3_random_prior.png', dpi=300, bbox_inches='tight')

def fig3_1():

    fig=plt.figure()
    fig.subplots(2,2)
    plt.title('random prior with no observations')
    nobs = A.calc.nobs.ravel()
    A.calc.nobs[:] = 0
    for i in range(4):
        plt.subplot(2,2,i+1)
        A.imshow(A.a_from_f(A.calc.random_prior_parameters()), 
                 xylimext=A.binning_count, vmin=-5, vmax=0)
        plt.colorbar()
        A.plot_coastlines(color='white', linewidth=2)




def fig4():

    for AAA, title in zip( (A, AS), ('5x5', '2x2')):
    #for AAA, title in zip( (A, ), ('grid 10',)):
        plt.figure()
        #plt.title(f'a-value stretto di messina 2015-2025 {title}')

        #compute a-value
        a = AAA.a_from_f(AAA.max_logposterior_estimator())
        AAA.imshow(a, xylimext=A.binning_count,  cmap='jet', vmin=-2.5, vmax=-0.5)
        plt.colorbar()
        AAA.plot_coastlines(color='white', linewidth=4)
        x, y = AAA.catalog.xy
        x += np.random.uniform(-AAA.BINSIZE/4, AAA.BINSIZE, size=len(x))
        y += np.random.uniform(-AAA.BINSIZE/4, AAA.BINSIZE, size=len(x))
        plt.scatter( x, y, c='black', s=2)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'fig4_mag_logposterior_{title}.png',bbox_inches='tight' )

def fig5():

    gamma = 10

    N = 10 #gamma * BBB.BINSIZE**2
    for BBB, title in zip((B, BS), ('large', 'small')):
    #for BBB, title in zip((B, BS), ('large', 'small')):


        plt.figure()
        plt.title('b-value stretto di messina 2015-2025')

        niter = 10
        burn_in = 0
        mean = 0
        var  = 0 

        for f in BBB.sample_posterior(n_iter=niter+burn_in, burn_in=burn_in):
            f = BBB.b_from_f(f)
            mean += f
            var += f**2

        mean /= niter
        var = var/niter - mean**2 
        #compute b-value
        b = mean
        bb = np.ones(BBB.calc.nbins)
        bb[BBB.I] = b


        BBB.imshow( bb.T.ravel(), #BBB.b_from_beta(nobs / magsum),  
               vmin=0.9, 
               vmax=1.5,
               xylimext = BBB.binning_count,  
               cmap='jet')
        #plt.xlim(BBB.extent[0], BBB.extent[1])

        BBB.plot_coastlines(color='white', linewidth=4)
        BBB.plot_event_locations(c='black', s=2)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig(f'b_value_{title}.png', bbox_inches='tight')

        """
        plt.figure()
        plt.title('counts stretto di messina 2015-2025')

        #compute b-value
        BBB.calc.imshow(BBB.calc.nobs,
               origin='lower', 
               extent=BBB.extent, cmap='jet')
        #plt.xlim(BBB.extent[0], BBB.extent[1])

        BBB.plot_coastlines(color='white', linewidth=4)
        BBB.plot_event_locations(color='black', marker='o', markersize=2)
        plt.colorbar()
        
        plt.figure()
        plt.title('bare-b-value stretto di messina 2015-2025')

        #compute b-value
        n, m = BBB.calc.n, BBB.calc.m

        BBB.calc.imshow(np.where(BBB.calc.nobs > 0, BBB.calc.weight/BBB.calc.nobs, 0),
               origin='lower', 
               aspect='equal',
               extent=BBB.extent, cmap='jet')
        #plt.xlim(BBB.extent[0], BBB.extent[1])

        BBB.plot_coastlines(color='white', linewidth=4)
        BBB.plot_event_locations(color='black', marker='o', markersize=2)
        plt.colorbar()
        """
def fig5_1():

    for AAA, title in zip( (A, AS), ('large', 'small')):
    #for AAA, title in zip( (A, ), ('large', )):
        f = AAA.max_logposterior_estimator()

        res = []
        i = 1
        mean = 0
        niter = 200
        #tracemalloc.start()
        #snap1=tracemalloc.take_snapshot()
        for ff in AAA.sample_posterior(burn_in=50, n_iter=niter+50):
            res.append(ff)
            mean += ff
        #snap2=tracemalloc.take_snapshot()
        #tracemalloc.stop()

        #for stat in snap2.compare_to(snap1, 'lineno')[:10]:
        #    print(stat)

    #sys.exit(0)

        mean /= niter
        #print(mean.shape)

        plt.figure()
        for i in range(4):
            plt.subplot(2,2,i+1)
            AAA.imshow(A.a_from_f(res[50+i*(niter//4)-1]), 
                    xylimext=A.binning_count, vmin=-2.5, vmax=-0.5, cmap='jet')
            #AAA.plot_event_locations(c='black', s=2)
            AAA.plot_coastlines(color='white', linewidth=2)
            plt.xticks([])
            plt.yticks([])
        plt.savefig(f'fig5_1_rp_{title}.png', bbox_inches='tight')

        plt.figure()

def fig5_2():

    for AAA, title in zip( (A, AS), ('large', 'small')):
    #for AAA, title in zip( (A, ), ('large', )):
        f = AAA.max_logposterior_estimator()
        plt.figure()
        AAA.imshow(AAA.a_from_f(f), 
                    xylimext=AAA.catalog.binning_count, vmin=-2.5, vmax=-0.5, cmap='jet')
        plt.colorbar()
        AAA.plot_event_locations(c='black', s=1)
        AAA.plot_coastlines(color='white', linewidth=2)
        plt.xticks([])
        plt.yticks([])

        plt.savefig(f'fig5_2_map_{title}.png', bbox_inches='tight')


        i = 1
        mean = 0
        sums = 0
        meanf = 0
        niter = 2000
        #tracemalloc.start()
        #snap1=tracemalloc.take_snapshot()
        for ff in AAA.sample_posterior(burn_in=50, n_iter=niter+50):
            fff = AAA.a_from_f(ff)
            meanf += AAA.calc.field_from_f(ff)
            mean += fff
            sums += fff**2
        #snap2=tracemalloc.take_snapshot()
        #tracemalloc.stop()

        #for stat in snap2.compare_to(snap1, 'lineno')[:10]:
        #    print(stat)

        #sys.exit(0)

        mean /= niter
        meanf /= niter

        var = sums/niter - mean**2

        #print(mean.shape)

        plt.figure()
        AAA.imshow(mean, 
                    xylimext=AAA.catalog.binning_count, vmin=-2.5, vmax=-0.5, cmap='jet')
        plt.colorbar()
        AAA.plot_event_locations(c='black', s=1)
        AAA.plot_coastlines(color='white', linewidth=2)
        plt.xticks([])
        plt.yticks([])

        plt.savefig(f'fig5_2_rm_{title}.png', bbox_inches='tight')

        plt.figure()
        AAA.imshow(np.sqrt(var), 
                    xylimext=AAA.catalog.binning_count,  cmap='jet',
                    vmin=0, vmax=0.3)
        plt.colorbar()
        AAA.plot_event_locations(c='red', s=1)
        AAA.plot_coastlines(color='white', linewidth=2)
        plt.xticks([])
        plt.yticks([])

        plt.savefig(f'fig5_2_std_{title}.png', bbox_inches='tight')


        plt.figure()
        norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=5)
        cmap = plt.get_cmap('RdBu_r')

        vmin, vmax = -10, 10
        # Position of 0 on a linear 0..1 scale
        zero_pos = (0 - vmin) / (vmax - vmin)   # = 10/11 ≈ 0.909

        # Build a colormap: dark red (negative) -> white at zero_pos -> dark blue (positive)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'centered_at_zero',
            [(0.0, "#0A06F9"),      # most negative
            (zero_pos, '#ffffff'), # zero, near the top
            (1.0, "#F83606")],     # most positive
        )

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # plain linear norm

        AAA.imshow(100*(AAA.calc.field_from_f(f)/meanf-1), 
                   xylimext=AAA.catalog.binning_count, #cmap='jet')
                   cmap=cmap, 
                   norm=norm)
        AAA.plot_event_locations(c='black', s=1)
        AAA.plot_coastlines(color='black', linewidth=2)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('local rate (MAP - PMEAN) / PMEAN) in %')
        
        plt.savefig(f"diff_{title}.png", bbox_inches='tight')


   



def fig6():

    nobs = B.calc.scanorder_to_image(B.calc.nobs)
    nobs[:,:] = 0
    nobs[10,20] = 10

    magsum = B.calc.scanorder_to_image(B.binning_mag['counts'])
    magsum[:,:] = 0
    magsum[10,20] = np.sum(np.log(1/np.random.uniform(size=10))+2.5)

    B.calc.set_obs(nobs)
    B.calc.weight = magsum



"""


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------
# Small plotting utilities
# ---------------------------------------------------------------------

def save_figure(fig, name):
    pdf = FIGURES_DIR / f"{name}.pdf"
    png = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {pdf}")


def add_coastlines(ax, cat):
    for poly in cat.coastlines:
        lon, lat = poly[:, 0], poly[:, 1]
        x, y = cat.coordinates.lonlat_to_rotated_xy(lon, lat)
        ax.plot(x, y, color="white", linewidth=0.65, zorder=3)

    xmin, xmax, ymin, ymax = cat.extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")


def vector_to_image(x, nbinx, nbiny):
    return np.asarray(x).reshape(nbinx, nbiny, order="C").T


def plot_map(ax, x, cat, nbinx, nbiny, title, cbar_label, cmap="viridis", norm=None):
    im = ax.imshow(
        vector_to_image(x, nbinx, nbiny),
        extent=cat.extent,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        zorder=1,
    )
    add_coastlines(ax, cat)
    ax.set_title(title)
    ax.set_xlabel(r"$x$ [km]")
    ax.set_ylabel(r"$y$ [km]")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)




def prepare_catalogue():
    # SicilyCalabria already applies the fixed rotated-coordinate spatial cut.
    cat = italy_data.SicilyCalabria(catalog=italy_data.CATALOGS["INGV"], BINSIZE=BIN_SIZE_KM)
    n_before = len(cat.catDataFrame)

    # Keep non-child events according to the NND declustering rule.
    cat.filter_decluster(Mc=MAG_MIN, f_eta_0=DECLUSTER_ETA0)
    n_after = len(cat.catDataFrame)

    binned = cat.binning_count
    counts = np.asarray(binned["counts"], dtype=int)
    nbinx = int(binned["nbinx"])
    nbiny = int(binned["nbiny"])

    if counts.shape != (nbinx, nbiny):
        raise ValueError(
            f"Count-grid shape {counts.shape} does not match ({nbinx}, {nbiny})."
        )

    return cat, counts, nbinx, nbiny, n_before, n_after


def prepare_sampler(counts, nbinx, nbiny):
    prior_precision = ck.precision_matern(
        n=nbinx,
        m=nbiny,
        rho=RHO_KM / BIN_SIZE_KM,
        v2=PRIOR_VARIANCE,
        boundary=BOUNDARY,
    )
    prior_mean = np.full(nbinx * nbiny, PRIOR_MEAN, dtype=float)

    sampler = pgd.PolyaGammaDensity2D(
        prior_precision=prior_precision,
        prior_mean=prior_mean,
        sparse=True,
        lam=LAMBDA,
        n=nbinx,
        m=nbiny,
        seed=RANDOM_SEED,
    )
    sampler.set_nobs(counts.ravel(order="C"))
    return sampler




def compute_map(sampler):
    print(f"\nComputing MAP with fixed lambda={LAMBDA:g}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Method TNC does not use Hessian information")
        warnings.filterwarnings("ignore", message="Unknown solver options")
        f_map = sampler.max_logposterior_estimator(niter=MAP_NITER)

    f_map = np.asarray(f_map, dtype=float)
    rate_map = LAMBDA * expit(f_map)
    return f_map, rate_map


def run_posterior(sampler, initial_f):
    nbins = initial_f.size

    f_mean = np.zeros(nbins)
    f_M2 = np.zeros(nbins)
    rate_mean = np.zeros(nbins)
    rate_M2 = np.zeros(nbins)

    total_rate_trace = []
    posterior_rate_samples = []

    n_expected = len(range(BURN_IN, N_ITER, THIN))
    sample_stride = max(1, n_expected // N_POSTERIOR_SAMPLES_TO_PLOT)

    generator = sampler.sample_posterior(
        n_iter=N_ITER,
        burn_in=BURN_IN,
        thin=THIN,
        initial_f=initial_f,
        random_seed=RANDOM_SEED,
        sample_lam=False,       # lambda stays fixed
        return_lam=False,
    )

    print(
        f"Running one chain: {N_ITER} iterations, {BURN_IN} burn-in, "
        f"lambda fixed at {LAMBDA:g}."
    )

    count = 0
    with open(SAMPLER_LOG, "w") as log_handle, redirect_stdout(log_handle):
        for f in generator:
            f = np.asarray(f, dtype=float)
            rate = LAMBDA * expit(f)
            count += 1

            delta_f = f - f_mean
            f_mean += delta_f / count
            f_M2 += delta_f * (f - f_mean)

            delta_rate = rate - rate_mean
            rate_mean += delta_rate / count
            rate_M2 += delta_rate * (rate - rate_mean)

            total_rate_trace.append(float(np.sum(rate)))

            if (
                count % sample_stride == 0
                and len(posterior_rate_samples) < N_POSTERIOR_SAMPLES_TO_PLOT
            ):
                posterior_rate_samples.append(rate.copy())

    if count < 2:
        raise RuntimeError("Too few retained posterior samples.")

    return {
        "n_kept": count,
        "f_mean": f_mean,
        "f_sd": np.sqrt(f_M2 / (count - 1)),
        "rate_mean": rate_mean,
        "rate_sd": np.sqrt(rate_M2 / (count - 1)),
        "total_rate_trace": np.asarray(total_rate_trace),
        "posterior_rate_samples": posterior_rate_samples,
    }



def make_main_figure(cat, counts, nbinx, nbiny, posterior):
    observed = counts.ravel(order="C")
    rate_mean = posterior["rate_mean"]
    rate_sd = posterior["rate_sd"]

    # Same scale for observed counts and posterior mean.
    intensity_norm = PowerNorm(
        gamma=0.5,
        vmin=0.0,
        vmax=max(float(observed.max()), float(rate_mean.max())),
    )
    sd_norm = PowerNorm(
        gamma=0.7,
        vmin=0.0,
        vmax=float(rate_sd.max()),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)

    plot_map(
        axes[0], observed, cat, nbinx, nbiny,
        "(a) Observed counts", "Count per cell",
        cmap="viridis", norm=intensity_norm,
    )
    plot_map(
        axes[1], rate_mean, cat, nbinx, nbiny,
        "(b) Posterior mean intensity", r"$E[\nu_i\mid n]$",
        cmap="viridis", norm=intensity_norm,
    )
    plot_map(
        axes[2], rate_sd, cat, nbinx, nbiny,
        "(c) Posterior SD", r"$\mathrm{SD}(\nu_i\mid n)$",
        cmap="magma", norm=sd_norm,
    )

    save_figure(fig, "south_italy_simple_main")
    plt.close(fig)


def make_map_comparison(cat, rate_map, posterior, nbinx, nbiny):
    rate_mean = posterior["rate_mean"]
    norm = PowerNorm(
        gamma=0.5,
        vmin=0.0,
        vmax=max(float(rate_map.max()), float(rate_mean.max())),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.7), constrained_layout=True)

    plot_map(
        axes[0], rate_map, cat, nbinx, nbiny,
        "(a) MAP intensity", r"$\nu_i^{\mathrm{MAP}}$",
        norm=norm,
    )
    plot_map(
        axes[1], rate_mean, cat, nbinx, nbiny,
        "(b) Posterior mean intensity", r"$E[\nu_i\mid n]$",
        norm=norm,
    )

    save_figure(fig, "south_italy_simple_map_vs_posterior")
    plt.close(fig)


def make_total_intensity_figure(total_count, posterior):
    total_intensity = np.asarray(
        posterior["total_rate_trace"],
        dtype=float,
    )

    posterior_mean = float(np.mean(total_intensity))
    q025, q975 = np.quantile(
        total_intensity,
        [0.025, 0.975],
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    ax.hist(
        total_intensity,
        bins=35,
        density=True,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
        label="Posterior",
    )

    ax.axvline(
        posterior_mean,
        linewidth=1.6,
        label=rf"Posterior mean = {posterior_mean:.1f}",
    )

    ax.axvline(
        total_count,
        color="0.15",
        linestyle="--",
        linewidth=1.6,
        label=rf"Observed total = {total_count}",
    )

    ax.axvspan(
        q025,
        q975,
        alpha=0.12,
        label=rf"95\% interval [{q025:.0f}, {q975:.0f}]",
    )

    ax.set_xlabel(r"Total intensity $T=\sum_i \nu_i$")
    ax.set_ylabel("Posterior density")
    ax.set_title("Posterior distribution of total intensity")

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "south_italy_simple_total_intensity",
    )
    plt.close(fig)


def make_sample_figure(cat, posterior, nbinx, nbiny):
    samples = posterior["posterior_rate_samples"]
    if not samples:
        return

    norm = PowerNorm(
        gamma=0.5,
        vmin=0.0,
        vmax=max(float(x.max()) for x in samples),
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.7, 7.7), constrained_layout=True)

    for j, (ax, sample) in enumerate(zip(axes.ravel(), samples), start=1):
        plot_map(
            ax, sample, cat, nbinx, nbiny,
            f"Posterior draw {j}", r"$\nu_i$",
            norm=norm,
        )

    save_figure(fig, "south_italy_simple_samples")
    plt.close(fig)


# ---------------------------------------------------------------------
# Numerical output
# ---------------------------------------------------------------------

def save_results(counts, f_map, rate_map, posterior, n_before, n_after):
    np.savez_compressed(
        RESULTS_DIR / "posterior_summary.npz",
        counts=counts,
        f_map=f_map,
        rate_map=rate_map,
        f_mean=posterior["f_mean"],
        f_sd=posterior["f_sd"],
        rate_mean=posterior["rate_mean"],
        rate_sd=posterior["rate_sd"],
        total_rate_trace=posterior["total_rate_trace"],
    )

    trace = posterior["total_rate_trace"]
    with open(RESULTS_DIR / "summary.txt", "w") as handle:
        print("SOUTH ITALY FIXED-LAMBDA EXPERIMENT", file=handle)
        print("=" * 60, file=handle)
        print(f"Events before declustering : {n_before}", file=handle)
        print(f"Events after declustering  : {n_after}", file=handle)
        print(f"Grid shape                 : {counts.shape}", file=handle)
        print(f"Observed total N           : {int(counts.sum())}", file=handle)
        print(f"Fixed lambda               : {LAMBDA:g}", file=handle)
        print(f"rho [km]                   : {RHO_KM:g}", file=handle)
        print(f"Prior variance parameter   : {PRIOR_VARIANCE:g}", file=handle)
        print(f"Prior mean                 : {PRIOR_MEAN:g}", file=handle)
        print(f"MCMC iterations            : {N_ITER}", file=handle)
        print(f"Burn-in                    : {BURN_IN}", file=handle)
        print(f"Retained samples           : {posterior['n_kept']}", file=handle)
        print(f"MAP total intensity        : {rate_map.sum():.6f}", file=handle)
        print(f"Posterior mean total rate  : {trace.mean():.6f}", file=handle)
        print(f"Posterior SD total rate    : {trace.std(ddof=1):.6f}", file=handle)




def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cat, counts, nbinx, nbiny, n_before, n_after = prepare_catalogue()
    total_count = int(counts.sum())

    print("\n" + "=" * 60)
    print("SOUTH ITALY EXPERIMENT")
    print("=" * 60)
    print(f"Events before declustering : {n_before}")
    print(f"Events after declustering  : {n_after}")
    print(f"Grid shape                 : {counts.shape}")
    print(f"Grid cells                 : {counts.size}")
    print(f"Observed total N           : {total_count}")
    print(f"Maximum cell count         : {int(counts.max())}")
    print("-" * 60)
    print(f"Fixed lambda               : {LAMBDA:g}")
    print(f"rho                         : {RHO_KM:g} km")
    print(f"Prior variance parameter   : {PRIOR_VARIANCE:g}")
    print(f"Prior mean                 : {PRIOR_MEAN:g}")
    print("=" * 60)

    sampler = prepare_sampler(counts, nbinx, nbiny)
    f_map, rate_map = compute_map(sampler)
    posterior = run_posterior(sampler, f_map)

    print(f"Retained posterior draws   : {posterior['n_kept']}")
    total_intensity = posterior["total_rate_trace"]
    q025, q50, q975 = np.quantile(
        total_intensity,
        [0.025, 0.5, 0.975],
    )

    print(f"Posterior mean total rate  : {total_intensity.mean():.3f}")
    print(f"Posterior median total rate: {q50:.3f}")
    print(f"95% posterior interval     : [{q025:.3f}, {q975:.3f}]")
    print(f"Observed total count       : {total_count}")
    print(f"Low-level sampler log      : {SAMPLER_LOG}")

    make_main_figure(cat, counts, nbinx, nbiny, posterior)
    make_map_comparison(cat, rate_map, posterior, nbinx, nbiny)
    make_sample_figure(cat, posterior, nbinx, nbiny)
    make_total_intensity_figure(total_count, posterior)
    save_results(counts, f_map, rate_map, posterior, n_before, n_after)

    print("\nSOUTH ITALY experiment completed.")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


"""

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    #fig1_0()
    #fig1()
    #fig2()
    #fig2_1()
    #fig2_3()
    #fig3()
    #fig3_1()
    #fig4()
    fig5()
    #fig5_1()
    #fig5_2()
    plt.show()


