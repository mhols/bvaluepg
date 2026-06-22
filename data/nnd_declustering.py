"""

Declustering HORUS earthquake catalogue using Zaliapin Nearest-neighbour method
 https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.018501

based on https://github.com/tgoebel/clustering-analysis

https://zaliapin.github.io/pubs/Zaliapin_Ben-Zion_JGR20.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import src.clustering as clustering
import os

#------------------------------clustering modules-------------------------------------- 
# EqCat is a Python object that is used for catlog processing
from src.EqCat import EqCat
eqCat = EqCat( )
#for methods check source code or uncomment the following line 
print( 'EqCat Methods: ', eqCat.methods)

# ============================================================
# HELPERS FOR SAVING CSV WITH ORIGINAL HORUS STRUCTURE
# ============================================================

def load_original_horus_with_matching_fields(path):

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    original_columns = list(df.columns)

    # Add helper columns while preserving the original HORUS columns.
    column_map = {
        "Year": "year",
        "Mo": "month",
        "Da": "day",
        "Ho": "hour",
        "Mi": "minute",
        "Se": "second",
        "Lat": "lat",
        "Lon": "lon",
        "Depth": "depth",
        "Mw": "mag_main",
        "Mw/Mpry": "mag_alt",
        "Iside n.": "event_id",
        "ISIDe nr.": "event_id",
    }

    for old, new in column_map.items():
        if old in df.columns:
            # If both Iside n. and ISIDe nr. exist, keep the first useful one.
            if new in df.columns:
                df[new] = df[new].fillna(df[old])
            else:
                df[new] = df[old]

    # Combine magnitude columns into helper column "mag".
    if "mag_main" in df.columns and "mag_alt" in df.columns:
        df["mag"] = pd.to_numeric(df["mag_main"], errors="coerce").fillna(
            pd.to_numeric(df["mag_alt"], errors="coerce")
        )
    elif "mag_main" in df.columns:
        df["mag"] = pd.to_numeric(df["mag_main"], errors="coerce")
    elif "mag_alt" in df.columns:
        df["mag"] = pd.to_numeric(df["mag_alt"], errors="coerce")
    else:
        raise ValueError("Could not find HORUS magnitude column: Mw or Mw/Mpry")

    required = ["year", "month", "day", "hour", "minute", "second", "lat", "lon", "depth", "mag"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column needed for matching/filtering: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "event_id" in df.columns:
        df["event_id_num"] = pd.to_numeric(df["event_id"], errors="coerce")
    else:
        df["event_id_num"] = np.nan

    sec = df["second"].fillna(0).to_numpy(float)
    whole_second = np.floor(sec).astype(int)
    microsecond = np.round((sec - whole_second) * 1_000_000).astype(int)

    carry = microsecond >= 1_000_000
    whole_second[carry] += 1
    microsecond[carry] -= 1_000_000

    df["datetime"] = pd.to_datetime(
        {
            "year": df["year"].astype("Int64"),
            "month": df["month"].astype("Int64"),
            "day": df["day"].astype("Int64"),
            "hour": df["hour"].fillna(0).astype(int),
            "minute": df["minute"].fillna(0).astype(int),
            "second": whole_second,
            "microsecond": microsecond,
        },
        errors="coerce",
    )

    df = df.dropna(subset=["datetime", "lat", "lon", "mag"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    year_start = pd.to_datetime(df["datetime"].dt.year.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((df["datetime"].dt.year + 1).astype(str) + "-01-01")
    df["decimal_year"] = (
        df["datetime"].dt.year
        + (df["datetime"] - year_start).dt.total_seconds()
        / (next_year_start - year_start).dt.total_seconds()
    )

    # Reconstruct N exactly as create_horus_mat_for_clust.py did.
    if df["event_id_num"].notna().any():
        event_id = df["event_id_num"].to_numpy(dtype=float, copy=True)
        missing = ~np.isfinite(event_id)
        event_id[missing] = np.arange(1, len(df) + 1, dtype=float)[missing]
    else:
        event_id = np.arange(1, len(df) + 1, dtype=float)

    if len(np.unique(event_id)) != len(event_id):
        event_id = np.arange(1, len(df) + 1, dtype=float)

    df["_N_match"] = event_id

    return df, original_columns


def save_original_structure_csv(original_catalog_file, eqCat, is_keep, Mmin, Mmax, tmin, tmax, out_csv):
    """
    Save selected HORUS events with original columns plus a boolean kept column.
    """
    df, original_columns = load_original_horus_with_matching_fields(original_catalog_file)

    # Apply same filtering as EqCat.
    if Mmin is not None:
        df = df[df["mag"] >= float(Mmin)].copy()
    if Mmax is not None:
        df = df[df["mag"] <= float(Mmax)].copy()
    if tmin is not None:
        df = df[df["decimal_year"] >= float(tmin)].copy()
    if tmax is not None:
        df = df[df["decimal_year"] <= float(tmax)].copy()

    df = df.sort_values("datetime").reset_index(drop=True)

    kept_by_id = dict(zip(eqCat.data["N"].astype(float), is_keep.astype(bool)))
    df["kept"] = df["_N_match"].astype(float).map(kept_by_id)

    # Fallback if there is a tiny ID mismatch but the order and length agree.
    if df["kept"].isna().any():
        if len(df) == len(is_keep):
            print("Warning: some IDs did not match; using selected-catalogue order for kept column.")
            df["kept"] = is_keep.astype(bool)
        else:
            missing = df["kept"].isna().sum()
            raise ValueError(
                f"Could not match {missing} original rows to EqCat events. "
                f"Original selected rows={len(df)}, EqCat rows={len(is_keep)}"
            )

    df["kept"] = df["kept"].astype(bool)

    # Preserve original HORUS structure, with one new column at the end.
    df[original_columns + ["kept"]].to_csv(out_csv, index=False)

    print("save original-structure CSV:", out_csv)
    print("CSV rows:", len(df))
    print("kept=True :", int(df["kept"].sum()))
    print("kept=False:", int((~df["kept"]).sum()))




# ============================================================
# PARAMETERS TO CHANGE
# ============================================================


file_in= 'HORUS_Ita_Catalog.mat'

ORIGINAL_CATALOG_FILE = 'HORUS_Ita_Catalog.txt'

# Filtering
Mmin, Mmax = 3, None    # use None for no magnitude filter
tmin, tmax = 1980, 2012    # use None for no time filter


#=================================2==============================================
#                            load data, select events
#================================================================================
eqCat.loadMatBin( f"{file_in}")
print(  'total no. of events', eqCat.size())
eqCat.selectEvents( Mmin, Mmax, 'Mag')
eqCat.selectEvents( tmin, tmax, 'Time')
print( 'no. of events after Mag/Time selection', eqCat.size())





    

#=================================3==============================================
#                          test plot with Basemap
#================================================================================
b_map = True
projection = 'cyl'
xmin,xmax = eqCat.data['Lon'].min(), eqCat.data['Lon'].max()
ymin,ymax = eqCat.data['Lat'].min(), eqCat.data['Lat'].max()
if b_map:
    # setup equi distance basemap.
    m = Basemap( llcrnrlat  =  ymin,urcrnrlat  =  ymax,
                 llcrnrlon  =  xmin,urcrnrlon  =  xmax,
                 projection = projection,lat_0=(ymin+ymax)*.5,lon_0=(xmin+xmax)*.5,
                 resolution = 'l')
    m.drawstates( linewidth = 1)
    m.drawcoastlines( linewidth= 2)
    a_x, a_y = m( eqCat.data['Lon'], eqCat.data['Lat'])
    m.plot( a_x, a_y, 'ko', ms = 1)
    sel6 = eqCat.data['Mag'] >= 6
    m.plot( a_x[sel6], a_y[sel6], 'ro', ms = 8, mew= 1.5, mfc = 'none')

    m.drawmeridians( np.linspace( int(xmin), xmax, 4),labels=[False,False,False,True],
                     fontsize = 12, fmt = '%.1f')
    m.drawparallels( np.linspace( int(ymin), ymax, 4),labels=[True,False,False,False],
                     fontsize = 12, fmt = '%.2f')

    plt.savefig( file_in.replace( 'mat', 'png'))



# ============================================================
# NEAREST-NEIGHBOR DECLUSTERING
# ============================================================

# set parameters:fractal dimension and b-value
dPar  = {   # fractal dimension and b for eq. (1) in Zaliapin & Ben-Zion
            'D'           : 1.6, # TODO: - these values should be contrained independently
            'b'           : 1.0, # can be estimated  using: https://github.com/tgoebel/magnitude-distribution for b-value
            'Mc'          : Mmin,
            #=================plotting==============
             # these parameters rarely have to be changes
            'eta_binsize' :  .3,
            'xmin' : -13, 'xmax' : 0,
          }


#================================================================================
#                           to cartesian coordinates
#================================================================================
# two ways to do the distance comp: 1 project into equal distance azimuthal , comp Cartersian distance in 3D
#                                   2 get surface distance from lon, lat (haversine), use pythagoras to include depth
if b_map:
    eqCat.toCart_coordinates( projection = 'eqdc')
    print( 'convert to cartesian using equi-distant projection')
#==================================2=============================================
#                       compute space-time-magnitude distance, histogram
#================================================================================
eqCat.data['Z'] = eqCat.data['Depth']
print('depth range: ', eqCat.data['Z'].min(), eqCat.data['Z'].max())
dNND = clustering.NND_eta( eqCat, dPar,  
                              correct_co_located = True, verbose= True)
###histogram
aBins        = np.arange( -13, 1, dPar['eta_binsize'], dtype = float)
aHist, aBins = np.histogram( np.log10( dNND['aNND'][dNND['aNND']>0]), aBins)
aBins = aBins[0:-1] + dPar['eta_binsize']*.5
# correct for binsize
aHist = aHist/dPar['eta_binsize']
# to pdf (prob. density)
aHist /= eqCat.size()
#=================================3==============================================
#                            save results
#================================================================================
import scipy.io
NND_file = '%s_NND_Mc_%.1f.mat'%( file_in.split('.')[0], dPar['Mc'])
print( 'save file', NND_file)
scipy.io.savemat( NND_file, dNND, do_compression  = True)

#=================================4==============================================
#                          plot histogram
#================================================================================
# load eta_0 value - only for plotting purposes
eta_0_file = f"HORUS_Ita_Catalog_Mc_{dPar['Mc']:.1f}_eta_0.mat"
if os.path.isfile( eta_0_file):
    print( 'load eta_0 from file'),
    f_eta_0 = np.loadtxt( eta_0_file, dtype = float)
    print( 'eta_0',f_eta_0)
else:
    f_eta_0 = -4.6
    print( 'could not find eta_0 file', eta_0_file, 'use value: ', f_eta_0)

fig, ax = plt.subplots()
#ax.plot( vBin, vHist, 'ko')
ax.bar( aBins, aHist, width =.8*dPar['eta_binsize'], align = 'edge', color = '.5', label = 'Mc = %.1f'%( dPar['Mc']))
ax.plot( [f_eta_0, f_eta_0], ax.get_ylim(), 'w-',  lw = 2, label = '$N_\mathrm{tot}$=%i'%( eqCat.size()))
ax.plot( [f_eta_0, f_eta_0], ax.get_ylim(), 'r--', lw = 2, label = '$N_\mathrm{cl}$=%i'%( dNND['aNND'][dNND['aNND']<1e-5].shape[0]))

ax.legend( loc = 'upper left')
ax.set_xlabel( 'NND, log$_{10} \eta$')
ax.set_ylabel( 'Number of Events')
ax.grid( 'on')
ax.set_xlim( dPar['xmin'], dPar['xmax'])



# ============================================================
# separate clusters from independent background and compile event families
# ============================================================

dPar['eta_0'] = f_eta_0
print( 'similarity threshold', dPar['eta_0'])

catalog_stem = Path(file_in).stem
clust_file = f"{catalog_stem}_Mc_{dPar['Mc']:.1f}_clusters.mat"
    
dNND['aNND'] = np.log10( dNND['aNND'])
# clustering according to eta_0 similarity criteria
dClust = clustering.compileClust( dNND, f_eta_0, useLargerEvents = False)

# IDs of child events to skip
a_ID_skip = np.unique(dNND['aEqID_c'][dNND['aNND'] < dPar['eta_0']])

# Boolean masks on the selected catalogue
is_skip = np.isin(eqCat.data['N'], a_ID_skip)
is_keep = ~is_skip

print("Kept events   :", is_keep.sum())
print("Skipped events:", is_skip.sum())

# Save CSV with original HORUS structure plus kept column
catalog_stem = Path(file_in).stem
csv_file = f"{catalog_stem}_declustered_Mc_{dPar['Mc']:.1f}_eta_{dPar['eta_0']:.2f}.csv"

save_original_structure_csv(
    ORIGINAL_CATALOG_FILE,
    eqCat,
    is_keep,
    Mmin,
    Mmax,
    tmin,
    tmax,
    csv_file,
)
#=================================4==========================================================================
#                           save results
#============================================================================================================
scipy.io.savemat( os.path.join( clust_file), dClust, do_compression=True)


# ============================================================
#  Diagnostics Plots
# ============================================================

#=======event-pair density in r-T============================
catChild=  EqCat()
catParent= EqCat()
catChild.copy(  eqCat)
catParent.copy( eqCat)
catChild.selEventsFromID(    dNND['aEqID_c'], repeats = True)
catParent.selEventsFromID(   dNND['aEqID_p'], repeats = True)
print( 'size of offspring catalog', catChild.size(), 'size of parent cat', catParent.size())  

#compute re-scaled interevent times and distances
a_R, a_T = clustering.rescaled_t_r(catChild, catParent, dPar)
# plot event pair density 
fig = clustering.plot_R_T( a_T, a_R, f_eta_0)



#========================== spanning tree======================================================

plt.figure( 1)
ax = plt.subplot(111)  
for iEv in range( catParent.size()):
    print( f"MS-ID, {int(catParent.data['N'][iEv]):d}, t-Par: {catParent.data['Time'][iEv]:.5f},'t-child', {eqCat.data['Time'][iEv]:.5f}", end= "\r")

    if dNND['aNND'][iEv] < dPar['eta_0']:#triggered cluster
        ax.plot( [catParent.data['Time'][iEv]], [catParent.data['Lat'][iEv]], 'ro', ms = 12, alpha = .2)
        ax.plot( [catParent.data['Time'][iEv],catChild.data['Time'][iEv]],
                  [catParent.data['Lat'][iEv], catChild.data['Lat'][iEv]], 'k-', marker = 'o', ms = 4, mew =1, mfc = 'none')
    else: # independent events
        ax.plot( [catChild.data['Time'][iEv]], [catChild.data['Lat'][iEv]], 'bo', ms = 5, alpha = .6)

eqCat = EqCat( )
#=================================1==============================================
#                            load data, select events
#================================================================================
eqCat.loadMatBin( f"{file_in}")

eqCat.selectEvents( Mmin, Mmax, 'Mag')
eqCat.selectEvents( tmin, tmax, 'Time')
N_tot = eqCat.size()
print(  'total no. of events', N_tot)


fig, ax = plt.subplots(figsize=(9, 8))

m = Basemap(
    llcrnrlat=ymin, urcrnrlat=ymax,
    llcrnrlon=xmin, urcrnrlon=xmax,
    projection=projection,
    lat_0=(ymin + ymax) * 0.5,
    lon_0=(xmin + xmax) * 0.5,
    resolution='l',
    ax=ax
)

m.drawstates(linewidth=1)
m.drawcoastlines(linewidth=2)

# Project coordinates
a_x, a_y = m(eqCat.data['Lon'], eqCat.data['Lat'])

# Plot skipped events first (so kept events appear on top)
m.plot(
    a_x[is_skip], a_y[is_skip],
    'o', ms=2, color='.1', alpha=0.5,
    label=f"Skipped / child ({is_skip.sum()})"
)

m.plot(
    a_x[is_keep], a_y[is_keep],
    'bo', ms=3, alpha=0.8,
    label=f"Kept / background ({is_keep.sum()})"
)



m.drawmeridians(
    np.linspace(int(xmin), xmax, 4),
    labels=[False, False, False, True],
    fontsize=12, fmt='%.1f'
)
m.drawparallels(
    np.linspace(int(ymin), ymax, 4),
    labels=[True, False, False, False],
    fontsize=12, fmt='%.2f'
)

plt.legend(loc='best')
plt.title(f"Declustering result, eta_0 = {dPar['eta_0']}")
plt.tight_layout()
plt.show()