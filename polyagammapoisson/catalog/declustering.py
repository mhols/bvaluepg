"""Nearest-neighbour earthquake declustering using tgoebel's clustering code.

https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.018501
https://github.com/tgoebel/clustering-analysis
https://zaliapin.github.io/pubs/Zaliapin_Ben-Zion_JGR20.pdf
"""

import numpy as np
import pandas as pd

from .tgoebel_clustering_analysis import EqCat, clustering


def _background_mask(eqcat, Mc, f_eta_0):
    if np.size(eqcat.data["N"]) < 2:
        return np.ones(np.size(eqcat.data["N"]), dtype=bool)

    # Give co-located events distinct longitudes in the working catalogue.
    locations = pd.DataFrame({"lon": eqcat.data["Lon"], "lat": eqcat.data["Lat"]})
    offsets = locations.groupby(["lon", "lat"], sort=False).cumcount().to_numpy()
    eqcat.data["Lon"] = eqcat.data["Lon"] + offsets * 1e-10

    distances = clustering.NND_eta(eqcat, {"D": 1.6, "b": 1.0, "Mc": Mc}, verbose=True)
    triggered = np.log10(distances["aNND"]) < f_eta_0
    # Excluding linked children retains both cluster roots and isolated events.
    return ~np.isin(eqcat.data["N"], distances["aEqID_c"][triggered])


def get_background_events(file_in, Mc=None, f_eta_0=None):
    """Return the background mask for a MATLAB-format earthquake catalogue."""
    eqcat = EqCat.EqCat()
    eqcat.loadMatBin(str(file_in))
    return _background_mask(eqcat, Mc, f_eta_0)


def decluster(df, Mc, f_eta_0):
    """Return a keep mask aligned with the normalized catalogue's input rows."""
    times = pd.to_datetime(df["Time"], errors="coerce").dt.tz_localize(None).dt.floor("us")
    locations = df[["Lon", "Lat", "MAG"]].apply(pd.to_numeric, errors="coerce")
    if times.isna().any() or locations.isna().any().any():
        raise ValueError("Declustering requires valid Time, Lon, Lat, and MAG values.")

    # CatalogMixin already sorts by Time. Sorting positions also supports
    # direct callers with unsorted input and preserves the mask's alignment.
    order = np.argsort(times.to_numpy(), kind="stable")
    times = times.iloc[order]
    years = times.dt.year
    year_start = pd.to_datetime(years.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((years + 1).astype(str) + "-01-01")
    decimal_year = years + (times - year_start) / (next_year_start - year_start)

    eqcat = EqCat.EqCat()
    eqcat.data = {
        "N": order + 1,
        "Time": decimal_year.to_numpy(dtype=float),
        "Lon": locations["Lon"].to_numpy(dtype=float)[order],
        "Lat": locations["Lat"].to_numpy(dtype=float)[order],
        "Mag": locations["MAG"].to_numpy(dtype=float)[order],
    }
    keep = np.empty(len(df), dtype=bool)
    keep[order] = _background_mask(eqcat, Mc, f_eta_0)
    return keep
