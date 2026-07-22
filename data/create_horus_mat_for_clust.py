"""
Create a .mat catalogue from HORUS for declustering


The .mat file contains the keys required for performing the nnd declustering:
    N, Time, Mag, Lat, Lon, Depth
plus date/time fields and lowercase aliases.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat


# ============================================================
# PARAMETERS TO CHANGE
# ============================================================

INPUT_FILE = "italy_ingv_m2point5_2015-2026.txt" # "italy_ingv_rotated_rect_events.csv"
OUTPUT_FILE = "italy_ingv_m2point5_2015-2026.mat"

# Optional filter before saving.
MIN_MAGNITUDE = None     # e.g. 3.0, or None
YEAR_MIN = None          # e.g. 1980, or None
YEAR_MAX = None          # e.g. 2020, or None


# ============================================================
# LOAD HORUS
# ============================================================

def load_horus(path):
    df = pd.read_csv(path, low_memory=False, sep='|', skiprows=0)
    df.columns = [str(c).strip() for c in df.columns]

    # Rename INGV CSV columns to the names expected by the rest of the script.
    df = df.rename(
        columns={
            "#EventID": "event_id",
            "Time": "datetime",
            "Latitude": "lat",
            "Longitude": "lon",
            "Depth/Km": "depth",
            "Magnitude": "mag",
        }
    )

    required = ["datetime", "lat", "lon", "depth", "mag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns after renaming: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["second"] = (
        df["datetime"].dt.second
        + df["datetime"].dt.microsecond / 1_000_000
    )

    for col in ["lat", "lon", "depth", "mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "event_id" in df.columns:
        df["event_id_num"] = pd.to_numeric(df["event_id"], errors="coerce")
    else:
        df["event_id_num"] = np.nan

    df = df.dropna(subset=["datetime", "lat", "lon", "mag"]).copy()

    return df


def add_time_fields(df):
    df = df.copy()

    sec = df["second"].fillna(0).to_numpy(float)
    whole_second = np.floor(sec).astype(int)
    microsecond = np.round((sec - whole_second) * 1_000_000).astype(int)

    carry = microsecond >= 1_000_000
    whole_second[carry] += 1
    microsecond[carry] -= 1_000_000

    df["datetime"] = pd.to_datetime(
        {
            "year": df["year"].astype(int),
            "month": df["month"].astype(int),
            "day": df["day"].astype(int),
            "hour": df["hour"].fillna(0).astype(int),
            "minute": df["minute"].fillna(0).astype(int),
            "second": whole_second,
            "microsecond": microsecond,
        },
        errors="coerce",
    )

    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Decimal year. This is what clust_SoCal.ipynb expects when it does
    # selectEvents(tmin, tmax, 'Time') with values such as 1980 and 2012.
    year_start = pd.to_datetime(df["datetime"].dt.year.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((df["datetime"].dt.year + 1).astype(str) + "-01-01")
    df["decimal_year"] = (
        df["datetime"].dt.year
        + (df["datetime"] - year_start).dt.total_seconds()
        / (next_year_start - year_start).dt.total_seconds()
    )

    # Numeric event IDs for EqCat. If HORUS IDs are not numeric, use 1..N.
    if df["event_id_num"].notna().any():
        event_id = df["event_id_num"].to_numpy(dtype=float, copy=True)
        missing = ~np.isfinite(event_id)
        event_id[missing] = np.arange(1, len(df) + 1, dtype=float)[missing]
    else:
        event_id = np.arange(1, len(df) + 1, dtype=float)

    # Ensure unique IDs, because later notebook cells use EqCat.selEventsFromID.
    # If duplicates exist, replace all IDs with 1..N.
    if len(np.unique(event_id)) != len(event_id):
        event_id = np.arange(1, len(df) + 1, dtype=float)

    df["N"] = event_id

    return df


def save_eqcat_mat(df, output_file):
    # EqCat.loadMatBin in the notebook needs these exact fields:
    # Time, Mag, Lat, Lon, Depth, N.
    # I also save common aliases to make the file usable by other scripts.
    mat = {
        # Main EqCat-style names
        "N": df["N"].to_numpy(float),
        "Time": df["decimal_year"].to_numpy(float),
        "Mag": df["mag"].to_numpy(float),
        "Lat": df["lat"].to_numpy(float),
        "Lon": df["lon"].to_numpy(float),
        "Depth": df["depth"].fillna(0.0).to_numpy(float),

        # Date/time fields
        "Year": df["year"].to_numpy(float),
        "Month": df["month"].to_numpy(float),
        "Day": df["day"].to_numpy(float),
        "Hour": df["hour"].fillna(0.0).to_numpy(float),
        "Minute": df["minute"].fillna(0.0).to_numpy(float),
        "Second": df["second"].fillna(0.0).to_numpy(float),

        # Lowercase aliases
        "time": df["decimal_year"].to_numpy(float),
        "mag": df["mag"].to_numpy(float),
        "latitude": df["lat"].to_numpy(float),
        "longitude": df["lon"].to_numpy(float),
        "depth": df["depth"].fillna(0.0).to_numpy(float),

        # Useful text time field as object array; ignored by most numerical code
        "datetime_iso": df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").to_numpy(dtype=object),
    }

    savemat(output_file, mat, do_compression=True)


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    print(f"Reading HORUS catalogue: {input_path}")
    df = load_horus(input_path)
    df = add_time_fields(df)

    if MIN_MAGNITUDE is not None:
        df = df[df["mag"] >= MIN_MAGNITUDE].copy()

    if YEAR_MIN is not None:
        df = df[df["decimal_year"] >= YEAR_MIN].copy()

    if YEAR_MAX is not None:
        df = df[df["decimal_year"] <= YEAR_MAX].copy()

    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"Events to save: {len(df)}")
    print("Time range:", df["decimal_year"].min(), "to", df["decimal_year"].max())
    print("Magnitude range:", df["mag"].min(), "to", df["mag"].max())
    print("Columns saved for EqCat: N, Time, Mag, Lat, Lon, Depth")

    save_eqcat_mat(df, output_path)

    print(f"Saved MAT file: {output_path.resolve()}")


if __name__ == "__main__":
    main()
