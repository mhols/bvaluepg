import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# -----------------------
# 0) Im Script-Ordner arbeiten
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# -----------------------
# 1) Datei wählen und laden
# -----------------------
DEFAULT_FILES = [
    "earthquakes_california_m2p5_2010_2025.txt",
    "earthquakes_2point5_ingv_italy_2015-2026.json",
    "earthquakes_mw2point5_bcsf_renass_france_1962-2021.json",
    "earthquakes_3point5_cl_2010-2020.json",
    "instrumental-seismicity-in-mainland-france_dataset_1962-2021.json",
    "italy_ingv_m2point5_2015-2026.txt",
    "earthquakes_3point5_cl_2010-2020.csv",
]

print("Working directory (script dir):", os.getcwd())
print("Data files here:", [f for f in os.listdir(".") if f.endswith((".json", ".csv", ".txt"))])

def choose_data_file() -> Path:
    if len(sys.argv) > 1:
        selected = Path(sys.argv[1])
        if not selected.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {selected}")
        return selected

    for filename in DEFAULT_FILES:
        path = Path(filename)
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Keine passende Daten-Datei gefunden in: {os.getcwd()}\n"
        "Du kannst eine Datei explizit angeben, z.B.:\n"
        "python explore_earthquake_data.py earthquakes_mw2point5_bcsf_renass_france_1962-2021.json"
    )


def normalize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rename_map = {
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Magnitude": "localMagnitude",
        "Mw": "mw",
        "Depth/km": "depth",
        "Depth/Km": "depth",
        "MAG": "mag",
        "LAT": "latitude",
        "LON": "longitude",
        "DEPTH": "depth",
    }
    gdf = gdf.rename(columns={k: v for k, v in rename_map.items() if k in gdf.columns})

    if "mag" not in gdf.columns:
        if "mw" in gdf.columns:
            gdf["mag"] = pd.to_numeric(gdf["mw"], errors="coerce")
        elif "localMagnitude" in gdf.columns:
            gdf["mag"] = pd.to_numeric(gdf["localMagnitude"], errors="coerce")

    if "depth" not in gdf.columns and gdf.geometry.geom_type.eq("Point").all():
        # Viele GeoJSONs speichern Tiefe als dritte Koordinate.
        depths = []
        for geom in gdf.geometry:
            try:
                depths.append(geom.z)
            except Exception:
                depths.append(np.nan)
        gdf["depth"] = depths

    for col in ["mag", "depth", "latitude", "longitude", "mw", "localMagnitude"]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    return gdf


def load_geojson(path: Path) -> gpd.GeoDataFrame:
    print(f"Loading GeoJSON: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return normalize_columns(gdf)


def load_csv(path: Path) -> gpd.GeoDataFrame:
    print(f"Loading CSV: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError(f"CSV braucht latitude/longitude-Spalten: {path}")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    return normalize_columns(gdf)


def load_ingv_txt(path: Path) -> gpd.GeoDataFrame:
    print(f"Loading INGV FDSN TXT: {path}")
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().lstrip("#").split("|")
        df = pd.read_csv(f, sep="|", names=header)

    df = df.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Magnitude": "mag",
            "Depth/Km": "depth",
            "Time": "time",
            "EventID": "event_id",
            "EventLocationName": "place",
            "MagType": "magType",
        }
    )
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"], df["depth"]),
        crs="EPSG:4326",
    )
    return normalize_columns(gdf)


def load_scedc_txt(path: Path) -> gpd.GeoDataFrame:
    print(f"Loading SCEDC TXT: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 13 or "/" not in parts[0]:
            continue
        date, time, event_type, geo_type, mag, mag_type, lat, lon, depth, quality, evid, nph, ngrm = parts[:13]
        rows.append(
            {
                "date": date,
                "time": time,
                "event_type": event_type,
                "geo_type": geo_type,
                "mag": mag,
                "magType": mag_type,
                "latitude": lat,
                "longitude": lon,
                "depth": depth,
                "quality": quality,
                "event_id": evid,
                "nph": nph,
                "ngrm": ngrm,
            }
        )

    if not rows:
        raise ValueError(f"Keine SCEDC-Eventzeilen gefunden: {path}")

    df = pd.DataFrame(rows)
    for col in ["latitude", "longitude", "mag", "depth", "nph", "ngrm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"], df["depth"]),
        crs="EPSG:4326",
    )
    return normalize_columns(gdf)


def load_txt(path: Path) -> gpd.GeoDataFrame:
    first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if first_line.startswith("#EventID|"):
        return load_ingv_txt(path)
    return load_scedc_txt(path)


data_file = choose_data_file()
data_stem = data_file.stem

if data_file.suffix.lower() in [".json", ".geojson"]:
    gdf = load_geojson(data_file)
elif data_file.suffix.lower() == ".csv":
    gdf = load_csv(data_file)
elif data_file.suffix.lower() == ".txt":
    gdf = load_txt(data_file)
else:
    raise ValueError(f"Nicht unterstützter Dateityp: {data_file}")

print("Rows:", len(gdf))
print("Columns:", list(gdf.columns))
print("CRS:", gdf.crs)

# -----------------------
# 2) Deskriptive Statistik
# -----------------------
num = gdf.drop(columns="geometry", errors="ignore").select_dtypes(include=[np.number])

print("\n=== describe() ===")
print(num.describe())

print("\n=== var() ===")
print(num.var(numeric_only=True))

if num.shape[1] >= 2:
    print("\n=== corr() ===")
    print(num.corr(numeric_only=True))

# -----------------------
# 3) Plots (wenn Spalten existieren)
# -----------------------
if "mag" in gdf.columns:
    plt.figure()
    plt.hist(pd.to_numeric(gdf["mag"], errors="coerce").dropna(), bins=40)
    plt.title("Magnitude (Histogram)")
    plt.xlabel("mag")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

if "depth" in gdf.columns:
    plt.figure()
    plt.hist(pd.to_numeric(gdf["depth"], errors="coerce").dropna(), bins=60)
    plt.title("Depth (Histogram)")
    plt.xlabel("depth (km)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

if "mag" in gdf.columns and "depth" in gdf.columns:
    mag = pd.to_numeric(gdf["mag"], errors="coerce")
    dep = pd.to_numeric(gdf["depth"], errors="coerce")
    m = (~mag.isna()) & (~dep.isna())

    plt.figure()
    plt.scatter(mag[m], dep[m], s=6, alpha=0.35)
    plt.gca().invert_yaxis()
    plt.title("Magnitude vs Depth")
    plt.xlabel("mag")
    plt.ylabel("depth (km)")
    plt.tight_layout()
    plt.show()

# -----------------------
# 4) Weltkarte / Hintergrund
# -----------------------
# GeoPandas hat die eingebauten Beispiel-Datasets (naturalearth_lowres) ab v1.0 entfernt.
# Daher:`geodatasets` installieren und NaturalEarth darüber.

fig, ax = plt.subplots(figsize=(12, 6))

import geodatasets  # pip install geodatasets

world_path = geodatasets.get_path("naturalearth land")
world = gpd.read_file(world_path)

if world.crs is None:
    world = world.set_crs("EPSG:4326", allow_override=True)

world = world.to_crs(gdf.crs)
world.plot(ax=ax, linewidth=0.3, color="lightgray", edgecolor="gray")

# Punkte plotten (Größe ~ Magnitude)
if "mag" in gdf.columns:
    mag = pd.to_numeric(gdf["mag"], errors="coerce")
    size = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
    size = 6 + 60 * size.fillna(0)
    gdf.plot(ax=ax, markersize=size, alpha=0.6, color="red")
else:
    gdf.plot(ax=ax, markersize=6, alpha=0.6, color="red")

ax.set_axis_off()
plt.title("Earthquakes (points)")
plt.tight_layout()
# plt.show()

# -----------------------
# 5) Anzahl Events in 30x30 Boxen (Grid / Heatmap-Daten)
# -----------------------

GRID_N = 500

# Bounding Box des Datensatzes
minx, miny, maxx, maxy = gdf.total_bounds

# Kanten (31 Kanten -> 30 Zellen)
x_edges = np.linspace(minx, maxx, GRID_N + 1)
y_edges = np.linspace(miny, maxy, GRID_N + 1)

# Punktkoordinaten
xs = gdf.geometry.x.to_numpy()
ys = gdf.geometry.y.to_numpy()

# ich muss mal EPSG:4326 checken, flächentreu oder Kugelgeometrie
# wahrscheinlich brauchen ich noch ne Projektion fuer gleich große Flaechen

# 2D-Histogramm: counts[y_bin, x_bin]
# https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
counts, y_edges_out, x_edges_out = np.histogram2d(ys, xs, bins=[y_edges, x_edges])
counts = counts.astype(int)

print(f"\nGrid counts shape: {counts.shape} (rows=y, cols=x)")
print(f"Total events counted in grid: {counts.sum()} (should equal rows: {len(gdf)})")

# DataFrame (jede Zelle als Zeile)
rows = []
for iy in range(GRID_N):
    for ix in range(GRID_N):
        rows.append(
            {
                "ix": ix,
                "iy": iy,
                "xmin": x_edges[ix],
                "xmax": x_edges[ix + 1],
                "ymin": y_edges[iy],
                "ymax": y_edges[iy + 1],
                "count": int(counts[iy, ix]),
            }
        )

grid_df = pd.DataFrame(rows)

# Zellenmittelpunkte
grid_df["x_center"] = 0.5 * (grid_df["xmin"] + grid_df["xmax"])
grid_df["y_center"] = 0.5 * (grid_df["ymin"] + grid_df["ymax"])

# Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(
    counts,
    origin="lower",
    extent=[minx, maxx, miny, maxy],
    aspect="auto",
    vmax=counts.max()
)
plt.colorbar()
plt.title("Event counts per 30x30 grid cell")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.tight_layout()
plt.show()

# --- Export für das Gibbs-Experiment ---
counts_file = f"{data_stem}_counts_{GRID_N}x{GRID_N}.npy"
grid_file = f"{data_stem}_grid_30x30.csv"
meta_file = f"{data_stem}_grid_meta.json"

np.save(counts_file, counts.astype(int))
grid_df.to_csv(grid_file, index=False)

meta = {
    "source_file": str(data_file),
    "GRID_N": GRID_N,
    "minx": float(minx), "maxx": float(maxx),
    "miny": float(miny), "maxy": float(maxy),
    "rows": int(len(gdf)),
}
with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved: {counts_file}, {grid_file}, {meta_file}")
