import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# -----------------------
# 0) Immer im Script-Ordner arbeiten
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# -----------------------
# 1) Datei wählen
# -----------------------
JSON_FILE = "earthquakes_3comma5_cl_2010-2020.json"
CSV_FILE = "earthquakes_3comma5_cl_2010-2020.csv"

print("Working directory (script dir):", os.getcwd())
print("Data files here:", [f for f in os.listdir(".") if f.endswith((".json", ".csv"))])

# Erst JSON versuchen, sonst CSV
if os.path.exists(JSON_FILE):
    print(f"Loading GeoJSON: {JSON_FILE}")
    gdf = gpd.read_file(JSON_FILE)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

elif os.path.exists(CSV_FILE):
    print(f"Loading CSV: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    # Standardspalten (falls sie anders heißen: hier anpassen)
    lat_col = "latitude"
    lon_col = "longitude"

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
else:
    raise FileNotFoundError(
        f"Keine Daten gefunden in: {os.getcwd()}\n"
        f"Erwartet: {JSON_FILE} oder {CSV_FILE} im gleichen Ordner wie das Skript."
    )

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
plt.show()

# -----------------------
# 5) Anzahl Events in 30x30 Boxen (Grid / Heatmap-Daten)
# -----------------------

GRID_N = 30

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


print("Saved: earthquake_grid_counts_30x30.npy")
print("Saved: earthquake_grid_counts_30x30.csv")

# Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(
    counts,
    origin="lower",
    extent=[minx, maxx, miny, maxy],
    aspect="auto",
)
plt.title("Event counts per 30x30 grid cell")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.tight_layout()
plt.show()