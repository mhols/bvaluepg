import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------------------------------
# 0) Im Script‑Ordner arbeiten
# -----------------------------------------------------
#
# Analog zur Datei ``explore_earthquake_data.py`` arbeitet dieses
# Skript zunächst immer im Verzeichnis, in dem es gespeichert ist.
# Dadurch wird sichergestellt, dass relative Pfade zu den
# Datenquellen (CSV/GeoJSON) korrekt aufgelöst werden können.
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# -----------------------------------------------------
# 1) Datei wählen und Daten laden
# -----------------------------------------------------
#
# Wie im ursprünglichen Beispiel versuchen wir zunächst die Datei
# ``earthquakes_3point5_cl_2010-2020.json`` zu laden. Falls diese
# nicht existiert, greifen wir auf die CSV‑Variante zurück. Wird
# beides nicht gefunden, wird eine Fehlermeldung ausgelöst. Das
# geladene Ergebnis ist in jedem Fall ein ``GeoDataFrame`` mit
# geographischem Koordinatensystem (EPSG:4326).
JSON_FILE = "earthquakes_3point5_cl_2010-2020.json"
CSV_FILE = "earthquakes_3point5_cl_2010-2020.csv"

print("Working directory (script dir):", os.getcwd())
print("Data files here:", [f for f in os.listdir(".") if f.endswith((".json", ".csv"))])

if os.path.exists(JSON_FILE):
    print(f"Loading GeoJSON: {JSON_FILE}")
    gdf = gpd.read_file(JSON_FILE)
    # Falls das CRS nicht gesetzt ist, nehmen wir WGS84 an
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
elif os.path.exists(CSV_FILE):
    print(f"Loading CSV: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    # Standardspaltennamen – falls anders, bitte hier anpassen
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

# -----------------------------------------------------
# 2) Deskriptive Statistik
# -----------------------------------------------------
#
# Wir übernehmen hier die gleiche Auswertung wie im ursprünglichen Skript,
# indem wir numerische Spalten beschreiben und deren Varianzen sowie
# Korrelationen berechnen. Dies ist hilfreich, um ein Gefühl für die
# Verteilung der Metadaten (z. B. Magnitude oder Tiefe) zu bekommen.
num = gdf.drop(columns="geometry", errors="ignore").select_dtypes(include=[np.number])

print("\n=== describe() ===")
print(num.describe())

print("\n=== var() ===")
print(num.var(numeric_only=True))

if num.shape[1] >= 2:
    print("\n=== corr() ===")
    print(num.corr(numeric_only=True))

# -----------------------------------------------------
# 3) Einfache Plots für Magnitude und Tiefe
# -----------------------------------------------------
#
# Wie im Originalskript erzeugen wir Histogramme für die
# Magnitude‑ und Tiefenverteilungen sowie ein Streudiagramm von
# Magnitude gegen Tiefe, sofern diese Spalten existieren. Die
# Histogramme werden nur angezeigt, wenn entsprechende Spalten
# vorhanden sind.
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
    # Je größer die Tiefe, desto weiter unten sollen die Punkte liegen
    plt.gca().invert_yaxis()
    plt.title("Magnitude vs Depth")
    plt.xlabel("mag")
    plt.ylabel("depth (km)")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------
# 4) Weltkarte / Hintergrund und Punkte
# -----------------------------------------------------
#
# Wir zeichnen eine Weltkarte als Hintergrund und legen die
# Erdbebenpunkte darüber. Die Größe der Punkte wird an die
# Magnitude angepasst, falls verfügbar. Ansonsten verwenden wir eine
# konstante Markergröße.

fig, ax = plt.subplots(figsize=(12, 6))

# Bounding Box des Datensatzes (für Zoom etc.)
minx, miny, maxx, maxy = gdf.total_bounds

import geodatasets  # pip install geodatasets, wenn nicht vorhanden

world_path = geodatasets.get_path("naturalearth land")
world = gpd.read_file(world_path)
if world.crs is None:
    world = world.set_crs("EPSG:4326", allow_override=True)
world = world.to_crs(gdf.crs)
world.plot(ax=ax, linewidth=0.3, color="lightgray", edgecolor="gray")

# -----------------------
# Optional: Kartenbereich einschränken
# -----------------------
# Padding hinzufügen (z. B. 5% Rand)
pad_x = 0.05 * (maxx - minx)
pad_y = 0.05 * (maxy - miny)

ax.set_xlim(minx - pad_x, maxx + pad_x)
ax.set_ylim(miny - pad_y, maxy + pad_y)

# Punkte plotten (Größe abhängig von der Magnitude)
if "mag" in gdf.columns:
    mag = pd.to_numeric(gdf["mag"], errors="coerce")
    size = 6 + (mag - mag.min()) / (mag.max() - mag.min() + 1e-9) * 60
    gdf.plot(ax=ax, markersize=size, alpha=0.6, color="red")
else:
    gdf.plot(ax=ax, markersize=6, alpha=0.6, color="red")

ax.set_axis_off()
plt.title("Earthquakes (points)")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 5) Adaptive QuadTree‑Grid (Multi‑Resolution)
# -----------------------------------------------------
#
# Anstatt ein fixes Raster zu verwenden, teilen wir den Raum
# rekursiv in vier Quadranten auf (QuadTree), bis die Anzahl der
# Ereignisse in jeder Zelle einen Schwellenwert nicht überschreitet.
# Dieser Schwellenwert NMAX kann angepasst werden. Für die
# folgenden Zeilen verwenden wir ein einfaches, selbst implementiertes
# QuadTree basierend auf den Punktkoordinaten. Die Grundidee:  ein QuadTree teilt einen Bereich so lange in vier
# Unterregionen, bis jede Region höchstens K Objekte enthält.
# Die rekursive Unterteilung wird gestoppt, wenn die Anzahl der
# Objekte die Schwelle unterschreitet oder eine maximale Tiefe
# erreicht ist.

# Punktkoordinaten aus dem GeoDataFrame
xs = gdf.geometry.x.to_numpy()
ys = gdf.geometry.y.to_numpy()

# Parameter: maximal erlaubte Ereignisse pro QuadTree‑Zelle (NMAX)
# und (optional) maximale Rekursionstiefe. Je kleiner NMAX, desto
# mehr Zellen werden erzeugt.
NMAX = 25
MAX_DEPTH = 12  # None für unbegrenzte Tiefe

# Liste, in der die Blätter (finale Zellen) gesammelt werden
leaves: list[dict] = []

# Alle Punkte als Indexliste
indices = np.arange(len(xs))

def subdivide(xmin: float, xmax: float, ymin: float, ymax: float,
              idx: np.ndarray, depth: int = 0) -> None:
    """
    Rekursive Funktion zum Erstellen des QuadTrees. Solange die
    Anzahl der Indizes größer als NMAX ist und die maximale Tiefe
    nicht erreicht ist, wird der Bereich in vier Quadranten unterteilt.

    :param xmin: linke Grenze des aktuellen Bereichs
    :param xmax: rechte Grenze des aktuellen Bereichs
    :param ymin: untere Grenze des aktuellen Bereichs
    :param ymax: obere Grenze des aktuellen Bereichs
    :param idx:  Array mit Indizes der Punkte, die in diesem Bereich liegen
    :param depth: aktuelle Rekursionstiefe
    """
    # Wenn keine Punkte vorhanden sind, nichts ablegen
    if len(idx) == 0:
        return
    # Abbruch: Wenn die Anzahl der Punkte gering genug oder die
    # maximale Tiefe erreicht ist, dieses Rechteck als Blatt speichern
    if len(idx) <= NMAX or (MAX_DEPTH is not None and depth >= MAX_DEPTH):
        leaves.append({
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "count": int(len(idx))
        })
        return
    # Mittelpunkte berechnen
    midx = 0.5 * (xmin + xmax)
    midy = 0.5 * (ymin + ymax)
    # Punkte den vier Quadranten zuordnen
    # Südwesten (SW): x <= midx, y <= midy
    idx_sw = idx[(xs[idx] <= midx) & (ys[idx] <= midy)]
    # Südosten (SE): x > midx, y <= midy
    idx_se = idx[(xs[idx] > midx) & (ys[idx] <= midy)]
    # Nordwesten (NW): x <= midx, y > midy
    idx_nw = idx[(xs[idx] <= midx) & (ys[idx] > midy)]
    # Nordosten (NE): x > midx, y > midy
    idx_ne = idx[(xs[idx] > midx) & (ys[idx] > midy)]
    # Rekursion für jeden Unterbereich
    subdivide(xmin, midx, ymin, midy, idx_sw, depth + 1)
    subdivide(midx, xmax, ymin, midy, idx_se, depth + 1)
    subdivide(xmin, midx, midy, ymax, idx_nw, depth + 1)
    subdivide(midx, xmax, midy, ymax, idx_ne, depth + 1)

# QuadTree generieren
subdivide(minx, maxx, miny, maxy, indices, depth=0)

print(f"\nNumber of quadtree cells (leaves): {len(leaves)}")

# In DataFrame überführen
grid_df = pd.DataFrame(leaves)
grid_df["x_center"] = 0.5 * (grid_df["xmin"] + grid_df["xmax"])
grid_df["y_center"] = 0.5 * (grid_df["ymin"] + grid_df["ymax"])

print(grid_df.head())

# -----------------------------------------------------
# 6) Visualisierung des adaptiven Gitters
# -----------------------------------------------------
#
# Wir nutzen Matplotlib und zeichnen die QuadTree‑Zellen als
# Rechtecke, deren Farbe die Anzahl der Erdbeben darstellt. Zudem
# legen wir wieder die Weltkarte im Hintergrund an und zeichnen
# Erdbebenpunkte über die Zellen (kleine Marker). Die Farben werden
# anhand eines Farbverlaufs (viridis) skaliert.
fig, ax = plt.subplots(figsize=(12, 6))
world.plot(ax=ax, linewidth=0.3, color="lightgray", edgecolor="gray")

# -----------------------
# Kartenbereich einschränken
# -----------------------
# Padding hinzufügen (z. B. 5% Rand)
pad_x = 0.05 * (maxx - minx)
pad_y = 0.05 * (maxy - miny)

ax.set_xlim(minx - pad_x, maxx + pad_x)
ax.set_ylim(miny - pad_y, maxy + pad_y)

counts = grid_df["count"]
norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
cmap = plt.cm.viridis

for _, row in grid_df.iterrows():
    color = cmap(norm(row["count"]))
    rect = Rectangle(
        (row["xmin"], row["ymin"]),
        row["xmax"] - row["xmin"],
        row["ymax"] - row["ymin"],
        facecolor=color,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.6,
    )
    ax.add_patch(rect)

# Erdbebenpunkte als schwarze Punkte; kleinere Marker für bessere Sichtbarkeit
# gdf.plot(ax=ax, markersize=2, alpha=0.8, color="black")

ax.set_axis_off()
plt.title(f"Adaptive grid (quadtree) with NMAX={NMAX}")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy; Matplotlib >= 3.3
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", label="Number of earthquakes")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 7) Export der QuadTree‑Daten
# -----------------------------------------------------
#
# Wir speichern die adaptiven Grid‑Daten als CSV und die
# dazugehörigen Ereigniszahlen als Numpy‑Array. So können die
# Ergebnisse leicht in anderen Anwendungen importiert werden.
grid_df.to_csv("eq_quadtree_grid.csv", index=False)
np.save("eq_quadtree_counts.npy", grid_df["count"].to_numpy())
print("Saved quadtree grid to eq_quadtree_grid.csv and eq_quadtree_counts.npy")