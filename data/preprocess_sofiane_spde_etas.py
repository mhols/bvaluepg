"""Formatiere Sofianes SPDE-ETAS-Kataloge fuer unsere BvaluePG-Pipeline.

Das Skript liest die numerischen Sofiane-Dateien aus data/sofiane_spde_etas und schreibt Pipe-CSV-Dateien mit
den Spaltennamen, die unser preprocess_nnd_rot_cut_bin.py direkt laden kann.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PARAMETERS TO CHANGE
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "sofiane_spde_etas"

INPUT_FILES = [
    INPUT_DIR / "synthetic_data_case_01_patches.txt",
    INPUT_DIR / "synthetic_data_case02_three_faults.txt",
]

START_TIME = "2015-01-01T00:00:00"
TIME_UNIT = "D"  # Sofianes Zeitfeld wird als Tage interpretiert.

# Sofianes Magnituden beziehen sich in seiner aktuellen Konfiguration auf einen Schwellenwert. Wir behalten
# die ursprünglichen Werte in „sofiane_mag_relative“ bei und erstellen eine Spalte mit
# absoluten Magnitudenwerten, damit unser Standardfilter „MIN_MAGNITUDE=2,5“ nicht alle
# Ereignisse entfernt.
MAGNITUDE_OFFSET = 2.5

DEPTH_KM = 0.0
OUTPUT_SEPARATOR = "|"
SHOW_PLOTS = True


def read_sofiane_catalog(path: Path) -> pd.DataFrame:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"{path} must contain at least four numeric columns.")

    columns = [
        "sofiane_time",
        "sofiane_mag_relative",
        "lon",
        "lat",
        "sofiane_parent_id",
        "sofiane_cluster_id",
        "sofiane_generation",
    ]
    if data.shape[1] != len(columns):
        raise ValueError(f"{path} has {data.shape[1]} columns; expected {len(columns)}.")

    return pd.DataFrame(data, columns=columns)


def add_bvaluepg_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    result = df.copy()
    start = pd.Timestamp(START_TIME)
    result["datetime"] = start + pd.to_timedelta(result["sofiane_time"], unit=TIME_UNIT)
    result["event_id"] = np.arange(1, len(result) + 1, dtype=int)
    result["depth"] = DEPTH_KM
    result["mag"] = MAGNITUDE_OFFSET + result["sofiane_mag_relative"]

    return pd.DataFrame(
        {
            "#EventID": result["event_id"],
            "Time": result["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "Latitude": result["lat"],
            "Longitude": result["lon"],
            "Depth/Km": result["depth"],
            "Author": "SOFIANE",
            "Catalog": "SPDE-ETAS",
            "Contributor": "",
            "ContributorID": "",
            "MagType": "relative_plus_offset",
            "Magnitude": result["mag"],
            "MagAuthor": "synthetic",
            "EventLocationName": source_name,
            "EventType": "synthetic_earthquake",
            "sofiane_time": result["sofiane_time"],
            "sofiane_mag_relative": result["sofiane_mag_relative"],
            "sofiane_parent_id": result["sofiane_parent_id"].astype(int),
            "sofiane_cluster_id": result["sofiane_cluster_id"].astype(int),
            "sofiane_generation": result["sofiane_generation"].astype(int),
        }
    )


def output_path_for(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_bvaluepg.csv")


def show_plots(original: pd.DataFrame, transformed: pd.DataFrame, source_name: str) -> None:
    if not SHOW_PLOTS:
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(source_name)

    axes[0, 0].scatter(
        original["lon"],
        original["lat"],
        c=original["sofiane_mag_relative"],
        s=8,
        alpha=0.7,
        cmap="viridis",
    )
    axes[0, 0].set_title("Sofiane: space, rel. magnitude")
    axes[0, 0].set_xlabel("lon / x")
    axes[0, 0].set_ylabel("lat / y")
    axes[0, 0].set_aspect("equal", adjustable="box")

    axes[0, 1].scatter(original["sofiane_time"], original["sofiane_mag_relative"], s=8, alpha=0.7)
    axes[0, 1].set_title("Sofiane: time vs rel. magnitude")
    axes[0, 1].set_xlabel("sofiane_time")
    axes[0, 1].set_ylabel("sofiane_mag_relative")

    axes[0, 2].hist(original["sofiane_generation"], bins=np.arange(-0.5, original["sofiane_generation"].max() + 1.5, 1))
    axes[0, 2].set_title("Sofiane: generation")
    axes[0, 2].set_xlabel("generation")
    axes[0, 2].set_ylabel("count")

    scatter = axes[1, 0].scatter(
        transformed["Longitude"],
        transformed["Latitude"],
        c=transformed["Magnitude"],
        s=8,
        alpha=0.7,
        cmap="viridis",
    )
    axes[1, 0].set_title("BvaluePG: space, Magnitude")
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")
    axes[1, 0].set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].scatter(pd.to_datetime(transformed["Time"]), transformed["Magnitude"], s=8, alpha=0.7)
    axes[1, 1].set_title("BvaluePG: Time vs Magnitude")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].tick_params(axis="x", labelrotation=30)

    background = transformed["sofiane_parent_id"] == 0
    axes[1, 2].bar(["background", "triggered"], [int(background.sum()), int((~background).sum())], color=["0.45", "0.75"])
    axes[1, 2].set_title("BvaluePG: parent status")
    axes[1, 2].set_ylabel("count")

    plt.show()


def main() -> None:
    for input_path in INPUT_FILES:
        df = read_sofiane_catalog(input_path)
        out = add_bvaluepg_columns(df, source_name=input_path.stem)
        output_path = output_path_for(input_path)
        out.to_csv(output_path, sep=OUTPUT_SEPARATOR, index=False)
        print(f"{input_path.name}: {len(out)} events -> {output_path}")
        show_plots(df, out, source_name=input_path.stem)


if __name__ == "__main__":
    main()
