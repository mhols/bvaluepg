"""Create SPDE-ETAS numeric Italy catalogues from the BvaluePG event table.

The Julia reference script reads four whitespace-separated numeric columns:
relative time, magnitude above the completeness threshold, x, and y.  This
adapter creates exactly that representation while retaining a separate mapping
file with event IDs and original values.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BVALUEPG_ROOT = PROJECT_ROOT.parent
INPUT_FILE = PROJECT_ROOT / "data" / "italy_mc3_comparison" / "events.csv"
OUTPUT_DIR = PROJECT_ROOT / "data"

MIN_MAGNITUDE = 3.0
PILOT_EVENT_COUNTS = (200, 500, 1000)
SPACE_SCALE_KM = 100.0

# Fixed analysis window from the existing preprocessing metadata.
X_MIN_ROT_KM = 265.7071505562919
X_MAX_ROT_KM = 1269.7572914814305
Y_MIN_ROT_KM = -1414.1965686237359
Y_MAX_ROT_KM = -34.860497554831255


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_selected_events() -> list[dict[str, object]]:
    required = {
        "event_id",
        "datetime",
        "mag",
        "x_rot_km",
        "y_rot_km",
        "inside_final_cut",
    }

    with INPUT_FILE.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|")
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")

        events = []
        for row in reader:
            if row["inside_final_cut"] != "True":
                continue
            magnitude = float(row["mag"])
            if magnitude < MIN_MAGNITUDE:
                continue

            events.append(
                {
                    "event_id": row["event_id"],
                    "datetime": parse_datetime(row["datetime"]),
                    "magnitude": magnitude,
                    "x_rot_km": float(row["x_rot_km"]),
                    "y_rot_km": float(row["y_rot_km"]),
                }
            )

    events.sort(key=lambda event: event["datetime"])
    if not events:
        raise ValueError("No events remain after the Italy selection.")
    if any(
        current["datetime"] <= previous["datetime"]
        for previous, current in zip(events, events[1:])
    ):
        raise ValueError("Selected event times are not strictly increasing.")
    return events


def transformed_rows(events: list[dict[str, object]]) -> list[dict[str, object]]:
    start = events[0]["datetime"]
    rows = []
    for event in events:
        time_days = (event["datetime"] - start).total_seconds() / 86_400.0
        rows.append(
            {
                **event,
                "time_days": time_days,
                "magnitude_excess": event["magnitude"] - MIN_MAGNITUDE,
                "x_model": (event["x_rot_km"] - X_MIN_ROT_KM) / SPACE_SCALE_KM,
                "y_model": (event["y_rot_km"] - Y_MIN_ROT_KM) / SPACE_SCALE_KM,
            }
        )
    return rows


def write_numeric(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        for row in rows:
            stream.write(
                f"{row['time_days']:.10e} "
                f"{row['magnitude_excess']:.10e} "
                f"{row['x_model']:.10e} "
                f"{row['y_model']:.10e}\n"
            )


def write_mapping(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "event_id",
        "datetime",
        "time_days",
        "magnitude",
        "magnitude_excess",
        "x_rot_km",
        "y_rot_km",
        "x_model",
        "y_model",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = load_selected_events()
    rows = transformed_rows(events)
    full_path = OUTPUT_DIR / "italy_mc3_sofiane.txt"
    mapping_path = OUTPUT_DIR / "italy_mc3_sofiane_mapping.csv"
    metadata_path = OUTPUT_DIR / "italy_mc3_sofiane_meta.json"

    write_numeric(full_path, rows)
    pilot_paths = {}
    for count in PILOT_EVENT_COUNTS:
        pilot_path = OUTPUT_DIR / f"italy_mc3_sofiane_first_{count}.txt"
        write_numeric(pilot_path, rows[: min(count, len(rows))])
        pilot_paths[str(count)] = str(pilot_path.relative_to(PROJECT_ROOT))
    write_mapping(mapping_path, rows)

    metadata = {
        "purpose": "adapter_to_sofiane_spde_etas_numeric_format",
        "source": str(INPUT_FILE.relative_to(BVALUEPG_ROOT)),
        "selection": {
            "inside_final_cut": True,
            "minimum_magnitude": MIN_MAGNITUDE,
            "declustering_filter_applied": False,
        },
        "columns": ["time_days", "magnitude_excess", "x_model", "y_model"],
        "time": {
            "unit": "days",
            "origin": rows[0]["datetime"].isoformat(),
            "end": rows[-1]["datetime"].isoformat(),
        },
        "magnitude": {
            "input": "absolute catalogue magnitude",
            "output": "magnitude - minimum_magnitude",
            "julia_catalog_M0": 0.0,
        },
        "space": {
            "input_unit": "rotated_km",
            "model_units_per_km": 1.0 / SPACE_SCALE_KM,
            "km_per_model_unit": SPACE_SCALE_KM,
            "x_min_rot_km": X_MIN_ROT_KM,
            "x_max_rot_km": X_MAX_ROT_KM,
            "y_min_rot_km": Y_MIN_ROT_KM,
            "y_max_rot_km": Y_MAX_ROT_KM,
            "domain_width_model": (X_MAX_ROT_KM - X_MIN_ROT_KM) / SPACE_SCALE_KM,
            "domain_height_model": (Y_MAX_ROT_KM - Y_MIN_ROT_KM) / SPACE_SCALE_KM,
            "isotropic_scaling": True,
        },
        "event_counts": {
            "full": len(rows),
            "pilots": {str(count): min(count, len(rows)) for count in PILOT_EVENT_COUNTS},
        },
        "outputs": {
            "full_numeric": str(full_path.relative_to(PROJECT_ROOT)),
            "pilot_numeric": pilot_paths,
            "mapping": str(mapping_path.relative_to(PROJECT_ROOT)),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    print(f"Full catalogue: {len(rows)} events -> {full_path}")
    for count, pilot_path in pilot_paths.items():
        print(f"Pilot catalogue: {count} events -> {PROJECT_ROOT / pilot_path}")
    print(f"Mapping: {mapping_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
