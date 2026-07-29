# Sofiane SPDE-ETAS data

Quelle: kopiert aus `SPDE-ETAS-main/data`.

Dateien:

- `synthetic_data_case_01_patches.txt`
- `synthetic_data_case02_three_faults.txt`

Beide Dateien sind numerische Textdateien ohne Header. Jede Zeile ist ein Ereignis.

## Spalten

| Spalte | Name | Bedeutung |
|---:|---|---|
| 1 | `time` | Ereigniszeit seit Start des synthetischen Katalogs |
| 2 | `mag` | Magnitude relativ zum Cutoff, nicht klassische Mw/Ml-Werte |
| 3 | `lon` / `x` | x-Koordinate im synthetischen Gebiet |
| 4 | `lat` / `y` | y-Koordinate im synthetischen Gebiet |
| 5 | `parent_id` | Parent-Ereignis; `0` bedeutet Background-Ereignis |
| 6 | `cluster_id` / `family_id` | Cluster-/Familien-ID der ETAS-Kaskade |
| 7 | `generation` | Trigger-Generation; `0` Background, `1` direktes Kind, usw. |

## Relative Groessen

### Magnitude

Die zweite Spalte ist in Sofianes synthetischen Daten keine absolute Mw/ML-Magnitude
wie in einem realen Erdbebenkatalog. Sie ist als Magnitude relativ zum Cutoff zu
lesen:

```text
Delta M_i = M_i - M0
```

Im aktuellen `Main_script.jl` wird der Katalog mit `M0 = 0.0` gebaut. Dadurch ist
die eingelesene Spalte `mag` direkt gleich `Delta M`. Das ist fuer ETAS natuerlich,
weil Produktivitaetsterme typischerweise mit der Ueberschreitung ueber der
Vollstaendigkeitsmagnitude arbeiten, nicht zwingend mit der absoluten Magnitude.

### Zeit

Die erste Spalte ist eine relative Ereigniszeit seit Beginn des synthetischen
Katalogs. Der Code setzt keinen echten Kalenderstart, sondern verwendet
`start_time = missing` und `Tmax = maximum(time)`. Fuer ETAS reicht diese relative
Zeitachse, weil die Trigger-Kerne mit Zeitabstaenden arbeiten:

```text
t_i - t_parent
```

Die Einheit ist im Code nicht explizit dokumentiert. Aus der Skala `0 ... ca. 1000`
ist naheliegend, sie als Modellzeit, vermutlich Tage, zu interpretieren.

### Raumkoordinaten

Die Spalten 3 und 4 heissen im Code `lon` und `lat`, sind hier aber synthetische
Koordinaten im Quadrat:

```text
0 <= lon <= 5
0 <= lat <= 5
```

Das passt zur Mesh-Domain in `Main_script.jl`, die als Quadrat `[0, 5] x [0, 5]`
definiert ist. Die Namen `lon` und `lat` werden also als generische x/y-
Koordinaten fuer den raeumlichen ETAS-Kern verwendet, nicht als echte geographische
Laengen- und Breitengrade.

Im aktuellen `SPDE-ETAS-main/Main_script.jl` werden nur die ersten vier Spalten gelesen:

```julia
time = data[:, 1]
mag  = data[:, 2]
lon  = data[:, 3]
lat  = data[:, 4]
```

Die Bedeutung der Spalten 5 bis 7 ist aus dem Zahlenmuster der Dateien
abgeleitet: Background-Ereignisse haben `parent_id = 0` und `generation = 0`;
getriggerte Ereignisse verweisen in `parent_id` auf ein frueheres Ereignis und
behalten die zugehoerige `cluster_id`.

## Italien-Daten in Sofianes Paper

Im lokal vorliegenden Code `SPDE-ETAS-main/Main_script.jl` wird aktuell nur der
synthetische Katalog geladen:

```julia
const DATA_FILE = "data/synthetic_data_case_01_patches.txt"
```

Im Paper wird die Italien-Anwendung aber beschrieben.
Dort steht:

- Quelle: HORUS-Katalog
- Download: `https://horus.bo.ingv.it`
- Referenz: Lolli et al. (2020), *The homogenized instrumental seismic catalog (HORUS) of Italy from 1960 to present*
- Gesamtdatensatz: mehr als 514,465 Ereignisse
- Zeitraum des Gesamtdatensatzes: 1960 bis 2025
- Gebiet: `[37°N, 47°N] x [7°E, 18.5°E]`
- Fuer den Fit verwendet: 14,802 Ereignisse
- Auswahl fuer den Fit: `Mw >= 3.0`, Zeitraum 1980 bis 2025

Damit nutzt Sofiane im Paper denselben HORUS/INGV-Datenkontext, aber
mit einer staerkeren Magnitudenschwelle und einem laengeren Zeitraum als unsere
aktuelle INGV-Mc-2.5-Anwendung.
