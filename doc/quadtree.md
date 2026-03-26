## QuadTree: Funktion `subdivide`

Die Funktion `subdivide` implementiert eine rekursive Raumaufteilung (QuadTree), um ein adaptives Gitter basierend auf der Dichte von Punkten (hier: Erdbeben) zu erzeugen.

### Idee

Anstatt ein fixes Raster zu verwenden, wird der Raum so lange in vier gleich große Teilbereiche (Quadranten) unterteilt, bis eine Abbruchbedingung erfüllt ist:

- entweder enthält die Zelle höchstens `NMAX` Punkte
- oder eine maximale Rekursionstiefe `MAX_DEPTH` ist erreicht

Dadurch entstehen kleine Zellen in dichten Regionen und große Zellen in dünn besetzten Bereichen.

### Parameter

- `xmin, xmax, ymin, ymax`  
  Begrenzen das aktuelle Rechteck (Bounding Box)

- `idx`  
  Indizes der Punkte, die innerhalb dieser Box liegen

- `depth`  
  Aktuelle Rekursionstiefe

### Ablauf

1. **Leerer Bereich**  
   Wenn keine Punkte enthalten sind → Abbruch

2. **Abbruchbedingung erfüllt**  
   Wenn `len(idx) <= NMAX` oder `depth >= MAX_DEPTH`:
   - Speichere die Zelle als Leaf
   - Beende Rekursion

3. **Aufteilung**  
   - Berechne Mittelpunkt `(midx, midy)`
   - Teile die Punkte in vier Quadranten:
     - SW (unten links)
     - SE (unten rechts)
     - NW (oben links)
     - NE (oben rechts)

4. **Rekursion**  
   Rufe `subdivide` für jeden Quadranten erneut auf

### Ergebnis

Die Funktion erzeugt eine Liste von „Blättern“ (`leaves`), wobei jede Zelle enthält:

- ihre räumliche Ausdehnung (`xmin, xmax, ymin, ymax`)
- die Anzahl der enthaltenen Punkte (`count`)

Diese Struktur entspricht einem adaptiven Grid und ist besonders gut geeignet für:

- räumliche Modellierung (z. B. Poisson-Modelle)
- Heatmaps mit variabler Auflösung
- effiziente Approximation dichter Punktwolken

```mermaid
flowchart TD
    A["subdivide(xmin, xmax, ymin, ymax, idx, depth)"] --> B{"len(idx) == 0?"}
    B -- Ja --> C["return"]
    B -- Nein --> D{"len(idx) <= NMAX<br/>oder<br/>depth >= MAX_DEPTH?"}

    D -- Ja --> E["leaf speichern:<br/>xmin, xmax, ymin, ymax,<br/>count = len(idx)"]
    E --> F["return"]

    D -- Nein --> G["midx = 0.5 * (xmin + xmax)<br/>midy = 0.5 * (ymin + ymax)"]
    G --> H["idx_sw: x <= midx & y <= midy"]
    H --> I["idx_se: x > midx & y <= midy"]
    I --> J["idx_nw: x <= midx & y > midy"]
    J --> K["idx_ne: x > midx & y > midy"]

    K --> L["subdivide(xmin, midx, ymin, midy, idx_sw, depth + 1)"]
    L --> M["subdivide(midx, xmax, ymin, midy, idx_se, depth + 1)"]
    M --> N["subdivide(xmin, midx, midy, ymax, idx_nw, depth + 1)"]
    N --> O["subdivide(midx, xmax, midy, ymax, idx_ne, depth + 1)"]
    O --> P["return"]
```