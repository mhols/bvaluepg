# Matrix-, Bild- und Scan-Order-Konventionen

Stand: 2026-06-08

## Grundkonvention

Die Matrix ist das mathematische Objekt. Ein Bild ist die Visualisierung dieser Matrix.

- Matrixform: `shape == (ny, nx)`
- `ny`: Anzahl Zeilen, also y-Richtung im Matrixbild
- `nx`: Anzahl Spalten, also x-Richtung im Matrixbild
- `A[0, 0]`: oberes linkes Matrixelement in der Standard-Bilddarstellung

## Scan-Order

Die interne Scan-Order ist Row-Major, also NumPy-C-Order.

```python
scan = image.ravel(order="C")
image = scan.reshape((ny, nx), order="C")
```

Damit werden die Matrixeintraege zeilenweise vektorisiert:

```text
[[1, 2, 3],
 [4, 5, 6]]

-> [1, 2, 3, 4, 5, 6]
```

Die zentralen Hilfsfunktionen sind `Mixin2D.image_to_scanorder()` und
`Mixin2D.scanorder_to_image()` in `source/polyagammadensity.py`. Neue direkte
Aufrufe von `ravel()` oder `reshape()` in raeumlichem Code sollten die Order
explizit setzen oder diese Hilfsfunktionen verwenden.

## Geographische Anzeige

Die Anzeigeoption `origin="lower"` ist nur eine Plot-Konvention. Sie kann fuer
geographische Raster sinnvoll sein, wenn Zeile 0 der suedlichen bzw. minimalen
Latitude entspricht. Sie aendert aber nicht die interne Matrixform und nicht die
Scan-Order.

Bei Count-Grids aus `numpy.histogram2d` bzw. den aktuellen Erdbebenrastern gilt:
`counts[y_bin, x_bin]`, also Zeilen fuer Latitude-Bins und Spalten fuer
Longitude-Bins. Deshalb bleibt `counts.ravel(order="C")` konsistent mit der
internen Scan-Order; `counts.T.ravel()` wechselt die transponierte Scan-Order.

## Tests

Nicht-quadratische Beispiele sind Pflicht, weil quadratische Matrizen
Transpositionsfehler verdecken koennen. Die Basisfaelle liegen in
`tests/test_scanorder.py` und pruefen `2 x 3`- und `3 x 2`-Raster.
