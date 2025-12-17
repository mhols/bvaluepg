# Dokumentation: Erzeugung eines Zufallsfeldes mittels Cholesky-Zerlegung

Dieses Skript erzeugt ein Zufallsfeld auf einem 2D-Gitter, dessen statistische Eigenschaften durch eine gegebene Kovarianzmatrix bestimmt werden. Die Kovarianzstruktur basiert auf einer Gaußschen Korrelationsfunktion. Die Cholesky-Zerlegung wird verwendet, um aus dieser Kovarianzmatrix eine Stichprobe aus einem mehrdimensionalen Normalverteilungsprozess zu ziehen.

---

## 1. Gaußsche Korrelationsfunktion

Die Funktion `corr_function(x1, y1, x2, y2)` berechnet die Korrelation zwischen zwei Punkten $ (x_1, y_1) $ und $ (x_2, y_2) $ anhand einer gaußförmigen Funktion:

$ \text{corr}(x_1, y_1, x_2, y_2) = \exp\left(-\frac{(x_1 - x_2)^2 + (y_1 - y_2)^2}{2 \sigma^2}\right) $ 

Dabei ist $ \sigma $ ein Maß für die Reichweite der Korrelation (hier 0.1).

Diese Funktion ist positiv definit und gewährleistet somit eine gültige Kovarianzmatrix.

---

## 2. Konstruktion der Kovarianzmatrix

Ein 2D-Meshgrid mit 30×30 Punkten im Einheitsquadrat wird erzeugt. Für jedes Punktepaar im Gitter wird die Korrelation berechnet. Die resultierende Matrix $ \mathbf{C} $ hat die Dimension $ 900 \times 900 $.

---

## 3. Cholesky-Zerlegung

Die Kovarianzmatrix $ \mathbf{C} $ ist symmetrisch und positiv definit, daher existiert eine Cholesky-Zerlegung:

$ \mathbf{C} = \mathbf{L} \mathbf{L}^T $

Um numerische Instabilitäten zu vermeiden, wird ein kleiner Wert auf die Diagonale addiert:

$ \mathbf{C}_{\text{stabil}} = \mathbf{C} + \varepsilon \cdot \mathbf{I} $

mit $ \varepsilon = 10^{-10} $.

---

## 4. Zufallsgenerierung

Ein Standardnormalvektor $ \mathbf{z} \sim \mathcal{N}(0, \mathbf{I}) $ wird erzeugt. Das Zufallsfeld ergibt sich durch:

$ \mathbf{x} = \mathbf{L} \mathbf{z} \sim \mathcal{N}(0, \mathbf{C}) $

$ \mathbf{x} $ wird anschließend auf das ursprüngliche 2D-Gitter zurückgeformt.

---

## 5. Transformation und Visualisierung

Zur Visualisierung wird auf das Zufallsfeld die Sigmoidfunktion:

$ \sigma(z) = \frac{1}{1 + e^{-z}} $

angewendet, um die Werte in den Bereich $[0, 1]$ zu bringen.

Das Feld wird als Heatmap geplottet.

---

## 6. Weitere Funktionen

- `create_meshgrid()`: Erzeugt ein leeres Gitter.
- `generate_random_field()`: Füllt das Gitter mit gleichverteilten Zufallszahlen.
- `print_grid()`, `plot_grid()`: Formatierte Text- bzw. Bildausgabe des Gitters.
- `update_cell(grid, x, y, value)`: Setzt den Wert einer Zelle manuell.

---

## Anwendungen

- Geostatistik (z.B. Bodenbeschaffenheit)
- Bildsynthese
- Simulation physikalischer Felder mit räumlicher Korrelation

