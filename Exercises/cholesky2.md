# Generierung von Zufallsfeldern mit gegebener Kovarianzstruktur mittels Cholesky-Zerlegung

Dieses Dokument erklärt die mathematischen und programmiertechnischen Grundlagen zur Erzeugung eines Zufallsfeldes mit vorgegebener Kovarianzstruktur auf einem 2D-Gitter.

## 1. Gaußsche Korrelationsfunktion

Die Funktion `corr_function(x1, y1, x2, y2)` definiert die Kovarianz zwischen zwei Punkten im Gitter mittels einer gaußförmigen Abstandsabhängigkeit:

\[
C((x_1, y_1), (x_2, y_2)) = \exp\left(-\frac{(x_1 - x_2)^2 + (y_1 - y_2)^2}{2 \sigma^2}\right)
\]

Hierbei ist \(\sigma\) ein Parameter, der die Reichweite der Korrelation bestimmt. Die entstehende Kovarianzmatrix ist symmetrisch und (semi-)positiv definit.

## 2. Meshgrid-Erzeugung

Das Gitter wird mit `np.meshgrid` erzeugt. Dieses erzeugt ein kartesisches Produkt aus den x- und y-Werten, wodurch eine regelmäßige Diskretisierung des Einheitsquadrats entsteht.

## 3. Aufbau der Kovarianzmatrix

Die Kovarianzmatrix `Cova_matrix` wird durch paarweises Anwenden der Korrelationsfunktion auf alle Gitterpunkte gebildet. Sie hat die Dimension \( N \times N \), wobei \( N = \text{Anzahl der Gitterzellen} \).

## 4. Cholesky-Zerlegung

Zur Simulation eines Zufallsfeldes mit der gegebenen Kovarianzstruktur wird eine Cholesky-Zerlegung durchgeführt:

\[
\mathbf{C} = \mathbf{L} \mathbf{L}^T
\]

Dabei ist \(\mathbf{C}\) die Kovarianzmatrix und \(\mathbf{L}\) eine untere Dreiecksmatrix. Damit kann ein Vektor \( \mathbf{z} \sim \mathcal{N}(0, I) \) so transformiert werden, dass:

\[
\mathbf{x} = \mathbf{L} \mathbf{z} \sim \mathcal{N}(0, \mathbf{C})
\]

Die kleine Regularisierung \(1\text{e-}10 \cdot I\) wird zur numerischen Stabilität hinzugefügt.

## 5. Transformation mit der Sigmoidfunktion

Zur Darstellung als normiertes Feld wird der so erzeugte Vektor mit einer Sigmoidfunktion transformiert:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

Dies bringt die Feldwerte in den Bereich \((0, 1)\) und ist hilfreich für Visualisierungen.

## 6. Visualisierung

Das resultierende Feld wird mit `matplotlib.pyplot.imshow` visualisiert. Die Farbdarstellung zeigt die Stärke der Korrelation an verschiedenen Stellen im Gitter.

## 7. Weitere Hilfsfunktionen

- `create_meshgrid`: erzeugt ein Dictionary-basiertes Gitter.
- `generate_random_field`: füllt ein Gitter mit Zufallswerten zwischen 0 und 1.
- `print_grid` & `plot_grid`: geben die Feldwerte als Text bzw. Bild aus.
- `update_cell`: erlaubt gezielte manuelle Manipulation von Zellwerten.

## Fazit

Diese Methode erlaubt es, Zufallsfelder mit beliebiger (symmetrischer und positiv definiter) Kovarianzstruktur zu simulieren. Die Cholesky-Zerlegung ist hierbei der zentrale numerische Baustein.