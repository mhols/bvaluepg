                                                                                                                                                      QW'''# cholesky.py
# Dieses Modul generiert Zufallsfelder auf einem 2D-Gitter unter Verwendung einer Kovarianzmatrix, 
# die durch eine Gaußsche Korrelationsfunktion definiert ist. Es beinhaltet Funktionen zur 
# Erstellung von Meshgrids, zur Berechnung von Korrelationsmatrizen, zur Erzeugung von Zufallsfeldern 
# mittels Cholesky-Zerlegung sowie zur Visualisierung und Manipulation der Felder.
#
# Hauptfunktionen:
# - corr_function: Berechnet die Gaußsche Korrelationsfunktion zwischen zwei Punkten.
# - sigmoid: Wendet die Sigmoidfunktion auf einen Wert oder ein Array an.
# - create_meshgrid: Erstellt ein 2D-Gitter als Liste von Dictionaries mit Zellinformationen.
# - generate_random_field: Erzeugt ein Zufallsfeld auf dem Gitter.
# - print_grid: Gibt das Gitter mit den Feldwerten formatiert aus.
# - plot_grid: Visualisiert das Zufallsfeld als Heatmap.
# - update_cell: Aktualisiert den Wert einer bestimmten Zelle im Gitter.
#
# Typische Anwendungsfälle:
# - Simulation von Zufallsfeldern mit vorgegebener Kovarianzstruktur (z.B. für Geostatistik, Bildverarbeitung).
# - Visualisierung und Analyse von Zufallsfeldern.
# - Interaktive Manipulation einzelner Zellen im Gitter.
#
# Abhängigkeiten:
# - numpy
# - matplotlib
# - math
#
# Autor: toni
# Datum: 2025-06-02
# Last Modified: 2025-06-11
'''

from math import exp
import numpy as np

# meshgrid_30x30.py
#%%

def corr_function(x1, y1, x2, y2):
    """
        soll abhaengen von x1,y1 und x2 x2,y2
        muss positiv definit sein 
        Gaussglocke
    """
    sigma = 0.1  # Standard deviation
    dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return exp(-dist_sq / (2 * sigma ** 2))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#%%
x_vals = np.linspace(0, 1, 30)
y_vals = np.linspace(0, 1, 30)
X, Y = np.meshgrid(x_vals, y_vals)
# meshgrid erzeugt ein 2D-Gitter von Punkten
# X und Y sind 2D-Arrays, die die x- und y-Koordinaten der Punkte enthalten
# X[i, j] gibt die x-Koordinate des Punktes in der i-ten Zeile und j-ten Spalte an
# Y[i, j] gibt die y-Koordinate des Punktes in der i-ten Zeile und j-ten Spalte an

ravelled_X = X.ravel()
ravelled_Y = Y.ravel()

Cova_matrix = np.zeros((len(ravelled_X), len(ravelled_Y)))
for i in range(len(ravelled_X)):
    for j in range(len(ravelled_Y)):
        Cova_matrix[i, j] = corr_function(ravelled_X[i], ravelled_Y[i], ravelled_X[j], ravelled_Y[j])

#zufallsrasuch bild das unserer korrelationsmatrix entspricht

#%%
# Cholesky Zerlegung um Zufallsfeld zu generieren

# L = np.linalg.cholesky(Cova_matrix)
L = np.linalg.cholesky(Cova_matrix + 1e-10 * np.eye(Cova_matrix.shape[0]))

#%%
random_vector = np.random.randn(Cova_matrix.shape[0])
# Erzeuge ein zufälliges Feld, das der Kovarianzmatrix entspricht

random_field = L @ random_vector

random_field_image = random_field.reshape(X.shape)

random_field_image_sigmoid = sigmoid(random_field_image)

#%%

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.imshow(random_field_image_sigmoid, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Feldwert')
plt.title('Zufallsfeld entsprechend der Kovarianzmatrix')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

#%%

def create_meshgrid(rows=30, cols=30):
    grid = []
    for y in range(rows):
        row = []
        for x in range(cols):
            cell = {
                "x": x,
                "y": y,
                "occupied": False,
                "type": None
            }
            row.append(cell)
        grid.append(row)
    return grid

picutre1 = create_meshgrid()


#%%
np.random.seed(42)  # Setze den Zufallszahlengenerator für Reproduzierbarkeit

# %%
def generate_random_field(rows=30, cols=30):
    grid = create_meshgrid(rows, cols)
    for y in range(rows):
        for x in range(cols):
            grid[y][x]["value"] = np.random.rand()  # Zufälliger Wert zwischen 0 und 1
    return grid
random_field_grid = generate_random_field()
# %%
def print_grid(grid):
    for row in grid:
        print(" | ".join(f"{cell['value']:.2f}" for cell in row))
print_grid(random_field_grid)
# %%
def plot_grid(grid):
    rows = len(grid)
    cols = len(grid[0])
    values = np.array([[cell['value'] for cell in row] for row in grid])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(values, cmap='viridis', origin='lower', extent=[0, cols, 0, rows])
    plt.colorbar(label='Feldwert')
    plt.title('Zufallsfeld-Gitter')
    plt.xlabel('Spalten')
    plt.ylabel('Zeilen')
    plt.tight_layout()
    plt.show()
plot_grid(random_field_grid)
# %%
def update_cell(grid, x, y, value):
    if 0 <= x < len(grid[0]) and 0 <= y < len(grid):
        grid[y][x]["value"] = value
    else:
        print("Ungültige Koordinaten")
# Beispiel: Aktualisiere die Zelle bei (5, 5) auf den Wert 0.8
update_cell(random_field_grid, 5, 5, 0.8)
# %%
