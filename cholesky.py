#%%

from math import exp
import numpy as np

# mechgrid_30x30.py
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

def create_mechgrid(rows=30, cols=30):
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

picutre1 = create_mechgrid()


#%%

