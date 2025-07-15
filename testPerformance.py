import numpy as np
import time

# Erzeugt eine 900x900 Matrix mit Nullen in Python

matrix = [[0 for _ in range(900)] for _ in range(900)]

# Zufallsmatrix erzeugen
random_matrix = np.random.rand(900, 900)

random_matrix @= random_matrix.T  # Symmetrisieren der Matrix

identity_matrix = np.eye(900)
random_matrix += identity_matrix  # Einheitsmatrix erzeugen

# Inverse berechnen
# inverse_matrix = np.linalg.inv(random_matrix)


#
# Cholesky-Zerlegung der Matrix
start_time = time.time()
for i in range(900):
    random_matrix[i, i] += 1e-6  # Sicherstellen, dass die Matrix positiv definit ist
    cholesky_matrix = np.linalg.cholesky(random_matrix)
end_time = time.time()
time_taken = end_time - start_time
print(cholesky_matrix)
print(f"Zeit: {time_taken:.4f}")