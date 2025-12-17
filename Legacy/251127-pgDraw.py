from polyagamma import random_polyagamma
import numpy

# Parameter für den Draw
h = 1                 # PG(h, z) → h = 1

# beta_j^(0)
beta_j_0 = 0.0   # Startwert
pg = random_polyagamma()     # Sampler initialisieren


# Ziehe ein ω_j^(1) ~ PG(1, beta_j^(0)) = PG(1, 0)
omega_j_1 = random_polyagamma(1, beta_j_0)  # h=1, z=beta_j_0

print("ω_j^(1) =", omega_j_1)