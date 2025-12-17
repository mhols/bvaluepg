'''

Paper
Bayesian inference for logistic models using
Polya-Gamma latent variables (Polson, Scott and Windle, Jesse
Journal of the American Statistical Association (2013))

nach der Herleitung in Kapitel 2 folgt in Kapitel 3 die Beschreibung des
Gibbs-Samplers fuer logistische Regression.
Den baue ich hier nach.

Vorher einmal den Posterior fuer eine Poisson-Rate mit Gamma-Prior
aufbauen, wie in Kapitel 1 beschrieben.




Date: 2025-11-15
Author: toni
last modified: 2025-11-15
'''


import numpy as np
import polyagamma as pg



def generate_catalog_logistic(n, x, beta0=0.5, beta1=1.0, seed=None):
    """
    Erzeuge einen Katalog von n Zufallszahlen fuer eine logistische Regression,
    die nur von einem einzigen Input-Wert x abhaengt.

    Parameters
    ----------
    n : int
        Anzahl der zu generierenden Zufallswerte.
    x : float
        Eingabewert.
    beta0 : float
        Intercept.
    beta1 : float
        Steigung.
    seed : int or None
        Optionaler Seed fuer Reproduzierbarkeit.

    Returns
    -------
    np.ndarray
        Ein Array aus n Bernoulli( p ) Zufallszahlen.
    """
    if seed is not None:
        np.random.seed(seed)

    # Logistische Funktion
    p = 1 / (1 + np.exp(-(beta0 + beta1 * x)))

    # Bernoulli-Samples erzeugen
    y = np.random.binomial(1, p, size=n)
    return y

# Aufruf
n = 1000
x = 2.0
catalog = generate_catalog_logistic(n, x, beta0=0.0, beta1=1.0, seed=42)
print(catalog)

import numpy as np
import matplotlib.pyplot as plt

def generate_photon_catalog_multinomial(
    N=10_000,
    nx=30,
    ny=30,
    sigma=0.5,
    seed=None
):
    """
    Erzeuge einen Katalog von Photonenzaehlungen pro Pixel auf einem nx×ny-Gitter
    mittels eines Multinomial-Modells.

    N     : Gesamtzahl der Photonen
    nx,ny : Anzahl Pixel in x- und y-Richtung
    sigma : Breite des glatten Feldes (Gauss-Fleck)
    seed  : Optionaler Zufallsseed
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Gitterkoordinaten
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # 2) Glattes Intensitaetsfeld (z.B. 2D-Gauß um (0,0))
    intensity = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # Numerische Stabilisierung: keine exakt 0
    intensity = np.clip(intensity, 1e-12, None)

    # 3) Normierung zu Wahrscheinlichkeiten
    p = intensity / intensity.sum()

    # 4) Multinomial-Ziehung
    counts_flat = np.random.multinomial(N, p.ravel())

    # 5) Zurueck in 2D-Form (nx×ny)
    counts = counts_flat.reshape(nx, ny)

    return counts, p

N = 10_000
catalog_counts, p_field = generate_photon_catalog_multinomial(N=N, nx=30, ny=30, seed=42)
print(catalog_counts.shape)  # (30, 30)



def plot_photon_catalog(catalog):
    """
    Plot eines Photonenkatalogs (nx × ny Array).
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(catalog, origin="lower", interpolation="nearest")
    # Gitterlinien alle 5 Pixel
    nx, ny = catalog.shape
    plt.xticks(np.arange(0, ny, 5))
    plt.yticks(np.arange(0, nx, 5))
    plt.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.colorbar(label="Photonen pro Pixel")
    plt.title("Photonen-Katalog")
    plt.xlabel("x-Pixel")
    plt.ylabel("y-Pixel")
    plt.tight_layout()
    plt.show()

# Plotten des Katalogs
plot_photon_catalog(catalog_counts)  #nur Katalog ist 1-dimensional
plot_photon_catalog(p_field)

# Data Augmentation via Polya-Gamma Latent Variables for Logistic Regression
# Kapitel 3 des Papers

# als Feature Vektor nur ein Input-Wert x_i  den pixel index

def create_design_matrix_index_only(N):
    """
    Erzeuge eine Design-Matrix X (N x 1),
    wobei jedes Pixel nur durch seinen Index beschrieben wird.

    Parameter
    ---------
    N : int
        Anzahl der Pixel (z.B. 900 für 30x30)

    Returns
    -------
    X : np.ndarray
        Design-Matrix der Form (N, 1)
        mit X[i,0] = i  (Pixelindex)
    """
    indices = np.arange(N)
    X = indices.reshape(-1, 1)
    return X

# erstmal kein Intercept beta0 gleich 0

'''
repeat:
    1. Ziehe ω_i ~ PG(1, x_i^T β) für i = 1..N
    2. Berechne κ_i = y_i - 1/2  für i = 1..N
    3. Bestimme X die Design-Matrix (N x p)
    4. Bestimme Ω = diag( ω_1, ..., ω_N )
    5. Bestimme b und B die Prior-Parameter für β:
       β ~ N(b, B)
    6. Berechne V = (X^T Ω X + B^-1)^-1
       und m = V( X^T κ + B^-1 b )
    7. Ziehe β ~ N(m, V)
    8. Wiederhole mit neuem β bei Schritt 1           
'''

X = create_design_matrix_index_only(900)



def draw_polya_gamma(b, c, size=None, random_state=None, method="devroye"):
    """
    Ziehe Zufallszahlen aus der Polya-Gamma-Verteilung PG(b, c)
    mithilfe der polyagamma-Bibliothek.

    Parameters
    ----------
    b : float oder np.ndarray
        Shape-Parameter der PG-Verteilung (oft ganzzahlig, z.B. 1 oder n_i).
    c : float oder np.ndarray
        Tilt-Parameter der PG-Verteilung (typischerweise x_i^T beta).
    size : int oder Tuple oder None
        Ausgabe-Shape. Wenn None, wird das Shape aus b/c durch Broadcasting bestimmt.
    random_state : np.random.Generator, np.random.RandomState oder None
        Optionaler Zufallsgenerator fuer Reproduzierbarkeit.
    method : {"hybrid", "devroye", "alternate", "gamma", "saddle"}
        Sampling-Methode; Standard ist der hybride Sampler.

    Returns
    -------
    np.ndarray oder float
        Ziehung(en) aus PG(b, c).
    """
    return pg.random_polyagamma(b, c, size=size, method=method, random_state=random_state)

print(X.shape)  # (900, 1)
# Beispielhafte Prior-Parameter
b = np.array([0.0])        # Prior-Mittelwert
B = np.array([[1.0]])      # Prior-Kovarianzmatrix
B_inv = np.linalg.inv(B)
num_iterations = 1
# Initialisiere β
beta = np.array([0.0]) # Startwert fuer β


# Beispielhafte Beobachtungen y (Bernoulli) (haben grad nichts mit dem Photonenkatalog zu tun)
y = np.random.binomial(1, 0.5, size=X.shape[0])  # Zufallswerte fuer y

for iteration in range(num_iterations):
    # 1. Ziehe ω_i ~ PG(1, x_i^T β)
    # Hier verwenden wir die echte PG-Ziehung aus der polyagamma-Bibliothek
    omega = draw_polya_gamma(1.0, (X @ beta))
    
    # 2. Berechne κ_i = y_i - 1/2
    kappa = y - 0.5
    
    # 4. Bestimme Ω = diag( ω_1, ..., ω_N )
    Omega = np.diag(omega)
    
    # 6. Berechne V und m
    V_inv = X.T @ Omega @ X + B_inv
    V = np.linalg.inv(V_inv)
    m = V @ (X.T @ kappa + B_inv @ b)
    
    # 7. Ziehe β ~ N(m, V)
    beta = np.random.multivariate_normal(mean=m.flatten(), cov=V)
    
    # Optional: Ausgabe des aktuellen β
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: β = {beta}")

# ok so bekomme ich nur ein beta raus, das den Pixelindex beschreibt
# klar meine Design-Matrix hat ja nur eine Spalte
# wenn ein beta pro Pixel kommen soll, brauche ich eine Design-Matrix mit 900 Spalten
def create_design_matrix_one_hot(N):
    """
    endlich mal wieder one-hot encoding
    Erzeuge eine Design-Matrix X (N x N),
    wobei jedes Pixel durch einen One-Hot-Vektor beschrieben wird.

    Parameter
    ---------
    N : int
        Anzahl der Pixel (z.B. 900 für 30x30)

    Returns
    -------
    X : np.ndarray
        Design-Matrix der Form (N, N)
        mit X[i,j] = 1 wenn i == j sonst 0
    """
    X = np.eye(N)
    return X

X_one_hot = create_design_matrix_one_hot(900)
print(X_one_hot.shape)  # (900, 900)
# Initialisiere β
beta_one_hot = np.zeros(900) # Startwert fuer β
num_iterations = 1
for iteration in range(num_iterations):
    # 1. Ziehe ω_i ~ PG(1, x_i^T β)
    omega = draw_polya_gamma(1.0, (X_one_hot @ beta_one_hot))
    
    # 2. Berechne κ_i = y_i - 1/2
    kappa = y - 0.5
    
    # 4. Bestimme Ω = diag( ω_1, ..., ω_N )
    Omega = np.diag(omega)
    
    # 6. Berechne V und m
    V_inv = X_one_hot.T @ Omega @ X_one_hot + B_inv
    V = np.linalg.inv(V_inv)
    m = V @ (X_one_hot.T @ kappa + B_inv @ b)
    
    # 7. Ziehe β ~ N(m, V)
    beta_one_hot = np.random.multivariate_normal(mean=m.flatten(), cov=V)
    
    # Optional: Ausgabe des aktuellen β
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: β (one-hot) = {beta_one_hot}")

#%%

'''
naechste Beispiel
ich habe eine Reihe von Gluehbirnen, die entweder leuchten oder nicht
ich werde das einmal als logistische Regression modellieren
und den Polya-Gamma Gibbs-Sampler dafuer bauen
'''
import numpy as np
import matplotlib.pyplot as plt

#Funktion baue Zufallig meine Gluehbirnen-Daten
def generate_bulb_data(n_bulbs=100, prob_on=0.3, seed=None):
    """
    Erzeuge Zufallsdaten fuer Gluehbirnen, die entweder an (1) oder aus (0) sind.

    Parameters
    ----------
    n_bulbs : int
        Anzahl der Gluehbirnen.
    prob_on : float
        Wahrscheinlichkeit, dass eine Gluehbirne an ist.
    seed : int or None
        Optionaler Seed fuer Reproduzierbarkeit.

    Returns
    -------
    np.ndarray
        Ein Array aus n_bulbs Binomial(1, prob_on) Zufallszahlen.
    """
    if seed is not None:
        np.random.seed(seed)

    # Bernoulli-Samples erzeugen
    bulb_states = np.random.binomial(1, prob_on, size=n_bulbs)
    return bulb_states
#aufruf
n_bulbs = 100
prob_on = 0.3
bulb_data = generate_bulb_data(n_bulbs=n_bulbs, prob_on=prob_on, seed=42)
print(bulb_data)

# Posterior fuer logistische Regression mit likelihood aus Gluehbirnen-Daten
def likelihood_logistic(beta, X, y):
    """
    Berechne die Likelihood fuer logistische Regression.

    Parameters
    ----------
    beta : np.ndarray
        Koeffizientenvektor der logistischen Regression.
    X : np.ndarray
        Design-Matrix (N x p).
    y : np.ndarray
        Beobachtungsvektor (N,).

    Returns
    -------
    float
        Likelihood-Wert.
    """
    linear_pred = X @ beta
    p = 1 / (1 + np.exp(-linear_pred))
    likelihood = np.prod(p**y * (1 - p)**(1 - y))
    return likelihood

