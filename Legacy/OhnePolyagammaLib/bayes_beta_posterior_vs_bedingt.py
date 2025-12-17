
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages

# ========================
# Parameter und Daten
# ========================

# Wahrer Wert von beta (nur für Illustration)
beta_true = 0.0

# Einfacher Datenpunkt: x_i = 1
x_i = 1.0

# Kappa-Wert ergibt sich aus y_i = 1: kappa_i = y_i - 1/2 = 0.5
kappa_i = 0.5

# Beta-Werte auf einem Gitter (für Plot)
beta_grid = np.linspace(-5, 5, 500)

# ========================
# Posterior ohne Augmentation (logistische Regression)
# ========================

def logistic_likelihood(beta):
    psi = x_i * beta
    return np.exp(kappa_i * psi) / (1 + np.exp(psi))

# Unnormierte Posterior-Dichte berechnen
unnormalized_posterior = logistic_likelihood(beta_grid)

# Normierung (numerische Integration)
posterior = unnormalized_posterior / np.trapz(unnormalized_posterior, beta_grid)

# ========================
# Posterior mit Augmentation (bedingte Normalverteilung)
# ========================

# Simulierter Wert für omega_i aus PG-Verteilung (angenommen 1.0 für einfaches Beispiel)
omega_i = 1.0

# Bedingter Mittelwert und Varianz von beta
mean_beta = kappa_i / (omega_i * x_i)
var_beta = 1 / (omega_i * x_i**2)

# Dichte der bedingten Verteilung: Normalverteilung
conditional_normal = norm.pdf(beta_grid, loc=mean_beta, scale=np.sqrt(var_beta))

# ========================
# Plot und Export als PDF
# ========================

pdf_path = "posterior_vs_bedingte_verteilung_beta.pdf"
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(8, 5))
    plt.plot(beta_grid, posterior, label="Posterior $p(\\beta \\mid y)$ (logistic)", lw=2)
    plt.plot(beta_grid, conditional_normal, label="Bedingt: $p(\\beta \\mid \\omega, y)$ (Normal)", lw=2, linestyle='--')
    plt.title("Posterior vs. Bedingte Verteilung von $\\beta$")
    plt.xlabel("$\\beta$")
    plt.ylabel("Dichte")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"PDF gespeichert unter: {pdf_path}")
