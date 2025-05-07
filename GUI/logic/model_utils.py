import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy.random import normal, gamma
import matplotlib.pyplot as plt
import io
import base64


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data(n, beta):
    X = np.random.randn(n, len(beta))
    logits = X @ np.array(beta)
    probs = sigmoid(logits)
    y = np.random.binomial(1, probs)
    return X, y

def logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model.coef_[0]


# Polya-Gamma Sampler (Approximativ)
def polya_gamma_sampler(b, c, size=1):
    # Simple Approximation: PG(b, c) ≈ Gamma(b, 1) / (2 * c^2)
    # (echte Sampler sind komplexer – z. B. über Devroye oder BayesLogit)
    c = np.clip(c, 1e-6, None)  # Vermeide Division durch 0
    return gamma(shape=b, scale=1.0, size=size) / (2 * c**2)

def polya_gamma_logreg(X, y, num_iter=10):
    n, d = X.shape
    beta = np.zeros(d)
    Xt = X.T

    for _ in range(num_iter):
        eta = X @ beta
        omega = polya_gamma_sampler(1.0, eta, size=n)

        # Posterior: V = (XᵀΩX)^(-1), m = V Xᵀ (y - 0.5)
        Omega = np.diag(omega)
        V_inv = Xt @ Omega @ X
        V = np.linalg.inv(V_inv + np.eye(d) * 1e-6)  # Regularisierung
        m = V @ Xt @ (y - 0.5)

        beta = normal(loc=m, scale=np.sqrt(np.diag(V)))

    return beta

def gibbs_sampler(X, y, num_samples=1000, burn_in=100):
    n, d = X.shape
    Xt = X.T
    beta = np.zeros(d)
    samples = []

    for i in range(num_samples + burn_in):
        eta = X @ beta
        omega = polya_gamma_sampler(1.0, eta, size=n)

        Omega = np.diag(omega)
        V_inv = Xt @ Omega @ X
        V = np.linalg.inv(V_inv + np.eye(d) * 1e-6)
        m = V @ Xt @ (y - 0.5)

        beta = normal(loc=m, scale=np.sqrt(np.diag(V)))
        
        if i >= burn_in:
            samples.append(beta.copy())

    return np.array(samples)


def plot_gibbs_traces(samples):
    fig, axs = plt.subplots(samples.shape[1], 1, figsize=(6, 3 * samples.shape[1]))
    if samples.shape[1] == 1:
        axs = [axs]  # für 1D-Fall
    for i in range(samples.shape[1]):
        axs[i].plot(samples[:, i])
        axs[i].set_title(f"Trace von Beta[{i}]")
    return fig_to_base64(fig)

def plot_gibbs_hist(samples):
    fig, axs = plt.subplots(samples.shape[1], 1, figsize=(6, 3 * samples.shape[1]))
    if samples.shape[1] == 1:
        axs = [axs]
    for i in range(samples.shape[1]):
        axs[i].hist(samples[:, i], bins=30, density=True)
        axs[i].set_title(f"Posterior von Beta[{i}]")
    return fig_to_base64(fig)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy.random import normal, gamma
import matplotlib.pyplot as plt
import io
import base64

# Hilfsfunktionen -----

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Daten erzeugen -----

def generate_data(n, beta):
    X = np.random.randn(n, len(beta))
    logits = X @ np.array(beta)
    probs = sigmoid(logits)
    y = np.random.binomial(1, probs)
    return X, y


# Klassische logistische Regression -----

def logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model.coef_[0]


# Polya-Gamma Data Augmentation -----

def polya_gamma_sampler(b, c, size=1):
    # Näherung: PG(b, c) ≈ Gamma(b, 1) / (2 * c^2)
    c = np.clip(c, 1e-6, None)  # Vermeide Division durch 0
    return gamma(shape=b, scale=1.0, size=size) / (2 * c**2)


def polya_gamma_logreg(X, y, num_iter=10):
    n, d = X.shape
    beta = np.zeros(d)
    Xt = X.T

    for _ in range(num_iter):
        eta = X @ beta
        omega = polya_gamma_sampler(1.0, eta, size=n)

        Omega = np.diag(omega)
        V_inv = Xt @ Omega @ X
        V = np.linalg.inv(V_inv + np.eye(d) * 1e-6)  # Regularisierung
        m = V @ Xt @ (y - 0.5)

        beta = normal(loc=m, scale=np.sqrt(np.diag(V)))

    return beta


# Gibbs Sampler -----

def gibbs_sampler(X, y, num_samples=1000, burn_in=100):
    n, d = X.shape
    Xt = X.T
    beta = np.zeros(d)
    samples = []

    for i in range(num_samples + burn_in):
        eta = X @ beta
        omega = polya_gamma_sampler(1.0, eta, size=n)

        Omega = np.diag(omega)
        V_inv = Xt @ Omega @ X
        V = np.linalg.inv(V_inv + np.eye(d) * 1e-6)
        m = V @ Xt @ (y - 0.5)

        beta = normal(loc=m, scale=np.sqrt(np.diag(V)))

        if i >= burn_in:
            samples.append(beta.copy())

    return np.array(samples)


# Plotly: Trace- und Histogrammdaten -----

def plotly_trace_data(samples):
    traces = []
    d = samples.shape[1]
    for i in range(d):
        traces.append({
            'x': list(range(len(samples))),
            'y': samples[:, i].tolist(),
            'mode': 'lines',
            'name': f'beta[{i}]'
        })
    return traces


def plotly_hist_data(samples):
    traces = []
    d = samples.shape[1]
    for i in range(d):
        traces.append({
            'x': samples[:, i].tolist(),
            'type': 'histogram',
            'name': f'beta[{i}]',
            'opacity': 0.75
        })
    return traces