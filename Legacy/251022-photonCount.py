'''
neue Version mit grösserer Abweichung
Damit die Variation im Poissonprozess deutlicher wird.
erzeuge schachbrettmuster 30 mal 30 mit 5*5 feldern
jede zelle 5x5 pixel
jede zelle hat einen basiswert + abweichung*pattern
'''

#%% Imports

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import polyagamma as polyga

#%% create pattern
# --- Parameter ---
rows, cols = 30, 30
poisson_lambda = 20 # Erwartungswert pro Feld
noise_std = 2
np.random.seed(42)

x = np.indices((rows, cols)).sum(axis=0) % 2
pattern = np.where(x == 0, 0.3, 1.0)


for i in range(rows):
    ii=(i // 5)%2
    for j in range(cols):
        jj=(j // 5)%2
        if (ii + jj)%2 == 0:
            pattern[i, j] = 0
        else:
            pattern[i, j] = 1

#plot pattern
plt.figure(figsize=(6, 5))
plt.imshow(pattern, cmap="gray")
plt.title("Schachbrettmuster")
plt.xlabel("Spalte")
plt.ylabel("Zeile")
plt.colorbar(label="Musterwert")
plt.show()  

#%%
# Poisson-Verteilung base rate + bild mit Schachbrettmuster
#base_rate = poisson_lambda * (1 + pattern)

base_rate = 100
deviation = 40  # darf nicht zu klein sein

feld=base_rate+deviation*pattern
# plot feld
plt.figure(figsize=(6, 5))
plt.imshow(feld, cmap="viridis")
plt.title("Photonenrate pro Feld (λ) mit Abweichung")
plt.xlabel("Spalte")
plt.ylabel("Zeile")
plt.colorbar(label="Photonenrate λ")
plt.show()
#%%
# ergebnis ist gleich np.zeros like feld

ergebnis = np.zeros_like(feld)
for i in range(rows):
    for j in range(cols):
        ergebnis[i,j]=np.random.poisson(feld[i,j],size=1)
# plot ergebnis
plt.figure(figsize=(6, 5))
plt.imshow(ergebnis, cmap="viridis")
plt.title("Anzahl detektierter Photonen pro Feld")
plt.xlabel("Spalte")
plt.ylabel("Zeile")

# %%
# sample posterior aus ergebnis
# iteration mindestens 100 mal

#%%
# 1. prior definieren

# --- Prior-Optionen (eine aktiv lassen, zwei auskommentieren) ---
# Wähle EINEN Prior, indem du die entsprechende Zeile für prior_type aktiv lässt
# und die anderen prior_type-Zeilen auskommentierst.

# (A) Flat Prior (improper) auf λ>0  => Posterior pro Zelle: Gamma(y+1, 1)
prior_type = "flat"  # "flat" | "gauss" | "gamma"
# Hinweis: Für den Flat-Prior verwenden ich später direkt alpha_post = y + 1, beta_post = 1.

# (B) Gauss-Prior auf log λ: η = log(λ) ~ N(mu0, sigma0^2)
# prior_type = "gauss"
mu0 = 4.5    # Vorwissen über log(λ)
sigma0 = 1.0 # Streuung des Vorwissens

# (C) Gamma-Prior auf λ: λ ~ Gamma(alpha0, beta0)  (beta0 = Rate)
# prior_type = "gamma"
alpha0 = 1.0
beta0  = 1.0



#%%
# 2. Modell definierten 
# --- Modell-Optionen (eine aktiv lassen, die andere auskommentieren) ---
# Wähle EIN Modell, indem du die Zeile für model_type aktiv lässt
# und die alternative model_type-Zeile auskommentierst.

# (A) Poisson-Regression für Zählungen: y ~ Poisson(λ), log(λ) = β0 + β1 * x
model_type = "poisson"  # "poisson" | "logistic"

# Prädiktor und Response für Poisson
X = pattern.reshape(-1, 1)
y_vec = ergebnis.reshape(-1)
# Design-Matrix mit Intercept: η = Xβ mit X = [1, x]
X_design = np.column_stack([np.ones_like(X), X])

# (B) Logistische Regression (optional): y_bin ~ Bernoulli(p), logit(p) = β0 + β1 * x
# model_type = "logistic"
# # Binäre Zielvariable (Photon vorhanden ja/nein)
# y_vec = ( ergebnis > 0 ).astype(int).reshape(-1)
# # Prädiktor bleibt das Schachbrettmuster
# X = pattern.reshape(-1, 1)
# # Design-Matrix mit Intercept
# X_design = np.column_stack([np.ones_like(X), X])

# Hinweis: In Schritt (3) berechnen wir die passende Likelihood abhängig von model_type.
#          In Schritt (4) folgt das (Bayes-)Update bzw. eine MLE-Schätzung.


#%%
# 3. Likelihood berechnen

# --- Likelihood-Funktionen (up to additive constant) ---
# Hinweis: Für Vergleiche in β genügt die Log-Likelihood bis auf additive Konstanten.
# Der Poisson-Term -log(y!) ist β-unabhängig und wird weggelassen.

import numpy as _np  # Alias nur lokal hier benutzt

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def loglik_poisson(beta, X, y):
    """Log-Likelihood (bis auf additive Konstante) für Poisson-Regression.
    beta: (p,) Parametervektor; X: (n,p); y: (n,)
    L(β) = sum( y * η - exp(η) ),  η = Xβ
    """
    eta = X @ beta
    lam = np.exp(eta)
    return float(np.sum(y * eta - lam))


def loglik_logistic(beta, X, y):
    """Bernoulli/Logit Log-Likelihood (exakt, ohne Konstante):
    L(β) = sum( y*η - log(1+exp(η)) ),  η = Xβ
    """
    eta = X @ beta
    # numerisch stabil: -log(1+exp(η)) = -softplus(η)
    # np.log1p(np.exp(eta)) ist ok bei moderaten Werten
    return float(np.sum(y * eta - np.log1p(np.exp(eta))))

# --- Diagnose: 1D-Profil der Log-Likelihood über β1 bei fixiertem β0 ---
# Wir zeichnen die Log-Likelihood als Funktion von β1, um das Verhalten zu sehen.

if model_type == "poisson":
    # sinnvoller Startwert für β0: log(mean(y))
    beta0_init = np.log(max(1e-8, float(np.mean(y_vec))))
    b1_grid = np.linspace(-3.0, 3.0, 241)
    vals = []
    for b1 in b1_grid:
        beta = np.array([beta0_init, b1])
        vals.append(loglik_poisson(beta, X_design, y_vec))
    plt.figure(figsize=(6,4))
    plt.plot(b1_grid, vals)
    plt.title("Log-Likelihood-Profil in β1 (Poisson-Regression)")
    plt.xlabel("β1")
    plt.ylabel("log L(β0, β1 | y)  (bis Konst.)")
    plt.tight_layout()
    plt.show()

elif model_type == "logistic":
    # Startwert für β0: logit(mean(y)) (Korrektur für 0/1-Randfälle)
    p_bar = np.clip(np.mean(y_vec), 1e-6, 1-1e-6)
    beta0_init = np.log(p_bar / (1.0 - p_bar))
    b1_grid = np.linspace(-8.0, 8.0, 241)
    vals = []
    for b1 in b1_grid:
        beta = np.array([beta0_init, b1])
        vals.append(loglik_logistic(beta, X_design, y_vec))
    plt.figure(figsize=(6,4))
    plt.plot(b1_grid, vals)
    plt.title("Log-Likelihood-Profil in β1 (Logistische Regression)")
    plt.xlabel("β1")
    plt.ylabel("log L(β0, β1 | y)")
    plt.tight_layout()
    plt.show()





#%%
# 4. posterior berechnen

# --- Posterior-Bestimmung (ohne Blackbox, möglichst „eigener“ Code) ---
# Wir behandeln zwei Fälle je nach model_type:
#  (A) Poisson: Posterior über λ pro Zelle
#      - prior_type = "gamma"  => conjugate: Gamma(alpha0+y, beta0+1)
#      - prior_type = "flat"   => Gamma(y+1, 1)
#      - prior_type = "gauss"  => Gauss-Prior auf η=log λ; Laplace-Approx.
#  (B) Logistic: Posterior über β per Laplace (Normal-Prior auf β)

if model_type == "poisson":
    y = ergebnis.astype(float)

    if prior_type == "gamma":
        alpha_post = alpha0 + y
        beta_post  = beta0 + 1.0
        post_mean_lambda = alpha_post / beta_post
        post_mode_lambda = (alpha_post - 1.0) / beta_post
        post_mode_lambda = np.where(alpha_post > 1, post_mode_lambda, 0.0)
        # einfache Unsicherheitsmetrik (Varianz der Gamma): alpha / beta^2
        post_var_lambda  = alpha_post / (beta_post ** 2)

    elif prior_type == "flat":
        alpha_post = 1.0 + y
        beta_post  = 1.0
        post_mean_lambda = alpha_post / beta_post
        post_mode_lambda = np.maximum(alpha_post - 1.0, 0.0) / beta_post
        post_var_lambda  = alpha_post / (beta_post ** 2)

    elif prior_type == "gauss":
        # Laplace-Approx in η = log λ: maximize  y*η - exp(η) - (η-mu0)^2/(2σ^2)
        y_flat = y.ravel()
        # Start bei log(y+eps)
        eta = np.log(np.maximum(y_flat, 1e-8))
        inv_sigma2 = 1.0 / (sigma0 ** 2)
        for _ in range(12):
            exp_eta = np.exp(eta)
            g = y_flat - exp_eta - (eta - mu0) * inv_sigma2          # Gradient
            H = -exp_eta - inv_sigma2                                # Hessian (negativ)
            eta = eta - g / H                                        # Newton-Schritt
        var_eta = -1.0 / H                                           # Varianz (Skalar je Zelle)
        mean_eta = eta
        # Posterior-Momente von λ unter Lognormal-Approx
        post_mean_lambda = np.exp(mean_eta + 0.5 * var_eta).reshape(y.shape)
        post_mode_lambda = np.exp(mean_eta - var_eta).reshape(y.shape)  # Mode des Lognormals
        # Delta-Regel für Varianz in λ
        post_var_lambda  = (np.exp(var_eta) - 1.0) * np.exp(2*mean_eta + var_eta)
        post_var_lambda  = post_var_lambda.reshape(y.shape)
    else:
        raise ValueError("Unbekannter prior_type für Poisson: {}".format(prior_type))

    # Visualisierung Posterior-Mittelwerte λ
    plt.figure(figsize=(6,5))
    plt.imshow(post_mean_lambda, cmap="viridis")
    plt.title(f"Posterior E[λ|D] — Modell: Poisson, Prior: {prior_type}")
    plt.xlabel("Spalte"); plt.ylabel("Zeile")
    plt.colorbar(label="E[λ | Daten]")
    plt.tight_layout(); plt.show()

    # Optional: Posterior-Unsicherheit (Std-Abw.)
    plt.figure(figsize=(6,5))
    plt.imshow(np.sqrt(post_var_lambda), cmap="viridis")
    plt.title(f"Posterior Std(λ|D) — Modell: Poisson, Prior: {prior_type}")
    plt.xlabel("Spalte"); plt.ylabel("Zeile")
    plt.colorbar(label="Std[λ | Daten]")
    plt.tight_layout(); plt.show()

elif model_type == "logistic":
    # Bayes-Logistische Regression per Laplace-Approximation (eigener Newton/IRLS)
    # Prior auf β: N(0, τ^2 I)
    tau2 = 10.0  # relativ schwacher, aber proper Prior
    n, p = X_design.shape
    beta = np.zeros(p)  # Startwert

    def logpost_and_grad_hess(beta, X, y, tau2):
        eta = X @ beta
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        # Log-Likelihood + Log-Prior
        ll = np.sum(y * eta - np.log1p(np.exp(eta)))
        lp = -0.5 * np.sum(beta**2) / tau2
        logpost = ll + lp
        # Gradient: X^T (y - p) - beta/tau2
        grad = X.T @ (y - p_hat) - beta / tau2
        # Hessian: - (X^T W X + I/tau2)
        W = p_hat * (1.0 - p_hat)
        # explizit bauen (p klein)
        XT_W = X.T * W
        H = -(XT_W @ X) - np.eye(p) / tau2
        return logpost, grad, H

    # Newton-Iterationen
    for _ in range(25):
        _, grad, H = logpost_and_grad_hess(beta, X_design, y_vec, tau2)
        # Löse H * step = grad  => step = H^{-1} grad
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            # leichte Regularisierung, falls H schlecht konditioniert
            H_reg = H - 1e-6 * np.eye(H.shape[0])
            step = np.linalg.solve(H_reg, grad)
        beta_new = beta - step
        if np.max(np.abs(step)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    # Laplace-Posterior: β | D ≈ N(β_hat, Σ),  Σ = (X^T W X + I/tau2)^{-1}
    _, _, H = logpost_and_grad_hess(beta, X_design, y_vec, tau2)
    Sigma = -np.linalg.inv(H)

    # Zusammenfassung
    print("Posterior (Laplace) für logistische Regression:")
    print("beta_hat =", beta)
    print("Sigma =\n", Sigma)

    # Visualisierung: vorhergesagte Wahrscheinlichkeiten p(x)=sigmoid(Xβ)
    eta_hat = X_design @ beta
    p_hat = 1.0 / (1.0 + np.exp(-eta_hat))
    p_img = p_hat.reshape(pattern.shape)

    plt.figure(figsize=(6,5))
    plt.imshow(p_img, cmap="viridis")
    plt.title("Vorhergesagte P(Y=1|x) — Logistische Regression (Bayes Laplace)")
    plt.xlabel("Spalte"); plt.ylabel("Zeile")
    plt.colorbar(label="p̂")
    plt.tight_layout(); plt.show()

else:
    raise ValueError("Unbekannter model_type: {}".format(model_type))

#%% 4.1 Posterior berechnen effizienter
# --- Effizient/mit Bibliotheken: Posterior (aggregiert) neu berechnen ---
# Ziel: den Abschnitt 4 mit mehr vorgefertigten Bibliotheken nachbauen.
#  - Poisson + (Gamma/Flat): scipy.stats.gamma für Momente/Sampling
#  - Poisson + Gauss(log λ): scipy.optimize.minimize (BFGS) für Laplace
#  - Logistic: statsmodels GLM Binomial für β̂ und Kovarianz, dann Laplace-Bayes mit Ridge-Prior

# Versuche, benötigte Bibliotheken zu laden (mit Fallback-Hinweisen)
try:
    import scipy.stats as st
    from scipy.optimize import minimize
    from scipy.special import expit
except Exception as _e:
    print("Hinweis: scipy wird empfohlen (stats/optimize/special). Fehler:", _e)
    # Wir laufen dennoch weiter; einige Teile funktionieren ohne scipy nicht.
    st = None
    minimize = None
    def expit(z):
        return 1.0/(1.0+np.exp(-z))

try:
    import statsmodels.api as sm
except Exception as _e:
    sm = None
    print("Hinweis: statsmodels wird für die logistische Regression empfohlen. Fehler:", _e)

if model_type == "poisson":
    # Aggregation nach Klassen (Schachbrett 0/1)
    y = ergebnis.astype(float)
    mask0 = (pattern == 0)
    mask1 = ~mask0
    n0, n1 = int(mask0.sum()), int(mask1.sum())
    y0_sum, y1_sum = float(y[mask0].sum()), float(y[mask1].sum())

    if prior_type in {"gamma", "flat"}:
        # Konjugierte Posterioren klassenweise
        if prior_type == "gamma":
            a0, b0 = alpha0 + y0_sum, beta0 + n0  # beta = Rate
            a1, b1 = alpha0 + y1_sum, beta0 + n1
        else:  # flat
            a0, b0 = 1.0 + y0_sum, 1.0
            a1, b1 = 1.0 + y1_sum, 1.0
        if st is not None:
            # scipy.stats.gamma parametrisiert mit shape=a, scale=1/rate
            lam0_mean = st.gamma.mean(a=a0, scale=1.0/b0)
            lam1_mean = st.gamma.mean(a=a1, scale=1.0/b1)
            lam0_var  = st.gamma.var(a=a0, scale=1.0/b0)
            lam1_var  = st.gamma.var(a=a1, scale=1.0/b1)
        else:
            lam0_mean = a0 / b0; lam1_mean = a1 / b1
            lam0_var  = a0 / (b0**2); lam1_var  = a1 / (b1**2)

    elif prior_type == "gauss":
        # Laplace für η_k = log λ_k via scipy.optimize.minimize (BFGS)
        if minimize is None:
            raise RuntimeError("Für prior_type='gauss' wird scipy.optimize.minimize benötigt.")

        def neg_logpost_eta(eta, y_sum, n, mu0, sigma0):
            # negative Logposterior: -( y*η - n*exp(η) - (η-mu0)^2/(2σ^2) )
            return -(y_sum*eta - n*np.exp(eta) - 0.5*((eta-mu0)/sigma0)**2)

        def laplace_class(y_sum, n, mu0, sigma0):
            # Start bei log(mean)
            eta0 = np.log(max(y_sum / max(n, 1), 1e-8))
            res = minimize(neg_logpost_eta, x0=np.array([eta0]), args=(y_sum, n, mu0, sigma0),
                           method="BFGS")
            eta_hat = float(res.x)
            # BFGS gibt approx. inverse Hessian (Kovarianz) in res.hess_inv
            if hasattr(res, "hess_inv"):
                var_eta = float(np.atleast_2d(res.hess_inv)[0,0])
            else:
                # numerischer Rückfall: zweite Ableitung am Optimum
                eps = 1e-5
                fpp = (neg_logpost_eta(eta_hat+eps, y_sum, n, mu0, sigma0)
                       - 2*neg_logpost_eta(eta_hat, y_sum, n, mu0, sigma0)
                       + neg_logpost_eta(eta_hat-eps, y_sum, n, mu0, sigma0)) / (eps**2)
                var_eta = 1.0 / max(fpp, 1e-12)
            # Lognormal-Momente
            lam_mean = np.exp(eta_hat + 0.5*var_eta)
            lam_var  = (np.exp(var_eta)-1.0) * np.exp(2*eta_hat + var_eta)
            return lam_mean, lam_var

        lam0_mean, lam0_var = laplace_class(y0_sum, n0, mu0, sigma0)
        lam1_mean, lam1_var = laplace_class(y1_sum, n1, mu0, sigma0)

    else:
        raise ValueError(f"Unbekannter prior_type={prior_type} für Poisson in 4.1")

    # Zurück auf Bildgröße mappen
    post_mean_lambda_agg = np.where(mask0, lam0_mean, lam1_mean)
    post_std_lambda_agg  = np.where(mask0, np.sqrt(lam0_var), np.sqrt(lam1_var))

    plt.figure(figsize=(6,5))
    plt.imshow(post_mean_lambda_agg, cmap="viridis")
    plt.title(f"(lib) Posterior E[λ|D] — Poisson, Prior: {prior_type}")
    plt.xlabel("Spalte"); plt.ylabel("Zeile")
    plt.colorbar(label="E[λ|D]")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,5))
    plt.imshow(post_std_lambda_agg, cmap="viridis")
    plt.title(f"(lib) Posterior Std(λ|D) — Poisson, Prior: {prior_type}")
    plt.xlabel("Spalte"); plt.ylabel("Zeile")
    plt.colorbar(label="Std[λ|D]")
    plt.tight_layout(); plt.show()

elif model_type == "logistic":
    # Mit statsmodels GLM (Binomial) β̂ bestimmen; dann Laplace-Bayes mit Ridge-Prior
    if sm is None:
        raise RuntimeError("Für 'logistic' wird statsmodels benötigt (pip install statsmodels)")

    # Falls versehentlich noch y_vec aus Poisson stammt, hier sicherstellen:
    if y_vec.ndim == 1 and y_vec.max() > 1:
        y_bin = (ergebnis > 0).astype(int).reshape(-1)
    else:
        y_bin = y_vec

    glm_bin = sm.GLM(y_bin, X_design, family=sm.families.Binomial())
    res = glm_bin.fit()
    beta_mle = res.params
    # Kovarianz der MLEs (observed information inverse)
    Sigma_mle = res.cov_params()

    # Ridge-Prior β ~ N(0, τ^2 I) → Posterior approx: N(β_map, Σ_post)
    tau2 = 10.0
    Prec_prior = np.eye(len(beta_mle)) / tau2
    Prec_mle   = np.linalg.inv(Sigma_mle)
    Prec_post  = Prec_mle + Prec_prior
    Sigma_post = np.linalg.inv(Prec_post)
    # MAP mit Prior-Mean 0: β_map = Σ_post (Prec_mle β_mle)
    beta_map = Sigma_post @ (Prec_mle @ beta_mle)

    print("(4.1 lib) Logistic: beta_mle=", beta_mle)
    print("(4.1 lib) Logistic: beta_map=", beta_map)
    print("(4.1 lib) Logistic: Sigma_post=\n", Sigma_post)

    p_hat = expit(X_design @ beta_map)
    p_img = p_hat.reshape(pattern.shape)

    plt.figure(figsize=(6,5))
    plt.imshow(p_img, cmap="viridis")
    plt.title("(lib) P(Y=1|x) — Logistische Regression (GLM + Laplace Bayes)")
    plt.xlabel("Spalte"); plt.ylabel("Zeile")
    plt.colorbar(label="p̂")
    plt.tight_layout(); plt.show()

else:
    raise ValueError(f"Unbekannter model_type: {model_type}")

#%%
# 5. sample aus posterior ziehen
# Ziel: Für alle drei Fälle Posterior-Samples erzeugen und visualisieren.
#  - Poisson + (Gamma/Flat): λ-Samples per Zelle (und optional per Klasse)
#  - Poisson + Gauss (log λ): ziehe η-Samples ~ N(η̂, Var̂) und exponentiere (per Zelle oder Klasse)
#  - Logistic: ziehe β ~ N(β̂, Σ) und projiziere p = σ(Xβ)

# Sampling-Parameter
N_SAMPLES = 1          # Anzahl Posterior-Samples, die geplottet werden (pro Fall 1 Heatmap)
sample_mode = "per_cell"  # "per_cell" oder "per_class" (nur relevant für Poisson)

if model_type == "poisson":
    y = ergebnis.astype(float)

    if prior_type in {"gamma", "flat"}:
        # --- Gamma/Flat: konjugiert, direkt Gamma-Sampling ---
        if sample_mode == "per_class":
            # Klassenweise Posterior bestimmen und ein Sample ziehen
            mask0 = (pattern == 0)
            mask1 = ~mask0
            n0, n1 = int(mask0.sum()), int(mask1.sum())
            y0_sum, y1_sum = float(y[mask0].sum()), float(y[mask1].sum())
            if prior_type == "gamma":
                a0, b0 = alpha0 + y0_sum, beta0 + n0
                a1, b1 = alpha0 + y1_sum, beta0 + n1
            else:  # flat
                a0, b0 = 1.0 + y0_sum, 1.0
                a1, b1 = 1.0 + y1_sum, 1.0
            lam0 = np.random.gamma(shape=a0, scale=1.0/b0, size=N_SAMPLES)
            lam1 = np.random.gamma(shape=a1, scale=1.0/b1, size=N_SAMPLES)
            # zu Bildern mappen (zeige erstes Sample)
            lam_img = np.where(mask0, lam0[0], lam1[0])
        else:
            # Zellweise Posterior-Parameter
            if prior_type == "gamma":
                alpha_post = alpha0 + y
                beta_post  = beta0 + 1.0
            else:  # flat
                alpha_post = 1.0 + y
                beta_post  = 1.0
            lam_img = np.random.gamma(shape=alpha_post, scale=1.0/beta_post)

        plt.figure(figsize=(6,5))
        plt.imshow(lam_img, cmap="viridis")
        plt.title(f"Posterior-Sample λ — Poisson, Prior: {prior_type} ({sample_mode})")
        plt.xlabel("Spalte"); plt.ylabel("Zeile")
        plt.colorbar(label="λ (Sample)")
        plt.tight_layout(); plt.show()

    elif prior_type == "gauss":
        # --- Gauss-Prior auf η=log λ: Laplace → Normal-Sampling in η ---
        if sample_mode == "per_class":
            # Klassenweise Laplace (wie in 4.1)
            mask0 = (pattern == 0)
            mask1 = ~mask0
            n0, n1 = int(mask0.sum()), int(mask1.sum())
            y0_sum, y1_sum = float(y[mask0].sum()), float(y[mask1].sum())

            def neg_logpost_eta(eta, y_sum, n, mu0, sigma0):
                return -(y_sum*eta - n*np.exp(eta) - 0.5*((eta-mu0)/sigma0)**2)

            # Klasse 0
            eta0 = np.log(max(y0_sum / max(n0,1), 1e-8))
            # numerische zweite Ableitung per Newton-Schritt (kurz) zur Varianzschätzung
            for _ in range(12):
                exp_eta = np.exp(eta0)
                g = y0_sum - n0*exp_eta - (eta0 - mu0)/(sigma0**2)
                H = -n0*exp_eta - 1.0/(sigma0**2)
                eta0 = eta0 - g / H
            var0 = -1.0 / H
            # Klasse 1
            eta1 = np.log(max(y1_sum / max(n1,1), 1e-8))
            for _ in range(12):
                exp_eta = np.exp(eta1)
                g = y1_sum - n1*exp_eta - (eta1 - mu0)/(sigma0**2)
                H = -n1*exp_eta - 1.0/(sigma0**2)
                eta1 = eta1 - g / H
            var1 = -1.0 / H
            # Sample in η und zurück nach λ (zeige erstes Sample)
            eta0_s = eta0 + np.sqrt(var0) * np.random.randn()
            eta1_s = eta1 + np.sqrt(var1) * np.random.randn()
            lam_img = np.where(mask0, np.exp(eta0_s), np.exp(eta1_s))
        else:
            # Zellweise Laplace (wie in Schritt 4) und dann je Zelle sampeln
            y_flat = y.ravel()
            eta = np.log(np.maximum(y_flat, 1e-8))
            inv_sigma2 = 1.0 / (sigma0 ** 2)
            for _ in range(12):
                exp_eta = np.exp(eta)
                g = y_flat - exp_eta - (eta - mu0) * inv_sigma2
                H = -exp_eta - inv_sigma2
                eta = eta - g / H
            var_eta = -1.0 / H
            eta_samp = eta + np.sqrt(var_eta) * np.random.randn(*eta.shape)
            lam_img = np.exp(eta_samp).reshape(y.shape)

        plt.figure(figsize=(6,5))
        plt.imshow(lam_img, cmap="viridis")
        plt.title(f"Posterior-Sample λ — Poisson, Prior: gauss ({sample_mode})")
        plt.xlabel("Spalte"); plt.ylabel("Zeile")
        plt.colorbar(label="λ (Sample)")
        plt.tight_layout(); plt.show()

    else:
        raise ValueError(f"Unbekannter prior_type: {prior_type}")

elif model_type == "logistic":
    # --- β-Sampling: Laplace-Approx → β ~ N(β̂, Σ) ---
    # Versuche statsmodels zu nutzen; fällt zurück auf IRLS, falls nicht vorhanden
    try:
        import statsmodels.api as sm
        use_sm = True
    except Exception:
        use_sm = False

    if use_sm:
        if y_vec.ndim == 1 and y_vec.max() > 1:
            y_bin = (ergebnis > 0).astype(int).reshape(-1)
        else:
            y_bin = y_vec
        glm_bin = sm.GLM(y_bin, X_design, family=sm.families.Binomial())
        res = glm_bin.fit()
        beta_hat = res.params
        Sigma_mle = res.cov_params()
        tau2 = 10.0
        Prec_post = np.linalg.inv(Sigma_mle) + np.eye(len(beta_hat)) / tau2
        Sigma_post = np.linalg.inv(Prec_post)
        beta_map = Sigma_post @ (np.linalg.inv(Sigma_mle) @ beta_hat)
    else:
        # Fallback: eigene IRLS (wie oben), danach Laplace-Kovarianz
        tau2 = 10.0
        n, p = X_design.shape
        beta_map = np.zeros(p)
        for _ in range(25):
            eta = X_design @ beta_map
            p_hat = 1.0 / (1.0 + np.exp(-eta))
            W = p_hat * (1 - p_hat)
            z = eta + (y_vec - p_hat) / np.maximum(W, 1e-12)
            XT_W = X_design.T * W
            A = XT_W @ X_design + np.eye(p) / tau2
            b = XT_W @ z
            try:
                beta_new = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
            if np.max(np.abs(beta_new - beta_map)) < 1e-6:
                beta_map = beta_new
                break
            beta_map = beta_new
        # Kovarianz
        eta = X_design @ beta_map
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        W = p_hat * (1 - p_hat)
        XT_W = X_design.T * W
        A = XT_W @ X_design + np.eye(p) / tau2
        Sigma_post = np.linalg.inv(A)

    # Ziehe 1 Sample (oder N_SAMPLES) und visualisiere p
    beta_s = np.random.multivariate_normal(mean=np.array(beta_map), cov=Sigma_post)
    p_img = 1.0 / (1.0 + np.exp(-(X_design @ beta_s))).reshape(pattern.shape)

    plt.figure(figsize=(6,5))
    plt.imshow(p_img, cmap="viridis")
    plt.title("Posterior-Sample p = σ(Xβ) — Logistische Regression")
    plt.xlabel("Spalte"); plt.ylabel("Zeile")
    plt.colorbar(label="p (Sample)")
    plt.tight_layout(); plt.show()

else:
    raise ValueError(f"Unbekannter model_type: {model_type}")


# wird ein ziehen aus 1 dimensionaler gaussverteilung

#cha