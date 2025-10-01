---
title: Inverse-Crime Pipeline — Photon Catalog & Polya‑Gamma Rekonstruktion
author: Toni Luhdo (konsolidierte Notizen)
date: 2025-09-30
---

# Inverse-Crime Pipeline — Übersicht

Ziel:  
1. Einen **Zufallskatalog** erzeugen (Anzahl Photonen pro Pixel/Bin) basierend auf einem zugrundeliegenden Bildmuster (z. B. Schachbrett).  
2. Bei der Rekonstruktion **kein inverse crime** begehen — d.h. bewusst ein leicht abweichendes Rekonstruktionsmodell verwenden.  
3. Mit **Pólya‑Gamma Gibbs‑Sampling** (Erweiterung für Multinomial / multinomial-logistische Formulierung) versuchen, aus dem Zufallskatalog die zugrunde liegenden Wahrscheinlichkeiten / Intensitäten wiederzugewinnen.  
4. Diagnose: Vergleich von wahren vs rekonstruierten p_i (oder Intensitäten), MSE, Posterior‑Unsicherheiten.

---

# 1. Notation & Setup

- Pixel-Anzahl: \(K\) (z. B. \(K=30\times30=900\)).  
- Wahres (glattes) Bild: \(\Lambda = (\lambda_1,\dots,\lambda_K)\), \(\lambda_i \ge 0\).  
- Gesamte Photonenzahl (Multinomial-Fall): \(N\) (z. B. \(N=10^4\) oder \(10^5\)).  
- Wahrscheinlichkeiten: \(p_i = \dfrac{\lambda_i}{\sum_{j=1}^K \lambda_j}\), \(\sum_i p_i = 1\).  
- Beobachtete Counts: \(Y=(Y_1,\dots,Y_K)\) mit \(\sum_i Y_i = N\).

---

# 2. Forward / Simulation (Zufallskatalog erzeugen)

## 2.1 Bild & PSF
- Erzeuge **wahres Bild** (z. B. Schachbrett, glatte Gauss‑Felder, etc.). Skaliere Intensitäten so, dass \(\sum_i \lambda_i\) sinnvolle Photonenzahl ergibt.  
- Wende optional eine PSF (Punktspreizfunktion) an: \(\tilde{\lambda} = \text{PSF} * \Lambda\) (Faltung). Die PSF wird **in der Simulation** verwendet (PSF_sim).

## 2.2 Multinomial-Ziehung (Zufallskatalog)
- Normiere: \(p_i = \tilde{\lambda}_i / \sum_j \tilde{\lambda}_j\).  
- Ziehe global:
\[
(Y_1,\dots,Y_K) \sim \text{Multinomial}(N, (p_1,\dots,p_K)).
\]
- Ergebnis: ein *Zufallskatalog* — eine Abbildung „Photonen pro Pixel“.

**Hinweis:** Alternativ kann man Poisson‑Counts pro Pixel simulieren (unabhängig): \(Y_i \sim \text{Poisson}(\mu_i)\). Beides hat unterschiedliche Eigenschaften; Multinomial behält Gesamtanzahl \(N\) fix.

---

# 3. Vermeidung des Inverse Crime

Inverse crime heißt: Simulation und Rekonstruktion verwenden exakt dasselbe Vorwärtsmodell — dann ist Rekonstruktion unrealistisch einfach. Maßnahmen:

1. **Unterschiedliche PSFs**: Verwende für Simulation `PSF_sim` und für Rekonstruktion `PSF_rec` mit leicht unterschiedlichen Parametern (andere Breite, andere Form).  
2. **Unterschiedlicher Regularisierer / Prior**: Simuliere mit glatteren/realistischeren Intensitäten, rekonstruiere mit einem anderen Prior (z. B. glättende L2 vs TV vs räumliche Gauß‑Markov).  
3. **Diskretisierung / Likelihood‑Formulierung ändern**: Simuliere Multinomial, rekonstruiere mit multinomial-logistischer Parametrisierung (oder approximative Binomial‑Form).  
4. **Störgrößen variieren**: füge in Simulation Hintergrund / zusätzliche Rauschkomponenten ein, die der Rekonstrukteur nicht exakt modelliert.

Ziel: Der Rekonstrukteur kennt das grobe Modell, nicht die exakte Simulationskonfiguration.

---

# 4. Rekonstruktionsmodell: Multinomial‑Logit + Pólya‑Gamma

Wir wollen die Multinomial‑Wahrscheinlichkeiten \(p_i\) in parametrischer Form modellieren. Eine übliche Parametrisierung ist die **multinomial-logistische (softmax)** Form:

- Wähle \(K\) Logit-Parameter \(\theta_i\). Für Identifizierbarkeit setze z.B. \(\theta_K = 0\) (Referenzkategorie).  
- Softmax:
\[
p_i = \frac{\exp(\theta_i)}{\sum_{j=1}^K \exp(\theta_j)}.
\]

Die Likelihood ist dann:
\[
P(Y \mid \theta) = \frac{N!}{\prod_i Y_i!} \prod_{i=1}^K p_i^{Y_i}.
\]

### Pólya‑Gamma für Multinomial?
Die klassische Pólya‑Gamma‑Augmentation gilt direkt für **Binomial/Logit**-Modelle. Für die Multinomial/Softmax existieren Erweiterungen:  
- Schreibe das Multinomial als Produkt von bedingten Binomialen (sequentielle Faktorisation) oder  
- verwende die **K‑1 Logit‑Vergleiche** gegen eine Referenzklasse: für \(i=1,\dots,K-1\) definieren wir log‑Odds gegen Klasse \(K\):
\[
\eta_i = \theta_i - \theta_K = \theta_i,
\]
und man kann eine geeignete Pólya‑Gamma‑Erweiterung einsetzen, indem man für jede Pixelmultinomial‑Ziehung \(Y\) K‑1 Pólya‑Gamma‑Hilfsvariablen einführt (eine pro Logit). Das führt dazu, dass die bedingte Verteilung der \(\theta\) näherungsweise normal wird — geeignet für Gibbs‑Sampling.

> **Praktischer Vorschlag:** Implementiere die Multinomial‑logit‑Rekonstruktion über die **sequentielle Binomialfaktorisierung**:
- Für \(i=1\ldots K-1\):
  - Betrachte \(Y^{(1)}_1 = Y_1 \sim \mathrm{Binomial}(N, q_1)\) mit \(q_1 = p_1\).
  - Dann \(Y^{(2)}_2 \sim \mathrm{Binomial}(N - Y_1, q_2')\) mit \(q_2' = p_2/(1-p_1)\), usw.
- Für jede Binomialbedingung kann Pólya‑Gamma Augmentation angewendet werden.
- Nachteil: Reihenfolge‑Abhängigkeit, aber praktikabel und einfach umzusetzen.

Alternativ kann man die **multinomial‑PG** Formulierung (mehrere Hilfsvariablen pro Beobachtung) implementieren; das ist etwas komplexer, aber sauberer und rechnet alle K‑1 Logits simultan.

---

# 5. Mathematischer Kern — Pólya‑Gamma Augmentation (Kurzbeschreibung)

Für Binomial‑Logit (ein Beobachtung mit Anzahl \(n\) und Erfolge \(y\) mit logit \(\psi\)):
\[
\Pr(y \mid \psi) \propto e^{y\psi} (1+e^\psi)^{-n}.
\]
Pólya‑Gamma Augmentation führt Hilfsvariable \(\omega \sim PG(n,\psi)\) ein, sodass
\[
\Pr(y,\omega \mid \psi) \propto \exp\!\Big\{\kappa \psi - \tfrac{1}{2}\omega\psi^2\Big\} f(\omega),
\]
mit \(\kappa = y - n/2\). Konditional auf \(\omega\) wird \(\psi\) gaussianartig (konjugiert). Das erlaubt Gibbs‑Updates.

Für Multinomial über eine sequentielle Binomialzerlegung wendest du diese Trick für jede Schritt‑Binomial‑Verteilung an.

---

# 6. Algorithmus (Sequentielle Binomial‑Gibbs mit Pólya‑Gamma)

**Input:** beobachtete Counts \(Y\), Gesamtanzahl \(N\), Initialparameter \(\theta^{(0)}\), PSF_rec (falls du Räumlichkeit/Blur modellieren willst), Prior \(p(\theta)\) (z. B. Gauß mit räumlicher Kovarianz).  
**Output:** Posterior‑Samples \(\{\theta^{(t)}\}\) oder Posterior‑Samples für \(p_i\).

1. Initialisiere \(\theta^{(0)}\) (z. B. \(\theta_i^{(0)} = \log(Y_i + 0.1)\) normalisiert).
2. Für Iteration \(t=1\dots T\):
   - Für \(i=1\) bis \(K-1\) (sequentiell):
     1. Betrachte die Binomialbedingung für \(Y_i\) gegeben die bisherigen \(Y_{1:i-1}\) (siehe Faktorisierung oben) mit Anzahl \(n_i = N - \sum_{j< i} Y_j\) und Erfolge \(y_i = Y_i\).
     2. Setze Modell \(y_i \sim \mathrm{Binomial}(n_i, \sigma(\psi_i))\) mit \(\psi_i = \theta_i - \log\sum_{j\ge i} \exp(\theta_j)\) (spezielle Formulierung der konditionalen Logits). Vereinfachung: nutze eine numerisch stabilisierte Form für Logit.
     3. Ziehe \(\omega_i \sim PG(n_i, \psi_i)\).
     4. Aktualisiere \(\theta_i\) aus der bedingten Normalverteilung (inkl. Prior und ggf. räumliche Kovarianz). Wenn Prior korreliert ist, wird das Update ggf. multivariat (Cholesky für Kovarianz).
   - Falls räumliche Prioren genutzt werden: aktualisiere ggf. Blockweise oder mit einem multivariaten Schritt (z. B. Gibbs-Update mit Cholesky).
3. Sammle Samples nach Burn‑in und berechne Posterior‑Summaries für \(p_i = \mathrm{softmax}(\theta_i)\).

**Praktischer Tipp:** Für K groß (z. B. 900) ist sequentielles Gibbs rechenintensiv; man kann parallelisieren oder Blockupdates verwenden.

---

# 7. Implementierungsdetails & Pseudocode

**Python‑Pakete (empfohlen):**
- `numpy`, `scipy`, `matplotlib`  
- `pypolyagamma` oder `polyagamma` (für Pólya‑Gamma Ziehen). Falls nicht verfügbar, gibt es Approximationen (z. B. akzeptiertes Rejection‑Sampling Implementierungen aus Literatur).

**Pseudocode (hochlevel):**
```python
# Inputs: Y (K-vector), N, K, T, burnin
theta = initialize_theta(Y)
for t in range(T):
    remaining = N
    for i in range(K-1):
        n_i = remaining
        y_i = Y[i]
        # compute current conditional logit psi_i from theta
        psi_i = compute_conditional_logit(theta, i)
        omega_i = sample_PG(n_i, psi_i)
        # compute conditional normal parameters for theta_i (include prior precision)
        mu_i, var_i = compute_conditional_normal(y_i, n_i, omega_i, prior_terms)
        theta[i] = np.random.normal(mu_i, sqrt(var_i))
        remaining -= y_i
    # set theta[K] = 0 (reference) or update last implicitly
    # optionally update spatial prior hyperparameters
    if t > burnin:
        store_sample(theta)
```

---

# 8. Diagnostik & Evaluation

- **Rekonstruiertes p̂ = softmax(θ̄)** (Posterior Mittel der θ -> p̂): Vergleiche mit wahren \(p\) aus Simulation.  
- Metriken: MSE(\(p\) vs \(\hat p\)), KL‑Divergenz: \(D_{KL}(p || \hat p)\), SSIM/MSE für umskalierte Intensitäten.  
- Posterior Unsicherheit: pixelweise Posterior‑Varianz, Credible Intervals.  
- Convergence: Trace‑plots für ausgewählte θ‑Parameter, Autokorrelation, ESS (effective sample size).  
- Check Inverse Crime: Variiere PSF_rec vs PSF_sim und beobachte, wie stark Rekonstruktion leidet; dokumentiere Unterschiede.

---

# 9. Praktische Parameterempfehlungen (Startwerte)

- Bild: \(30\times30\) (K=900).  
- Photonenzahl: \(N = 10^4\) (relativ dünn), alternativ \(10^5\).  
- Iterationen: T = 5000, Burn‑in = 2000 (je nach Mixing).  
- Prior: \(\theta \sim \mathcal{N}(m, \tau^{-1} I)\) mit \(\tau=10^{-3}\) initial; für räumliche Korrelation: GMRF mit Präzisionsmatrix \(Q\) (einfache Laplace Präzision).  
- PSF‑Unterschied: sigma_sim = 1.8, sigma_rec = 1.2 (kleiner Unterschied genügt).

---

# 10. Was du konkret überprüfen willst (Checklist)

- [ ] Kannst du aus dem Zufallskatalog (Multinomial) mit dem Pólya‑Gamma Gibbs‑Sampler die zugrundeliegenden p_i zurückgewinnen (innerhalb der Posterior‑Unsicherheit)?  
- [ ] Wie wirkt sich N (Photonenzahl) auf Rekonstruktion & Unsicherheit aus?  
- [ ] Wie groß muss die Abweichung zwischen PSF_sim und PSF_rec sein, damit Rekonstruktion merklich schlechter wird?  
- [ ] Funktioniert die sequentielle Binomial‑Faktorisierung numerisch stabil für K groß? Wenn nein: probiere multinomial‑PG Erweiterung.  
- [ ] Dokumentiere mixing (Traceplots), MSE, KL, und visuelle Vergleiche (Wahrscheinlichkeits‑Karten).

---

# 11. Weiterführende Optionen

- Implementiere die **vollständige multinomial‑PG** Augmentation (K‑1 PG‑Variablen pro Beobachtung) statt sequentieller Faktorisierung (sauberer, aber komplexer).  
- Verwende spatial prior (GMRF) und block‑Gibbs mit Cholesky (effizientere multivariate Updates).  
- Baue eine Jupyter‑Notebook‑Version mit interaktiven Plots (PSF, N, Prior strength).  
- Implementiere Vergleich mit deterministischem MAP‑Rekonstruktor (z. B. L‑BFGS MAP wie im Python‑Script zuvor) als Baseline.

---

# 12. Referenz‑Pseudocode & Hinweise zum Testen

- Start: Erzeuge `true_img`, `psf_sim`, `Y = Multinomial(N, normalize(psf_sim * true_img))`.  
- Rekonstruktion: Implementiere `PG_sampler_multinomial_sequential(Y, N, K, prior, psf_rec)` wie oben.  
- Auswertung: Visualisiere `true p`, `observed counts`, `posterior mean p`, `posterior sd p`.

---

Wenn du willst, kann ich:
- dir direkt ein **Jupyter‑Notebook** mit einer minimalen, aber lauffähigen Implementation der sequentiellen Binomial‑PG‑Gibbs‑Variante schreiben (inkl. Visualisierungen), **oder**
- das Markdown-Dokument als Datei speichern (ich habe es gerade erstellt), oder
- die komplexere **vollständige multinomial‑PG** Implementierung (etwas aufwändiger) ausprogrammieren.

Welche der drei Optionen hättest du gern als nächstes?