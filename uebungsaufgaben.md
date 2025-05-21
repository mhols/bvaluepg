---
title: Übungsaufgaben
author: Toni Luhdo
created: 2025-05-06
last_modified: 2025-05-13

---

## Übungsaufgaben:

Pythonskript: logRegUebungsaufgaben.py

### Bayessche logistische Regression (1 Feature, 10 Beobachtungen)

### Problemstellung

Ein klassifikatorisches Modell soll ermitteln, ob eine Person ein Produkt kauft (\( y_i = 1 \)) oder nicht (\( y_i = 0 \)), abhängig von der **Anzahl der Werbeanzeigen** \( x_i \), die sie gesehen hat.

Gegeben sind folgende 10 Beobachtungen:

| \( x_i \) (Anzahl Werbeanzeigen) | \( y_i \) (Kauf: Ja = 1, Nein = 0) |
|----------------------------------|-------------------------------------|
| 0                                | 0                                   |
| 1                                | 0                                   |
| 2                                | 0                                   |
| 3                                | 1                                   |
| 3                                | 0                                   |
| 4                                | 1                                   |
| 5                                | 1                                   |
| 6                                | 1                                   |
| 7                                | 1                                   |
| 8                                | 1                                   |

### Modellannahmen

Wir modellieren:

\[
y_i \sim \text{Bernoulli}(\sigma(x_i \cdot \beta)), \quad \text{mit} \quad \sigma(z) = \frac{1}{1 + e^{-z}}
\]

Prior:

\[
\beta \sim \text{Uniform}(-10, 10)
\]

---

### Aufgaben

0. **Bestimme Mittelwert und Varianz**

1. **Likelihood aufstellen:**  
   Formuliere die Likelihood-Funktion \( L(\beta) \) der Daten gegeben \( \beta \).

2. **Log-Likelihood formulieren:**  
   Gib die log-Likelihood \( \log L(\beta) \) an.

3. **Prior einbeziehen:**  
   Formuliere den unnormalisierten log-Prior (hier konstant im Intervall).

4. **Posterior proportional:**  
   Bestimme die unnormierte Posterior-Dichte \( p(\beta | \mathbf{y}, \mathbf{x}) \propto L(\beta) \cdot p(\beta) \).

5. **Plots:**  
   Zeichne grob, wie der Posterior aussieht (z. B. mit Python oder von Hand).

6. **(Optional)**:  
   Implementiere einen Gibbs-Sampler mit Pólya-Gamma-Datenaugmentation zur Schätzung des Posterior-Verlaufs von \( \beta \).

---

### Loesungen



#### Lösung zu Aufgabe 0

Finde den Mittelwert und die Varianz der gegebenen Werbekontaktzahlen \( x_i \), um ein erstes Gefühl für die Datenlage zu bekommen. Die Varianz wird hier als Stichprobenvarianz berechnet (Division durch \( n - 1 \)).
\[
\text{Mittelwert } \bar{x} = \frac{0 + 1 + 2 + 3 + 3 + 4 + 5 + 6 + 7 + 8}{10} = 3{,}9
\]
\[
\text{Varianz } s^2 = \frac{1}{9} \sum_{i=1}^{10} (x_i - \bar{x})^2 = 6{,}54
\]

#### Lösung zu Aufgabe 1

Die Likelihood-Funktion beschreibt die Wahrscheinlichkeit der Daten gegeben dem Parameter \( \beta \). Bei einer Bernoulli-Verteilung ergibt sich die Produktform aus den einzelnen Wahrscheinlichkeiten.
\[
L(\beta) = \prod_{i=1}^{n} \sigma(x_i \beta)^{y_i} (1 - \sigma(x_i \beta))^{1 - y_i}
\]

In ausgeschriebener Form ergibt sich:
\[
L(\beta) = \prod_{i=1}^{n} \left( \frac{1}{1 + e^{-x_i \beta}} \right)^{y_i} \left( 1 - \frac{1}{1 + e^{-x_i \beta}} \right)^{1 - y_i}
\]

Die Sigmoidfunktion sorgt dafür, dass der lineare Prädiktor \( x_i \cdot \beta \) in eine Wahrscheinlichkeit transformiert wird. Kleine oder sehr große Werte von \( z \) führen zu Wahrscheinlichkeiten nahe 0 bzw. 1. Der Übergang ist glatt und monoton.

#### Lösung zu Aufgabe 2

Die Log-Likelihood ist die logarithmierte Version der Likelihood, was die Optimierung und Interpretation erleichtert, da Produkte zu Summen werden. Sie ist zentral für die Maximum-Likelihood-Schätzung und für den Posterior.
\[
\log L(\beta) = \sum_{i=1}^{n} \left[ y_i \log(\sigma(x_i \beta)) + (1 - y_i) \log(1 - \sigma(x_i \beta)) \right]
\]

#### Lösung zu Aufgabe 3

Der Prior spiegelt unser Vorwissen über \( \beta \) wider. Hier ist er gleichverteilt im Intervall \([-10, 10]\), also konstant in diesem Bereich.
\[
p(\beta) \propto \begin{cases}
1 & \text{für } \beta \in [-10, 10] \\
0 & \text{sonst}
\end{cases}
\quad \Rightarrow \quad \log p(\beta) = \text{const} \text{ (innerhalb des Intervalls)}
\]

#### Lösung zu Aufgabe 4

Nach dem Satz von Bayes ist der Posterior proportional zur Likelihood multipliziert mit dem Prior. Da wir uns nur für die Form des Posteriors interessieren, ignorieren wir die Normierungskonstante.
\[
p(\beta \mid \mathbf{y}, \mathbf{x}) \propto L(\beta) \cdot p(\beta)
\quad \Rightarrow \quad
\log p(\beta \mid \mathbf{y}, \mathbf{x}) \propto \log L(\beta) + \log p(\beta)
\]

#### Lösung zu Aufgabe 5

Der Posterior kann grafisch untersucht werden, indem man die unnormierte Posterior-Dichte über verschiedene \(\beta\)-Werte aufträgt. Dies liefert visuelle Einsichten über Lage und Unsicherheit der Parameterschätzung.

```python
# Aufgabe 5 – Likelihood-Plot
likelihood_vals = np.array([likelihood(b) for b in beta_vals])
plt.plot(beta_vals, likelihood_vals)
plt.xlabel("β")
plt.ylabel("Likelihood")
plt.title("Likelihood-Verlauf")
plt.grid(True)
plt.tight_layout()
plt.show()
```

#### Lösung zu Aufgabe 6

Polya-Gamma-Datenaugmentation erlaubt es, aus einer komplizierten posterior-Verteilung eine bedingte Normalverteilung für \( \beta \) zu machen, was Gibbs-Sampling ermöglicht. Super bei logistischer Regression.
Der Gibbs-Sampler mit Polya-Gamma-Datenaugmentation nutzt eine latente Variable \( \omega_i \sim \text{PG}(1, x_i \beta) \). Die Schritte sind:
1. Ziehe \( \omega_i \sim \text{PG}(1, x_i \beta) \)
2. Ziehe \( \beta \sim \mathcal{N}(m, V) \), wobei:
   \[
   V = (X^\top \Omega X + \tau^{-2})^{-1}, \quad
   m = V X^\top (y - 0.5)
   \]
   mit \(\Omega = \text{diag}(\omega_1, \dots, \omega_n)\)

### Versuch Mathematischer Erklaerung

Der Gibbs-Sampler verwendet die Polya-Gamma-Datenaugmentation, um aus der Posterior-Verteilung von β zu ziehen.
Für jede Iteration werden die folgenden Schritte durchgeführt:

1. **Auxiliary Sampling**: Ziehe \(ω_i\) ~ \(PG(1, x_i * β)\)
   – diese Hilfsvariablen erlauben es, die Bernoulli-Logistik zu konditionieren.

2. **β aus Normalverteilung ziehen**:
   \[
   V = \left(X^\top \Omega X + \tau^{-2} \right)^{-1}, \quad
   m = V X^\top (y - 0.5), \quad
   \beta \sim \mathcal{N}(m, V)
   \]

Dabei ist:
- \( \Omega = \text{diag}(\omega_1, \dots, \omega_n) \)
- \( \tau^2 \) die Prior-Varianz von β (Normalprior: \( \beta \sim \mathcal{N}(0, \tau^2) \))


#### \( V \)

Der Ausdruck \( V = (X^\top \Omega X + \tau^{-2})^{-1} \) gibt die Varianz (bzw. bei mehreren Parametern die Kovarianzmatrix) der bedingten Posteriorverteilung von \( \beta \) an.

Diese ist notwendig, weil der Gibbs-Sampler in jedem Schritt eine neue Stichprobe von \( \beta \) aus einer Normalverteilung zieht:
\[
\beta \sim \mathcal{N}(m, V)
\]

Dabei beschreibt:
- **\( V \)**: die Unsicherheit über \( \beta \), gegeben die aktuellen Daten und Hilfsvariablen \( \omega \).
- **\( m \)**: den Mittelwert, der die „beste Schätzung“ von \( \beta \) in dieser Iteration darstellt.

Je größer die Information der Daten (d.h. je stärker \( X^\top \Omega X \)), desto kleiner wird \( V \), und desto schärfer wird die Verteilung um \( m \).

Mathematisch ergibt sich \( V \) aus der Kombination:
- der Likelihood-Information via \( X^\top \Omega X \)
- und dem Regularisierungseinfluss des Priors via \( \tau^{-2} \).

Durch die Kombination dieser Schritte ergibt sich eine effiziente Sampling-Prozedur, die konjugierte Strukturen nutzt.

# 
### Erweiterte Problemstellung: Logistische Regression mit Intercept

In dieser erweiterten Version wird zusätzlich zum Effekt der Werbeanzeigen ein Intercept (Achsenabschnitt) berücksichtigt. Die Modellgleichung lautet:

\[
y_i \sim \text{Bernoulli}(\sigma(\beta_0 + \beta_1 x_i)), \quad \text{mit} \quad \sigma(z) = \frac{1}{1 + e^{-z}}
\]

Dabei ist:
- \( \beta_0 \) der Intercept (Basiswahrscheinlichkeit ohne Werbung)
- \( \beta_1 \) der Effekt pro zusätzlicher Werbeanzeige

Prior:
\[
\beta_0, \beta_1 \sim \mathcal{N}(0, \tau^2)
\]

---

### Aufgaben

1. **Modell erklaeren:**  
   Erklaere, was der Intercept \( \beta_0 \) und der Koeffizient \( \beta_1 \) jeweils bedeuten.

2. **Likelihood ableiten:**  
   Leite die Likelihood-Funktion für die Parameter \( (\beta_0, \beta_1) \) her.

2. **Log-Likelihood ableiten:**  
   Leite die Log-Likelihood-Funktion für die Parameter \( (\beta_0, \beta_1) \) her.

3. **Posterior formulieren:**  
   Formuliere die unnormierte Posterior-Funktion bei gegebenem Normalprior mit Varianz \( \tau^2 \).

4. **Visualisierung:**  
   Zeichne beispielhaft eine Sigmoidkurve mit einem positiven Intercept und moderiatem Steigungsparameter.

5. **Gibbs-Sampling:**
   Beschreibe, wie sich das Pólya-Gamma-Gibbs-Sampling ändert, wenn ein Intercept eingeführt wird. Welche Matrixdimensionen und Schritte ändern sich?

### Loesungen



#### Lösung zu Aufgabe 1

Der Intercept \( \beta_0 \) gibt die log-Odds (bzw. die Ausgangswahrscheinlichkeit) an, wenn \( x = 0 \), also wenn keine Werbeanzeigen gezeigt wurden.  
Der Koeffizient \( \beta_1 \) beschreibt den Effekt jeder zusätzlichen Werbeanzeige auf die log-Odds für einen Kauf.  
Positive Werte von \( \beta_1 \) bedeuten steigende Kaufwahrscheinlichkeit mit mehr Anzeigen.

#### Lösung zu Aufgabe 2

Die Likelihood lautet:
\[
L(\beta_0, \beta_1) = \prod_{i=1}^{n} \sigma(\beta_0 + \beta_1 x_i)^{y_i} \cdot (1 - \sigma(\beta_0 + \beta_1 x_i))^{1 - y_i}
\]

#### Lösung zu Aufgabe 3

Die Log-Likelihood ist:
\[
\log L(\beta_0, \beta_1) = \sum_{i=1}^{n} \left[ y_i \log(\sigma(\beta_0 + \beta_1 x_i)) + (1 - y_i) \log(1 - \sigma(\beta_0 + \beta_1 x_i)) \right]
\]

#### Lösung zu Aufgabe 4

Bei einer beispielhaften Wahl \( \beta_0 = -1 \), \( \beta_1 = 0.8 \) ergibt sich eine Sigmoidfunktion:
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
beta0 = -1
beta1 = 0.8
p = 1 / (1 + np.exp(-(beta0 + beta1 * x)))

plt.plot(x, p)
plt.xlabel("Anzahl Werbeanzeigen (x)")
plt.ylabel("Kaufwahrscheinlichkeit")
plt.title("Sigmoid mit Intercept β₀ = -1, Steigung β₁ = 0.8")
plt.grid(True)
plt.show()
```

#### Lösung zu Aufgabe 5

Beim Gibbs-Sampling mit Intercept erweitert sich das Design-Matrix \( X \) um eine Spalte für den Bias-Term:
\[
X = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}
\]

Der Rest des Samplers bleibt formal gleich, aber:
- \( \beta \) ist jetzt ein Vektor mit 2 Elementen: \( \beta = [\beta_0, \beta_1] \)
- Die Dimensionen von \( X \), \( \Omega \), \( V \) und \( m \) passen sich entsprechend an.

Die Posterior-Bedingung für \( \beta \) lautet dann:
\[
V = \left( X^\top \Omega X + \tau^{-2} I \right)^{-1}, \quad
m = V X^\top (y - 0.5)
\]
wobei \( I \) die 2x2-Einheitsmatrix ist (falls beide Parameter denselben Prior bekommen).


### Erweiterte Problemstellung 2: Logistische Regression nur Intercept

Gleiche Fragestellung. Siehe py-skript