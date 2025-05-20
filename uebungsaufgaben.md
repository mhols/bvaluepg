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

Die Pólya-Gamma-Datenaugmentation erlaubt es, aus einer komplizierten posterior-Verteilung eine bedingte Normalverteilung für \( \beta \) zu machen, was Gibbs-Sampling ermöglicht. Dies ist besonders nützlich bei logistischer Regression.
Der Gibbs-Sampler mit Pólya-Gamma-Datenaugmentation nutzt eine Hilfsvariable \( \omega_i \sim \text{PG}(1, x_i \beta) \). Die Schritte sind:
1. Ziehe \( \omega_i \sim \text{PG}(1, x_i \beta) \)
2. Ziehe \( \beta \sim \mathcal{N}(m, V) \), wobei:
   \[
   V = (X^\top \Omega X + \tau^{-2})^{-1}, \quad
   m = V X^\top (y - 0.5)
   \]
   mit \(\Omega = \text{diag}(\omega_1, \dots, \omega_n)\)
