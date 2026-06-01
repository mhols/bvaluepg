# Einfluss der Parameter auf die Verteilungen

Dieses Experiment betrachtet die Bausteine getrennt: Link-Funktion, Poisson-Likelihood, eindimensionale Posterior-Dichte und raeumlicher Gaussian-Prior. Das ist sinnvoll, weil dieselben Parameter im Posterior zusammenwirken, aber einzeln leichter zu interpretieren sind.

## Link-Funktionen

Die Link-Funktion bildet das latente Gaussian-Feld `f` auf eine positive Poisson-Rate ab.

`sigmoid`: `rate = lam * sigmoid(beta0 + beta1 * f)`

- `lam` ist eine harte obere Schranke der Rate. Groesseres `lam` streckt die Rate nach oben, aendert aber auch die Skala, auf der der Prior auf `f` in Raten uebersetzt wird.
- `beta0` verschiebt die Kurve. Positive Werte erhoehen die Rate schon bei gleichem `f`, negative Werte senken sie.
- `beta1` kontrolliert Steilheit und Richtung. Kleine Werte machen die Rate fast konstant, grosse Werte erzeugen eine harte Schwelle. Negative Werte drehen den Zusammenhang um.
- Die Ableitung ist in der Mitte am groessten und nahe `0` oder `lam` sehr klein. Dort wird die Inferenz auf `f` oft traeger, weil grosse Aenderungen in `f` nur kleine Aenderungen in der Rate erzeugen.

`softplus`: `rate = softplus(k * f) / k`

- `k` kontrolliert, wie scharf die Funktion vom positiven linearen Bereich in den fast-null Bereich uebergeht.
- Kleine `k` machen den Uebergang weich und verteilen Masse breiter.
- Grosse `k` naehern eine ReLU an: negative `f` werden stark gegen Rate `0` gedrueckt, positive `f` sind fast linear.
- Im Gegensatz zur Sigmoid-Funktion gibt es keine obere Schranke.

`exp`: `rate = exp(f)`

- Die Rate ist unbeschraenkt und waechst multiplikativ.
- Gaussian-Variation in `f` wird zu lognormaler Variation in der Rate.
- Groessere Prior-Varianz erzeugt schnell schwere rechte Tails und sehr grosse moegliche Raten.

## Induzierte Rate-Verteilungen

Wenn `f ~ N(mu, v2)`, entsteht durch die Link-Funktion eine Verteilung auf der Rate.

- `mu` verschiebt die typische Rate. Bei `exp` ist der Median `exp(mu)`. Bei `sigmoid` liegt die typische Rate ungefaehr bei `lam * sigmoid(mu)`.
- `v2` verbreitert die Verteilung in `f`. Nach der Transformation ist die Wirkung nicht symmetrisch: bei `exp` entstehen rechte Tails, bei `sigmoid` staut sich Masse nahe `0` und `lam`.
- Bei `sigmoid` ist die Rate immer in `(0, lam)`. Dadurch kann das Modell grosse Counts nur erklaeren, wenn `lam` gross genug ist.
- Bei `softplus` bestimmt `k`, wie stark negative `f` in sehr kleine Raten komprimiert werden.

## Poisson-Likelihood

Fuer beobachtete Counts `n` gilt `p(n | rate) proportional rate^n * exp(-rate)`.

- Als Funktion der Rate liegt das Maximum bei `rate = n` fuer `n > 0`.
- Fuer `n = 0` faellt die Likelihood monoton mit der Rate.
- Als Funktion von `f` haengt die Form stark vom Link ab:
  - `exp`: Maximum bei `f = log(n)` fuer `n > 0`.
  - `softplus`: Maximum bei dem `f`, fuer das `softplus(kf)/k = n`.
  - `sigmoid`: Maximum bei dem `f`, fuer das `lam * sigmoid(f) = n`, sofern `n < lam`. Wenn `n >= lam`, drueckt die Likelihood an die obere Saettigung.

## Eindimensionale Posterior-Dichte

Die neuen Posterior-Plots betrachten eine einzelne Zelle isoliert:

`p(f | n) proportional p(n | rate(f)) * N(f | mu, v2)`

- `n` ist der beobachtete Count in dieser Zelle. Kleine Counts ziehen den Posterior zu kleinen Raten, grosse Counts zu groesseren Raten.
- `mu` und `v2` bestimmen den Gaussian-Prior auf `f`. Kleine `v2` halten den Posterior nahe am Prior-Mittelwert, grosse `v2` lassen die Likelihood staerker dominieren.
- Bei `sigmoid` kann der Posterior fuer grosse `n` an die Saettigung laufen, wenn `lam` zu klein ist.
- Bei `softplus` veraendert `k` besonders den linken Posterior-Tail und die Schaerfe des Uebergangs nahe Rate `0`.
- Bei `exp` liegt der Likelihood-Peak fuer `n > 0` nahe `log(n)`, aber der Prior kann diesen Peak deutlich verschieben oder verbreitern.
- Die Plots zeigen bewusst `prior`, `likelihood` und `posterior` zusammen, damit sichtbar ist, welcher Bestandteil die Form dominiert.

## Raeumlicher Gaussian-Prior

Der Prior legt fest, welche Felder `f` vor den Daten plausibel sind.

- `v2` ist die marginale Varianz. Groesseres `v2` erlaubt staerkere lokale Ausschlaege in `f` und dadurch staerkere Ratenkontraste.
- `rho` ist die Korrelationslaenge. Kleines `rho` erzeugt schnell wechselnde, fleckige Felder. Grosses `rho` erzeugt glatte, grossraeumige Strukturen.
- Der Kernel bestimmt die Glattheit:
  - Gaussian-Kernel ist sehr glatt und faellt fuer kleine Distanzen langsam ab.
  - Matern 1/2 ist rauer und entspricht exponentieller Korrelation.
  - Matern 3/2 und 5/2 liegen dazwischen; 5/2 ist glatter als 3/2.
- Ein kleiner Jitter auf der Diagonale stabilisiert die Cholesky-Zerlegung, veraendert aber die Modellidee praktisch nicht.

## Praktische Interpretation

- Wenn die Daten Counts groesser als `lam` enthalten, ist der Sigmoid-Link mit diesem `lam` strukturell zu eng.
- Wenn einzelne Zellen sehr grosse Counts haben koennen, ist `exp` flexibel, aber empfindlich gegen grosse `v2`.
- `softplus` ist ein Kompromiss: positive Raten sind unbeschraenkt, aber der linke Tail ist kontrollierter als bei `exp`.
- `rho` und `v2` sollten zusammen betrachtet werden: grosses `v2` mit kleinem `rho` erzeugt noisy Felder, grosses `v2` mit grossem `rho` erzeugt starke grossraeumige Gradienten.

## Experiment ausfuehren

```bash
python3 exp_parameter_influence.py
```

Die Plots werden standardmaessig nach `../results/parameter_influence/` geschrieben. Mit `--outdir` kann ein anderer Zielordner gesetzt werden.
