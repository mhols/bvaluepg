---
title: Poisson–Logistic Model with Pólya–Gamma Augmentation
author: Toni Luhdo
created: 2026-01-12
last_modified: 2026-01-12
project: BVALUEPG
description: >
  Herleitung der bedingten Gaußschen A-posteriori-Verteilung für ein Poisson-Logistik-Modell
  unter Verwendung von Pólya-Gamma-Datenanreicherung.
---

# Modell + Herleitung

## 1) Modellannahmen

Wir beobachten Zähldaten \(n_i\) und modellieren (wie in der Skizze):

\[
n_i \mid f_i \sim \text{Poisson}(\lambda_i),
\qquad 
\lambda_i := \lambda\,\sigma(f_i),
\qquad 
\sigma(f) := \frac{1}{1+e^{-f}}=\frac{e^{f}}{1+e^{f}}.
\]

Prior auf dem latenten Feld/Parametervektor \(f\in\mathbb{R}^N\):

\[
f \sim \mathcal N(\mu,\Sigma).
\]

---

## 2) Likelihood und Log-Likelihood

Die Likelihood ist

\[
L(f;n)
=\prod_{i=1}^N \frac{(\lambda\sigma(f_i))^{n_i}}{n_i!}\,e^{-\lambda\sigma(f_i)}.
\]

Log-Likelihood (Konstanten in \(f\) werden oft in „\(+c\)“ gepackt):

\[
\ell(f\mid n)
=\sum_{i=1}^N \Big(n_i\log(\lambda\sigma(f_i)) -\lambda\sigma(f_i)\Big) + c.
\]

Posterior (bis auf Normierung):

\[
p(f\mid n)\propto L(f;n)\,p(f).
\]

Log-Posterior:

\[
\log p(f\mid n)
=\sum_{i=1}^N \Big(n_i\log(\lambda\sigma(f_i)) -\lambda\sigma(f_i)\Big)
-\frac12 (f-\mu)^\top \Sigma^{-1}(f-\mu) + c.
\]

**Wichtige Beobachtung:** Wegen \(\sigma(f_i)\) ist das **nicht** quadratisch in \(f\) \(\Rightarrow\) Posterior ist **nicht** direkt Gauß.

---

## 3) Trick 1: Einführung von \(k_i\) (Poisson-Splitting)

Wir nutzen die Identität (TODO explain better). Der trick ist eigentlich

\[
e^{-\lambda \sigma(f_i)} = e^{-\lambda (1-\sigma(-f_i))} = e^{-\lambda} \sum_{k_i=0}^\infty 
\frac{(\lambda\sigma(-f_i))^{k_i}}{k_i!} 
\]

\[
\lambda\sigma(f_i)=\frac{\lambda e^{f_i}}{1+e^{f_i}}.
\]

Führe latente Zählvariable \(k_i\) ein, so dass

\[
k_i\mid f_i \sim \text{Poisson}\!\left(\lambda\sigma(-f_i)\right)
=\text{Poisson}\!\left(\frac{\lambda}{1+e^{f_i}}\right),
\]
und \(n_i\mid f_i\) wie oben, **unabhängig** gegeben \(f_i\).

Dann ist

\[
b_i := n_i+k_i \sim \text{Poisson}(2)
\quad\text{(f-frei!)}
\]

und die gemeinsame (augmentierte) Likelihood \((n_i,k_i)\mid f_i\) vereinfacht zu einer logistischen Form:

\[
p(n_i,k_i\mid f_i)
\propto
\frac{e^{n_i f_i}}{(1+e^{f_i})^{\,b_i}}
\quad (\text{Konstanten in }f_i\text{ weggelassen}).
\]

Definiere außerdem 

\[
b_i := n_i+k_i,
\qquad
\kappa_i := \frac{n_i-k_i}{2}.
\]

Dann gilt

\[
\frac{e^{n_i f_i}}{(1+e^{f_i})^{b_i}}
=\frac{e^{\kappa_i f_i}}{(1+e^{f_i})^{b_i}}\cdot e^{\frac{b_i}{2}f_i}
\propto
\frac{e^{\kappa_i f_i}}{(1+e^{f_i})^{b_i}}.
\]

Damit hat man pro \(i\) den „PG-kompatiblen“ Faktor

\[
\frac{e^{\kappa_i f_i}}{(1+e^{f_i})^{b_i}}.
\]

---

## 4) Trick 2: Pólya–Gamma-Augmentation

Pólya–Gamma-Identität (schematisch):

\[
\frac{e^{\kappa f}}{(1+e^{f})^{b}}
=\int \exp\!\left(-\frac{\omega f^{2}}{2}+\kappa f\right)\,p(\omega)\,d\omega,
\qquad
\omega\sim \text{PG}(b,0)
\]
(bzw. in manchen Darstellungen \(\omega\mid f\sim\text{PG}(b,f)\); entscheidend ist:
**bedingt auf \(\omega\) wird es quadratisch in \(f\)**).

Für alle Komponenten zusammen (Vektorform):

- \(\omega=(\omega_1,\dots,\omega_N)^\top\)
- \(\kappa=(\kappa_1,\dots,\kappa_N)^\top\)

Dann wird (bis auf Konstante)

\[
\log L(f\mid n,k,\omega)
=-\frac12 f^\top \operatorname{diag}(\omega)\,f + \kappa^\top f + c.
\]

**Kommentar:**  
„mit \(\omega\) quadratisch in \(f\)“.

---

## 5) Log-Posterior mit Prior zusammenfassen

Prior:

\[
\log p(f)= -\frac12 (f-\mu)^\top \Sigma^{-1}(f-\mu)+c.
\]

Addiere Likelihood-Teil:

\[
\log p(f\mid \omega,n,k)=-\frac12 f^\top \operatorname{diag}(\omega)\,f + \kappa^\top f -\frac12 (f-\mu)^\top \Sigma^{-1}(f-\mu) + c.
\]

### Prior-Term ausmultiplizieren

\[
(f-\mu)^\top\Sigma^{-1}(f-\mu)
=f^\top\Sigma^{-1}f -2f^\top\Sigma^{-1}\mu + \mu^\top\Sigma^{-1}\mu.
\]

Damit (Konstante \(\mu^\top\Sigma^{-1}\mu\) in \(f\) ignorierbar, quadratische Terme zusammenziehen):

\[
-\frac12 (f-\mu)^\top\Sigma^{-1}(f-\mu)
=-\frac12 f^\top\Sigma^{-1}f + f^\top\Sigma^{-1}\mu + c.
\]

Einsetzen:

\[
\log p(f\mid \omega,n,k)
=-\frac12 f^\top(\Sigma^{-1}+\operatorname{diag}(\omega))f + f^\top(\Sigma^{-1}\mu+\kappa) + c.
\]

---

## 6) Identifikation als Gauß-Verteilung

Setze:

\[
Q := \Sigma^{-1}+\operatorname{diag}(\omega),
\qquad
h := \Sigma^{-1}\mu+\kappa.
\]

Dann

\[
\log p(f\mid \omega,n,k)
=-\frac12 f^\top Q f + f^\top h + c.
\]

Da \(Q\) symmetrisch und (typisch) positiv definit ist, ist das eine Gauß-Form.

### Gradient-Schritt

Für symmetrisches \(Q\):

\[
\nabla_f\left(\frac12 f^\top Q f\right)=Qf,
\qquad
\nabla_f(f^\top h)=h.
\]

Also

\[
\nabla_f \log p(f\mid \omega,n,k)= -Qf + h.
\]

Maximum erfüllt:

\[
-Qf+h=0 \quad\Rightarrow\quad Qf=h \quad\Rightarrow\quad f=Q^{-1}h.
\]

Definiere

\[
m := Q^{-1}h.
\]

Dann durch quadratisches Ergänzen:

\[
-\frac12 f^\top Q f + f^\top h
=-\frac12 (f-m)^\top Q (f-m) + c.
\]

**Posterior (bedingt auf \(\omega\) und \(k\))**:

\[
f\mid \omega,n,k \sim \mathcal N\!\left(m,\;Q^{-1}\right),
\]
mit

\[
Q=\Sigma^{-1}+\operatorname{diag}(\omega),
\qquad
m=Q^{-1}(\Sigma^{-1}\mu+\kappa).
\]

---

## 7) Zusatz: quadratisches Ergänzen 

Start:

\[
-\frac12 f^\top Q f + f^\top h.
\]

Ansatz:

\[
-\frac12(f-m)^\top Q (f-m) + c.
\]

Ausmultiplizieren:

\[
(f-m)^\top Q (f-m)
= f^\top Q f -2 f^\top Q m + m^\top Q m.
\]

Also:

\[
-\frac12(f-m)^\top Q (f-m)
=-\frac12 f^\top Q f + f^\top Q m -\frac12 m^\top Q m.
\]

Koeffizientenvergleich mit \(-\frac12 f^\top Q f + f^\top h\) liefert:

\[
f^\top h \equiv f^\top Q m
\quad\Rightarrow\quad
h=Qm
\quad\Rightarrow\quad
m=Q^{-1}h.
\]

---
## Gradient
### Ableitung Sigmoid


Sigmoid-Funktion

\[
\sigma(x) = \frac{1}{1 + e^{-x}} = (1 + e^{-x})^{-1}.
\]

---
Kettenregel

Funktion als Potenz:

\[
\sigma(x) = (1 + e^{-x})^{-1}.
\]

Setze:

\[
u(x) = 1 + e^{-x},
\qquad
\sigma(x) = u(x)^{-1}.
\]

Dann gilt:

\[
u'(x) = -e^{-x}.
\]

Ableitung mit der Kettenregel:

\[
\sigma'(x)
= -1 \cdot u(x)^{-2} \cdot u'(x)
= - (1 + e^{-x})^{-2} \cdot (-e^{-x})
\]

also

\[
\sigma'(x)
= \frac{e^{-x}}{(1 + e^{-x})^{2}}.
\]

Umformung mit der Sigmoid-Funktion


\[
\sigma(x) = \frac{1}{1 + e^{-x}},
\qquad
1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}}.
\]

Damit folgt:

\[
\sigma'(x)
= \frac{e^{-x}}{1 + e^{-x}} \cdot \frac{1}{1 + e^{-x}}
= (1 - \sigma(x))\,\sigma(x).
\]

Symmetrische Darstellung

Es gilt

\[
1 - \sigma(x) = \sigma(-x),
\]

DAraus folgt

\[
\boxed{
\sigma'(x) = \sigma(x)\,\sigma(-x)
}
\]

oder
\[
\boxed{
\sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)
}
\]

Diese Form ist besonders nützlich in:
- logistischer Regression
- Pólya–Gamma-Augmentation
- Gradienten von Log-Likelihoods