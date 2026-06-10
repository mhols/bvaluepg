# Laplace-Randbedingungen im Matern-Precision-Code

Datum: 2026-06-09  



In `source/covariance_kernels.py` wurde der 5-Punkt-Laplace-Anteil aus `precision_matern` in zwei explizite Hilfsfunktionen ausgelagert:

```python
laplacian_1d(n, boundary="zero")
laplacian_2d(ny, nx, boundary="zero")
```

`precision_matern` hat jetzt zusaetzlich den optionalen Parameter:

```python
precision_matern(n, m, rho, v2, boundary="zero")
```

Der Default `boundary="zero"` erhaelt das bisherige Verhalten. Neu ist `boundary="symmetric"` fuer eine Neumann-artige Randbehandlung, bei der die Randwerte nach aussen gespiegelt werden:

```math
u_{-1} = u_0,
\qquad
u_n = u_{n-1}.
```

Fuer ein 2D-Bild `A` mit Zeilen-/Spaltenindizes gilt entsprechend:

```math
A_{\mathrm{ext}}(-1, j) = A(0, j),
\qquad
A_{\mathrm{ext}}(ny, j) = A(ny-1, j),
```

und analog in x-Richtung:

```math
A_{\mathrm{ext}}(i, -1) = A(i, 0),
\qquad
A_{\mathrm{ext}}(i, nx) = A(i, nx-1).
```

Diese Variante entspricht einer Spiegelung mit mitgenommener Randzelle. In NumPy-Sprache liegt sie naeher an `symmetric` als an `reflect`.

## Motivation

Der bisherige 5-Punkt-Laplace-Operator behandelt Werte ausserhalb der Matrix implizit wie Nullwerte. Das ist fuer einige Differentialgleichungsprobleme sinnvoll, aber fuer das b-Wert-Feld auf einem kuenstlich ausgeschnittenen Kartenfenster problematisch:

- Der Kartenrand ist kein physikalischer Nullrand.
- Das Modellfenster endet dort nur aus technischen oder datenpraktischen Gruenden.
- Eine Null-Fortsetzung zieht Randzellen kuenstlich gegen Null.
- Randzellen werden dadurch anders regularisiert als innere Zellen.

Die neue `symmetric`-Randbehandlung setzt stattdessen voraus, dass das Feld unmittelbar ausserhalb des Fensters lokal mit dem Randwert weiterlaeuft. Dadurch wird am Rand keine kuenstliche Nullwelt erzeugt.

## Indexkonvention

Im Projekt gilt fuer 2D-Felder:

```python
image.shape == (ny, nx)
```

Der erste Index ist die Zeile bzw. y-Richtung, der zweite Index ist die Spalte bzw. x-Richtung:

```math
A(i,j) = A[\text{Zeile } i, \text{Spalte } j].
```

Die Scan-Order ist Row-Major:

```python
image.ravel(order="C")
```

Beispiel:

```text
A =
a b c
d e f
g h i
```

Dann gilt:

```text
A(0,0) = a
A(0,1) = b
A(0,2) = c
A(1,0) = d
A(1,1) = e
A(1,2) = f
A(2,0) = g
A(2,1) = h
A(2,2) = i
```

## Symmetric Boundary

Fuer die gewuenschte Spiegelung mit Randzelle wird bei Padding-Breite 1 aus

```text
a b c
d e f
g h i
```

gedanklich:

```text
a a b c c
a a b c c
d d e f f
g g h i i
g g h i i
```

Diese Darstellung ist nur die anschauliche Extension. Im Code wird kein grosses gepaddetes Array aufgebaut. Stattdessen wird direkt die passende sparse Laplace-Matrix konstruiert.

Am linken Rand in 1D:

```math
(Lu)_0 = 2u_0 - u_{-1} - u_1.
```

Mit `boundary="symmetric"` gilt:

```math
u_{-1} = u_0.
```

Damit:

```math
(Lu)_0 = 2u_0 - u_0 - u_1 = u_0 - u_1.
```

Am rechten Rand analog:

```math
(Lu)_{n-1} = u_{n-1} - u_{n-2}.
```

Die 1D-Matrix fuer `n=3` ist:

```math
L_{\mathrm{symmetric}} =
\begin{pmatrix}
1 & -1 & 0 \\
-1 & 2 & -1 \\
0 & -1 & 1
\end{pmatrix}.
```

Ein konstantes Feld liegt im Nullraum dieses Laplace-Operators:

```math
L_{\mathrm{symmetric}} \mathbf{1} = 0.
```

Das ist der wichtigste schnelle Test fuer die Randbedingung.

## Zero Boundary

Der bisherige Default bleibt:

```python
boundary="zero"
```

Dabei gilt ausserhalb:

```math
u_{-1} = 0,
\qquad
u_n = 0.
```

Am linken Rand:

```math
(Lu)_0 = 2u_0 - 0 - u_1 = 2u_0 - u_1.
```

Die 1D-Matrix fuer `n=3` ist:

```math
L_{\mathrm{zero}} =
\begin{pmatrix}
2 & -1 & 0 \\
-1 & 2 & -1 \\
0 & -1 & 2
\end{pmatrix}.
```

Ein konstantes Feld wird am Rand nicht zu Null abgebildet, weil die kuenstlichen Nullwerte ausserhalb sichtbar werden.

## Umsetzung im Code

Die neue 1D-Hilfsfunktion baut den positiven diskreten Laplace-Operator:

```python
def laplacian_1d(n, boundary="zero"):
    main = 2.0 * np.ones(n)
    off = -np.ones(n - 1)

    if boundary == "zero":
        pass
    elif boundary == "symmetric":
        main[0] = 1.0
        main[-1] = 1.0
    else:
        raise ValueError(f"unknown boundary: {boundary}")

    return sps.diags([off, main, off], [-1, 0, 1], shape=(n, n), format="csr")
```

Der 2D-Laplace-Operator wird danach mit Kronecker-Produkten gebaut:

```python
def laplacian_2d(ny, nx, boundary="zero"):
    Ly = laplacian_1d(ny, boundary=boundary)
    Lx = laplacian_1d(nx, boundary=boundary)
    Iy = sps.eye(ny, format="csr")
    Ix = sps.eye(nx, format="csr")

    return (
        sps.kron(Ly, Ix, format="csr")
        + sps.kron(Iy, Lx, format="csr")
    )
```

`precision_matern` verwendet jetzt:

```python
laplacian = laplacian_2d(n, m, boundary=boundary)
```

Der restliche Matern-Pfad bleibt unveraendert:

```python
alpha = 0.5 / (np.cosh(1/rho) - 1)
Q = I + alpha L
Q = Q.T @ Q
```

Danach wird wie bisher ueber die mittlere Zelle auf `v2` normalisiert.


```text
A =
a b c
d e f

A.ravel(order="C") = [a, b, c, d, e, f]
```

Deshalb koppelt

```math
I_y \otimes L_x
```

die horizontalen Nachbarn innerhalb jeder Zeile, und

```math
L_y \otimes I_x
```

koppelt vertikale Nachbarn zwischen den Zeilen.

Der komplette 2D-Operator ist:

```math
L_{2D} = L_y \otimes I_x + I_y \otimes L_x.
```

Fuer einen inneren Punkt bleibt der normale 5-Punkt-Stencil:

```text
 0  -1   0
-1   4  -1
 0  -1   0
```

Fuer die linke obere Ecke im `3 x 3`-Beispiel gilt bei `boundary="symmetric"`:

```math
(LA)_{0,0}
= (a - b) + (a - d)
= 2a - b - d.
```

Bei `boundary="zero"` waere es:

```math
(LA)_{0,0}
= (2a - b) + (2a - d)
= 4a - b - d.
```

Der Unterschied ist genau der kuenstliche Nullrand.


## API-Entscheidung

Aktuell sind zwei Boundary-Werte implementiert:

```text
zero
symmetric
```

`reflect` wurde bewusst noch nicht implementiert. Der Name ist leicht missverstaendlich, weil zwei verschiedene Spiegelkonventionen existieren:

```math
\text{symmetric: } u_{-1} = u_0
```

gegenueber

```math
\text{reflect: } u_{-1} = u_1.
```

Die jetzt implementierte Variante ist die gewuenschte Spiegelung mit mitgenommener Randzelle.

