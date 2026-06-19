# Mathematische Spezifikation des 5-Punkt-Precision-Priors

**Status:** Entwurf fuer die verbindliche Referenzimplementation  
**Geltungsbereich:** regulaere rechteckige 2D-Gitter, 5-Punkt-Laplacian  
**Nicht enthalten:** 9-Punkt-Stencil, Quadtree-Gitter, Masken und physikalisch
skalierte Koordinatengitter

## 1. Zweck und verbindliche Notation

Dieses Dokument trennt die drei mathematischen Objekte, die im bisherigen Code
teilweise unter demselben Namen `Q` zusammengefasst werden:

\[
L := \text{positiver diskreter 5-Punkt-Laplacian},
\]

\[
A := I + \alpha L,
\]

\[
P_0 := A^\top A.
\]

Dabei ist \(P_0\) die unskalierte Prior-Precision. Nach der
Varianznormierung wird die endgueltige Precision mit \(P\) bezeichnet:

\[
f \sim \mathcal N(\mu,P^{-1}).
\]

In Code, Dokumentation, Tests und Paper sollen diese Bezeichnungen dieselbe
Bedeutung haben. Insbesondere soll weder \(A\) noch \(P\) allgemein nur `Q`
genannt werden.

## 2. Gitter und Vektorisierung

Das Feld hat die Form

\[
U=(u_{i,j})\in\mathbb R^{n_y\times n_x},
\qquad
0\leq i<n_y,\quad 0\leq j<n_x.
\]

Die Vektorisierung ist NumPy-C-Order (Row-Major):

```python
u = U.ravel(order="C")
U = u.reshape((ny, nx), order="C")
```

Der lineare Index der Zelle \((i,j)\) ist daher

\[
r(i,j)=i n_x+j.
\]

Die x-Richtung ist der schnelle Index. Entsprechend lautet der 2D-Operator

\[
L=L_y\otimes I_x+I_y\otimes L_x.
\]

Diese Reihenfolge ist fuer nichtquadratische Gitter beizubehalten.

## 3. Positiver 1D-Laplacian

Im Inneren gilt fuer beide Randbedingungen

\[
(L_1u)_i=2u_i-u_{i-1}-u_{i+1}.
\]

### 3.1 Nullfortsetzung (`boundary="zero"`)

Die Ghost-Werte werden auf null gesetzt:

\[
u_{-1}=0,
\qquad
u_n=0.
\]

Damit ist

\[
L_{1,\mathrm{zero}}=
\begin{pmatrix}
2&-1&&0\\
-1&2&\ddots&\\
&\ddots&\ddots&-1\\
0&&-1&2
\end{pmatrix}.
\]

Dies ist eine Dirichlet-artige Nullfortsetzung. Ein konstantes Feld liegt
nicht im Nullraum. Bei einer kuenstlich ausgeschnittenen Kartenregion ist
dies eine modellierende Randannahme und nicht nur ein numerisches Detail.

### 3.2 Symmetrische Fortsetzung (`boundary="symmetric"`)

Die Randzelle selbst wird gespiegelt:

\[
u_{-1}=u_0,
\qquad
u_n=u_{n-1}.
\]

Damit ist

\[
L_{1,\mathrm{symmetric}}=
\begin{pmatrix}
1&-1&&0\\
-1&2&\ddots&\\
&\ddots&\ddots&-1\\
0&&-1&1
\end{pmatrix}.
\]

Dies ist eine diskrete homogene Neumann-artige Randbedingung. Es gilt

\[
L_{1,\mathrm{symmetric}}\mathbf 1=0.
\]

## 4. Positiver 2D-Laplacian

Mit derselben Randbedingung in beiden Richtungen wird

\[
L=L_y\otimes I_x+I_y\otimes L_x
\]

definiert. Fuer eine innere Zelle gilt

\[
(LU)_{i,j}
=4u_{i,j}-u_{i-1,j}-u_{i+1,j}-u_{i,j-1}-u_{i,j+1}.
\]

Der innere Stencil ist somit

```text
 0  -1   0
-1   4  -1
 0  -1   0
```

Fuer `symmetric` liegt das konstante 2D-Feld im Nullraum von \(L\). Fuer
`zero` gilt dies nicht.

Diese Spezifikation setzt Gitterabstand \(h_x=h_y=1\) voraus. Bei anderen
oder anisotropen Abstaenden muessen die Richtungsoperatoren durch
\(h_y^{-2}\) beziehungsweise \(h_x^{-2}\) skaliert werden. Die Parameter
\(n_y,n_x\) beschreiben nur die Gitterdimension und nicht den physikalischen
Zellabstand.

## 5. Basisoperator und unskalierte Precision

Fuer

\[
\rho>0
\]

wird im aktuellen Modell

\[
\alpha(\rho)
=\frac{1}{2\left(\cosh(1/\rho)-1\right)}
\]

gesetzt. Danach werden

\[
A=I+\alpha L
\]

und

\[
P_0=A^\top A
\]

gebildet. Da \(L\) symmetrisch ist, gilt zwar \(A^\top A=A^2\); die Form
\(A^\top A\) bleibt jedoch die verbindliche Definition.

Fuer \(\alpha>0\) ist \(A\) unter beiden Randbedingungen symmetrisch positiv
definit. Daher ist auch \(P_0\) symmetrisch positiv definit. Das gilt bei
`symmetric` trotz des konstanten Nullmodus von (L), weil der
Identitaetsanteil in (A) diesen Modus auf Eigenwert eins abbildet.

## 6. Herkunft und begrenzte Bedeutung der `rho`-Skalierung

Auf einem unendlichen beziehungsweise periodischen Einheitsgitter hat der
5-Punkt-Laplacian das Fourier-Symbol

\[
\ell(k_x,k_y)
=4-2\cos(k_x)-2\cos(k_y).
\]

Das Symbol von (A) ist

\[
a(k_x,k_y)=1+\alpha\ell(k_x,k_y),
\]

und das Symbol der unskalierten Precision \(P_0\) ist

\[
p_0(k_x,k_y)=a(k_x,k_y)^2.
\]

Entlang der x-Achse, also fuer \(k_y=0\), gilt

\[
a(k,0)=1+2\alpha(1-\cos k).
\]

Mit

\[
\gamma=e^{-1/\rho}
\]

kann dieser Ausdruck bis auf einen konstanten Faktor an

\[
1-2\gamma\cos k+\gamma^2
\]

angepasst werden. Der Koeffizientenvergleich liefert die oben verwendete
Formel fuer \(\alpha(\rho)\).

### Wichtige Einschraenkung

Dieser Vergleich kalibriert den **unquadrierten Operator (A)** entlang
einer Koordinatenachse. Der hier definierte Prior verwendet jedoch

\[
P_0=A^\top A
\]

und besitzt deshalb auf dem idealisierten Gitter das Kovarianzspektrum

\[
\widehat\Sigma_0(k_x,k_y)
=\frac{1}{a(k_x,k_y)^2}.
\]

Die geometrische 1D-Korrelation, aus der die `cosh`-Formel hergeleitet wird,
hat dagegen ein Spektrum proportional zu \(1/a(k)\), nicht \(1/a(k)^2\).
Daher darf `rho` fuer den finalen Prior \(P_0^{-1}\) vorerst nur als
**Operator-Skalenparameter** bezeichnet werden. Es ist ohne weitere
Herleitung oder numerische Kalibrierung nicht belegt, dass `rho` exakt die
e-Folding-Korrelationslaenge der finalen 2D-Kovarianz ist.

Endliche Gitter und Randbedingungen veraendern die Kovarianz zusaetzlich.
Bis diese Frage geklaert ist, soll weder im Code noch im Paper eine exakte
Matérn-Kovarianz oder eine exakt realisierte Korrelationslaenge behauptet
werden. Die sachlich sichere Bezeichnung ist **Matérn-artiger
5-Punkt-Precision-Prior**.

## 7. Normierung auf die Zielvarianz `v2`

Es sei

\[
\Sigma_0=P_0^{-1}.
\]

Als Referenzzelle wird

\[
(i_\star,j_\star)
=\left(\left\lfloor\frac{n_y}{2}\right\rfloor,
       \left\lfloor\frac{n_x}{2}\right\rfloor\right)
\]

mit linearem Index

\[
r_\star=i_\star n_x+j_\star
\]

verwendet. Fuer gerade Gitterdimensionen ist dies eine eindeutig festgelegte
der mittleren Zellen.

Der Delta-Vektor \(e_\star\) ist definiert durch

\[
(e_\star)_r=
\begin{cases}
1,&r=r_\star,\\
0,&r\neq r_\star.
\end{cases}
\]

Die unskalierte marginale Varianz an der Referenzzelle ist

\[
s_\star
=e_\star^\top P_0^{-1}e_\star.
\]

Sie wird numerisch durch Loesen des linearen Systems

\[
P_0x=e_\star
\]

und anschliessendes Auslesen

\[
s_\star=(e_\star)^\top x=x_{r_\star}
\]

bestimmt. Eine dichte Inverse soll dafuer nicht gebildet werden.

Die endgueltige Precision lautet

\[
\boxed{
P=\frac{s_\star}{v^2}P_0
}
\qquad\text{mit}\qquad v^2>0.
\]

Denn

\[
P^{-1}=\frac{v^2}{s_\star}P_0^{-1}
\]

und damit

\[
e_\star^\top P^{-1}e_\star=v^2.
\]

Ein Einsvektor mit einer zentralen Null ist **kein** Delta-Impuls und kann
nicht zur Normierung der marginalen Varianz verwendet werden.

Die Normierung garantiert nur die Varianz der Referenzzelle. Bei endlichen
Gittern koennen die marginalen Varianzen anderer Zellen wegen der
Randbedingung davon abweichen. Diese Abweichungen sind zu messen und zu
dokumentieren.

## 8. Verbindliches Prior-Modell

Nach den obigen Definitionen lautet das Prior-Modell

\[
f\mid\mu,\rho,v^2,\mathcal B
\sim
\mathcal N\!\left(\mu,
P(\rho,v^2,\mathcal B)^{-1}\right),
\]

wobei \(\mathcal B\in\{\texttt{zero},\texttt{symmetric}\}\) die
Randbedingung bezeichnet.

Die Parameter haben damit folgende gesicherte Bedeutung:

- `mu`: Mittelwert des latenten Feldes;
- `v2`: marginale Varianz an der festgelegten Referenzzelle;
- `rho`: Skalenparameter von \(A=I+\alpha(\rho)L\), noch nicht als exakte
  Korrelationslaenge des finalen Priors validiert;
- `boundary`: modellierende Annahme zur Fortsetzung des Feldes ausserhalb des
  rechteckigen Gitters.

## 9. Vorgeschlagene reine Builder-API

Die spaetere Implementierung soll die mathematischen Ebenen sichtbar halten.
Eine moegliche API ist:

```python
laplacian_2d(ny, nx, boundary="zero") -> L

matern_basis_operator_5pt(
    ny, nx, rho, boundary="zero"
) -> A

matern_precision_5pt_unscaled(
    ny, nx, rho, boundary="zero"
) -> P0

matern_precision_5pt(
    ny, nx, rho, v2, boundary="zero"
) -> P
```

Alle Builder muessen ohne Plot, Zufallsziehung, globale Zustandsaenderung oder
interaktive Seiteneffekte arbeiten. Diagnostik und Visualisierung gehoeren in
separate Experimentfunktionen.

Die konkreten Funktionsnamen sind noch eine API-Entscheidung. Die Trennung von
\(L\), \(A\), \(P_0\) und \(P\) ist dagegen Teil dieser mathematischen
Spezifikation.

## 10. Verbindliche mathematische Tests

Vor einer Nutzung fuer Paper-Ergebnisse muessen mindestens folgende Tests
vorliegen:

1. **Dimension und Scan-Order**
   - rechteckige Gitter `(3, 4)` und `(4, 3)`;
   - Vergleich der Matrixanwendung mit einer direkten 2D-Stencil-Anwendung.

2. **Randbedingungen**
   - `symmetric`: \(L\mathbf1=0\);
   - `zero`: \(L\mathbf1\neq0\);
   - explizite Werte fuer Innenpunkt, Kante und Ecke.

3. **Symmetrie und Definitheit**
   - \(L=L^\top\);
   - \(A=A^\top\) und alle Eigenwerte von \(A\) positiv;
   - \(P_0=P_0^\top\) und alle Eigenwerte von \(P_0\) positiv.

4. **Algebraische Zerlegung**
   - numerisch \(P_0=A^\top A\);
   - getrennte Tests verhindern eine versehentliche zweite Quadrierung.

5. **Varianznormierung**
   - Loesen von \(Px=e_\star\);
   - Test \(x_{r_\star}=v^2\) bis auf numerische Toleranz;
   - Test mit mindestens einem rechteckigen Gitter.

6. **Parameterverhalten**
   - \(\alpha(\rho)>0\) fuer \(\rho>0\);
   - \(\alpha(\rho)\) waechst monoton mit \(\rho\);
   - groessere `rho` erzeugen im numerischen Kovarianzvergleich eine groessere
     raeumliche Reichweite, ohne bereits exakte Gleichheit von `rho` und
     e-Folding-Laenge zu behaupten.

7. **Randempfindlichkeit**
   - marginale Varianzen in Zentrum, Kante und Ecke vergleichen;
   - `zero` und `symmetric` getrennt dokumentieren.

## 11. Noch offene mathematische Entscheidung

Vor der finalen Freigabe muss entschieden werden, welche Bedeutung `rho` im
Paper haben soll:

1. **Operator-Skalenparameter:** Die aktuelle Formel und \(P=A^\top A\)
   bleiben bestehen. `rho` wird nicht als exakt realisierte
   Korrelationslaenge bezeichnet.

2. **Tatsaechliche Korrelationslaenge:** `alpha` wird fuer den finalen Prior
   so analytisch oder numerisch kalibriert, dass eine vorher exakt definierte
   Korrelationsmetrik den Zielwert `rho` erreicht.

3. **Anderes Matérn/SPDE-Modell:** Ordnung und Skalierung des Operators werden
   aus einer festgelegten Matérn- beziehungsweise SPDE-Parametrisierung neu
   hergeleitet.

Bis zu dieser Entscheidung ist Option 1 die einzig vollstaendig durch den
aktuellen Operator gedeckte Interpretation.

## 12. Abgrenzung zum 9-Punkt-Stencil

Aus dieser Spezifikation folgt keine 9-Punkt-Precision. Insbesondere duerfen
die `rho`-Skalierung, Randbehandlung oder Varianznormierung nicht ungeprueft auf
einen 9-Punkt-Stencil uebertragen werden. Dessen Koeffizienten, Fourier-Symbol,
Randoperator und Kalibrierung sind separat herzuleiten und zu testen.

## 13. Numerischer Validierungsstand vom 19. Juni 2026

Das deterministische Experiment
`experiments/exp_validate_precision_matern_5pt.py` wertet die Kovarianz
\(P_0^{-1}\) auf einem \(41\times41\)-Gitter fuer

\[
\rho\in\{0.5,1,2,4,8\}
\]

und beide Randbedingungen aus. Gemessen wird die erste linear interpolierte
Distanz, bei der die Pearson-Korrelation auf \(1/e\) faellt. Das Skript gibt
die Messwerte direkt im Terminal aus und zeigt die Korrelationsprofile
interaktiv mit Matplotlib an.

Der Referenzlauf zeigt:

- Fuer \(\rho=1,2,4\) liegt die effektive Achsenlaenge ungefaehr zwischen
  \(1.55\rho\) und \(1.62\rho\), nicht bei \(\rho\).
- Fuer \(\rho=8\) ist der Einfluss des endlichen Gitters sichtbar. Die
  effektive Achsenlaenge betraegt ungefaehr \(1.57\rho\) bei `zero` und
  \(1.80\rho\) bei `symmetric`.
- Fuer \(\rho=0.5\) liegt die Zielskala unterhalb der Gitteraufloesung; die
  interpolierte Laenge ist daher nur eingeschraenkt quantitativ belastbar.
- Achsen- und Diagonalprofile stimmen fuer mittlere `rho` weitgehend ueberein.
  Kleine Abweichungen entstehen durch Gitterdiskretisierung und bei grossen
  `rho` durch die Randbedingung.

Damit ist numerisch widerlegt, dass der Eingabeparameter `rho` in der aktuellen
Konstruktion allgemein die tatsaechliche \(1/e\)-Korrelationslaenge von
\(P^{-1}\) ist. Bis zu einer separaten Kalibrierung gilt verbindlich die
Interpretation als **Operator-Skalenparameter**.
