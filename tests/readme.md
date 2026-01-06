# Liste sinnvoller Tests für `polyagammadensity.py`

Diese Liste dient als **Checkliste** für die Implementierung von Tests mit `pytest`.
Sie ist speziell auf das Modul `polyagammadensity.py` zugeschnitten.

---

## 1. `sigmoid(f)`

**Unit Tests**
- Rückgabewert liegt immer in `[0, 1]`
- `sigmoid(0) == 0.5`
- Vektor-Input: Output hat gleiche Shape wie Input

**Randfälle / Numerik**
- `f = 1000` → kein `NaN`, Ergebnis nahe `1`
- `f = -1000` → kein `NaN`, Ergebnis nahe `0`

---

## 2. `nbins` (Property)

**Unit Tests**
- `nbins == len(prior_mean)`
- funktioniert für verschiedene Dimensionen

---

## 3. `set_data(nobs)`

**Unit Tests**
- Happy Path: `len(nobs) == nbins` setzt `self.nobs`
- Fehlerfall: falsche Länge → `AssertionError`

**Randfälle**
- `nobs = np.zeros(nbins)` ist erlaubt

---

## 4. `Lprior` (Property, Lazy Evaluation)

**Unit Tests**
- `Lprior @ Lprior.T ≈ prior_covariance`
- Mehrfacher Zugriff liefert denselben Wert (Caching)

**Fehlerfälle**
- nicht positiv definite Kovarianz → `LinAlgError`

> Hinweis: Lazy-Check sollte auf `None` prüfen (`if self._Lprior is None`).

---

## 5. `random_prior_prameters()`

**Unit Tests**
- Rückgabe ist `np.ndarray`
- Länge entspricht `nbins`

**Zufall / Reproduzierbarkeit**
- gleicher Seed → gleiche Ausgabe
- unterschiedlicher Seed → unterschiedliche Ausgabe

---

## 6. `random_prior_field()`

**Unit Tests**
- Shape == `nbins`
- Wertebereich: `0 ≤ pf ≤ lam`

**Randfälle**
- `lam = 0` → alle Werte sind `0`

---

## 7. `random_prior()`

**Unit Tests**
- Länge == `nbins`
- alle Werte sind ganzzahlig und ≥ 0

**Zufall**
- deterministisch bei gesetztem Seed

---

## 8. `loglikelihood(f)`

**Unit Tests**
- Rückgabe ist ein Skalar
- `nobs = 0` → Ergebnis entspricht `-sum(pf)`
- moderate Inputs → Ergebnis ist endlich (`np.isfinite`)

**Randfälle**
- `pf → 0` bei `nobs > 0` → `-inf` (bewusstes Verhalten)

---

## 9. `logposterior(f)`

**Unit Tests**
- Methode ist ausführbar ohne Crash
- Rückgabe ist Skalar

**Regressionstest**
- Absicherung gegen falsche Verwendung von `np.solve`
- Korrekt: `np.linalg.solve(...)`

---

## 10. Mini-Integrationstest

**Ziel**
- Zusammenspiel der Kernmethoden prüfen

**Ablauf**
- Objekt erzeugen
- `set_data`
- `random_prior_prameters`
- `logposterior`

**Erwartung**
- kein Crash
- Ergebnis endlich oder dokumentiert (`-inf`)

---

## Testarten (Übersicht)

- Unit Tests (jede Methode)
- Randfall- & Numeriktests
- Zufall / Seed-Tests
- Regressionstests
- Mini-Integrationstests

---

**Merksatz:**  
*Unit Tests prüfen Methoden – Integrationstests prüfen Annahmen.*
