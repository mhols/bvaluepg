# Toni Notes: SPDE-ETAS

Stand: 15. September 2026

## Zweck des Projekts

Das Projekt fuehrt eine bayesianische Schaetzung eines raeumlichen ETAS-Modells fuer Erdbebenkataloge durch. Ereignisse werden im MCMC latent entweder als Hintergrundereignisse oder als Nachkommen frueherer Ereignisse behandelt. Die raeumlich variable Hintergrundrate wird als Log-Gaussian-Cox-Process modelliert und ueber eine SPDE auf einem Dreiecksnetz approximiert.

Der Einstiegspunkt ist `Main_script.jl`. Das Skript

1. liest einen synthetischen Katalog ein,
2. erzeugt ein Dreiecksnetz fuer das Gebiet `[0,5] x [0,5]`,
3. baut die SPDE-Matrizen auf,
4. startet den ETAS-SPDE-MCMC-Sampler und
5. schreibt Parameterketten, Intensitaeten, Intensitaetsquantile und die Anzahl der Hintergrundereignisse nach `mcmc_results/`.

Das Projekt erzeugt derzeit keine Plots. Die exportierten Textdateien sind die Grundlage fuer spaetere Trace-Plots und raeumliche Intensitaetskarten.

## Projekt lokal starten

Aus dem Projektverzeichnis:

```bash
cd "/Users/toni/Library/Mobile Documents/com~apple~CloudDocs/UP/BvaluePG/bvaluepg/SPDE-ETAS-main"
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. Main_script.jl
```

`--project=.` aktiviert die Abhaengigkeiten aus der lokalen `Project.toml`. Das vorherige `cd` ist ebenfalls notwendig, weil Daten- und Include-Pfade im Skript relativ zum aktuellen Arbeitsverzeichnis angegeben sind.

Lokal getestet wurde mit Julia 1.12.6, Optim 2.2.1 und NLSolversBase 8.0.0.

## Urspruengliche Probleme

### Relativer Datenpfad

Beim Start aus einem anderen Arbeitsverzeichnis trat folgender Fehler auf:

```text
Cannot open 'data/synthetic_data_case_01_patches.txt': not a file
```

Ursache: `DATA_FILE` ist relativ zum aktuellen Arbeitsverzeichnis. Der Lauf muss daher aus dem Projektstamm gestartet werden. Eine zukuenftige Verbesserung waere, Dateipfade mit `@__DIR__` am Ort von `Main_script.jl` auszurichten.

### Inkompatibilitaet mit Optim 2.x

Der Code verwendete `Optim.only_fgh!`. In der aktuellen Paketstruktur gehoert diese Funktion zu `NLSolversBase`. Nach der ersten Anpassung zeigte sich ausserdem, dass Optim 2.x fuer den benutzerdefinierten Cholesky-Wrapper eine vollstaendigere Matrix- und Kopierschnittstelle erwartet.

Umgesetzt wurden:

- `NLSolversBase` als direkte Abhaengigkeit in `Project.toml`;
- `using NLSolversBase` in `src/sampling_utilities.jl`;
- Aufruf von `NLSolversBase.only_fgh!(fgh!)`;
- Implementierungen von `size`, `similar`, `copy` und `copyto!` fuer `OptimCholWrapper`;
- Umstellung des Wrappers auf `mutable struct`, damit ein kopierter CHOLMOD-Faktor gezielt eingesetzt werden kann.

Der Wrapper behaelt dabei die duennbesetzte Cholesky-Struktur bei; die Hessian wird nicht unnoetig in eine dichte Matrix umgewandelt.

### Nichtinteraktive TriangleMesh-Konfiguration

`set_area_max=true` weist `TriangleMesh` an, den Wert interaktiv von stdin abzufragen. Bei einem normalen Skriptlauf kam keine Eingabe an. Das Paket meldete deshalb:

```text
Area must be a positive real number
```

Ausserdem waren die vom Paket erzeugten leeren Default-Matrizen fuer Punktmarker und Punktattribute in dieser Version unpassend orientiert. Dadurch erschien die Meldung:

```text
Number of point markers > 1. Only 0 or 1 admissible
```

Die Netzkonfiguration ist jetzt vollstaendig nichtinteraktiv:

```julia
const MAX_TRIANGLE_AREA = 0.5

point_marker = zeros(Int, size(corners, 1), 0)
point_attribute = zeros(Float64, size(corners, 1), 0)
set_area_max = false
add_switches = "a$(MAX_TRIANGLE_AREA)"
```

Verglichene maximale Dreiecksflaechen ergaben:

| Maximale Flaeche | Knoten | Dreiecke |
|---:|---:|---:|
| 1.00 | 26 | 34 |
| 0.75 | 39 | 59 |
| 0.50 | 45 | 71 |
| 0.40 | 57 | 88 |
| 0.30 | 79 | 124 |
| 0.25 | 89 | 144 |

Gewaehlt wurde `0.5`, weil die vorher vorhandene, aber wirkungslose Konstante `GRID_POINTS = 50` auf eine ungefaehre Zielgroesse von 50 Netzknoten hindeutete. Diese Interpretation ist eine technische Annahme und sollte vor einer wissenschaftlichen Auswertung fachlich bestaetigt werden.

Das Skript meldet die tatsaechliche Netzgroesse nun beim Start:

```text
Mesh: 45 points, 71 triangles
```

## Validierung

Nach den Korrekturen wurde `julia --project=. Main_script.jl` vollstaendig mit der unveraenderten Einstellung `NITER = 100` ausgefuehrt. Der Lauf erreichte Schritt 100 und speicherte alle vorgesehenen Dateien.

Validierte Ausgaben:

| Datei | Inhalt und Dimension |
|---|---|
| `mesh_points_case01_patches.txt` | 45 Netzkoordinaten |
| `mcmc_results/chains_parameters_case01_patches.txt` | Kopfzeile plus 91 Parametersaetze mit je 10 Werten |
| `mcmc_results/chains_intensity_case01_patches.txt` | 91 Samples mit je 45 Intensitaetswerten |
| `mcmc_results/chains_intensity_quantiles_case01_patches.txt` | Kopfzeile plus 45 Netzknoten mit je 5 Quantilen |
| `mcmc_results/chains_nbg_case01_patches.txt` | Kopfzeile plus 91 Werte fuer die Zahl der Hintergrundereignisse |

Bei `BURNIN = 10` verwendet der aktuelle Code den inklusiven Bereich `10:end`. Dadurch bleiben bei 100 Iterationen 91 statt 90 Samples erhalten. Ob dies beabsichtigt ist oder ein Off-by-one-Fehler, ist noch zu klaeren.

## Offene Punkte

- Es gibt noch keine Plot- oder Auswertungsfunktionen im Repository.
- Die fachlich angemessene Netzaufloesung muss bestaetigt werden; `MAX_TRIANGLE_AREA = 0.5` ist eine begruendete technische Startwahl.
- Die aktuelle Demo mit 100 Iterationen und kurzem Burn-in reicht nicht fuer belastbare Posterior-Aussagen.
- Der Zufallsgenerator des Hauptsamplers hat keinen expliziten Seed; Laeufe sind daher nicht exakt reproduzierbar.
- `Project.toml` enthaelt noch keine `[compat]`-Grenzen. Die direkte Abhaengigkeit ist dokumentiert, aber Paketversionen sind ohne mitgegebenes `Manifest.toml` nicht dauerhaft fixiert.
- Eingabe- und Ausgabepfade koennten mit `@__DIR__` unabhaengig vom Arbeitsverzeichnis gemacht werden.
- Vor wissenschaftlicher Interpretation sollten Konvergenzdiagnostik, Trace-Plots, effektive Stichprobengroessen und Sensitivitaet gegenueber Netzaufloesung und Startwerten ergaenzt werden.

## Geplanter Italien-Lauf mit Sofianes Julia-Code

### Warum wir diesen Vergleich machen

Unser bisheriger PG-Ansatz und Sofianes SPDE-ETAS-Modell schaetzen nicht automatisch dieselbe Groesse:

- Der PG-Ansatz modelliert raeumliche Counts auf einem festen Raster. Werden alle Ereignisse verwendet, enthaelt die geschaetzte Flaeche sowohl langfristigen Hintergrund als auch zeitlich konzentrierte Nachbebencluster.
- Die bisherige PG-Hintergrundanalyse entfernt Ereignisse vorher mit NND-Declustering und modelliert dann die Counts der behaltenen Ereignisse.
- Sofianes Modell verwendet Raum, Zeit und Magnitude des vollstaendigen Katalogs gemeinsam. Es ordnet Ereignisse innerhalb des MCMC probabilistisch dem Hintergrund oder einem frueheren Elternereignis zu und schaetzt gleichzeitig eine glatte raeumliche Hintergrundrate.

Der wissenschaftlich interessante Vergleich lautet deshalb nicht schlicht "welches Bild sieht besser aus?", sondern:

1. Findet eine gemeinsame ETAS-SPDE-Schaetzung dieselben stabilen Hintergrundstrukturen wie NND plus PG?
2. Wo schreibt PG auf allen Ereignissen Nachbebencluster faelschlich der Hintergrundrate zu?
3. Wie stark haengt die geschaetzte Hintergrundseismizitaet von der Declustering-Methode ab?
4. Welche raeumlichen Strukturen bleiben ueber Modell, Netzaufloesung, Magnitudenschwelle und Startwerte stabil?
5. Welche ETAS-Parameter und welcher Hintergrundanteil werden fuer den Italien-Katalog plausibel geschaetzt?

Auf echten Italien-Daten gibt es keine bekannte wahre Hintergrundrate. Uebereinstimmung zwischen Methoden ist deshalb Robustheitsevidenz, aber kein Wahrheitsbeweis. RMSE gegen eine bekannte Wahrheit bleibt den synthetischen Experimenten vorbehalten.

### Vorhandene Daten und Experimente

Der aktuelle Hauptworkflow im uebergeordneten Repository ist:

```text
data/preprocess_nnd_rot_cut_bin.py
-> experiments/exp_italy_preprocess_nnd_rot_cut_pg.py
```

Daneben existieren bereits:

- `experiments/exp_sofiane_spde_etas_pg.py`: PG auf Sofianes synthetischen Katalogen, jeweils mit allen Ereignissen und mit den bekannten Background-Ereignissen;
- `data/preprocess_sofiane_spde_etas.py`: Adapter von Sofianes synthetischem Format in das BvaluePG-Katalogformat;
- `data/preprocess_nnd_rot_cut_bin_Mc_2.5_eta_-4.60_dkm_20_events.csv`: zeitlich sortierter Italien-Katalog mit projizierten und rotierten Kilometerkoordinaten sowie NND-Status;
- zugehoerige Count-, Bin- und Metadateien fuer das 20-km-PG-Raster.

Der vorhandene Mc-2.5-Katalog umfasst:

| Auswahl | Ereignisse |
|---|---:|
| gesamter gefilterter Katalog | 11.435 |
| alle Ereignisse im finalen Raumfenster | 11.327 |
| NND-behaltene Ereignisse im Raumfenster | 4.943 |
| NND-getriggerte Ereignisse im Raumfenster | 6.384 |

Der Zeitraum reicht vom 1. Januar 2015 bis 16. Mai 2026 beziehungsweise ueber etwa 4.153,7 Tage. Die Magnituden liegen zwischen 2,5 und 6,5. Die rotierten Koordinaten des verwendeten Fensters erstrecken sich ungefaehr ueber 1.001 km in x- und 1.372 km in y-Richtung. Die Zeitstempel sind sortiert, nicht dupliziert und die fuer Julia benoetigten Felder sind vollstaendig.

Eine nachtraegliche Magnitudenfilterung ergibt fuer das aktuelle Raumfenster:

| Magnitudenschwelle | alle Ereignisse | aktuell als NND-Background markiert |
|---:|---:|---:|
| 2,5 | 11.327 | 4.943 |
| 3,0 | 3.622 | 1.570 |
| 3,5 | 1.121 | 528 |
| 4,0 | 331 | 154 |

Die NND-Zahlen oberhalb 3,0 sind hier nur deskriptiv: Die vorhandenen Labels wurden mit `Mc=2.5` berechnet. Fuer einen methodisch sauberen Mc-3.0-Vergleich muss NND von Anfang an mit `Mc=3.0` neu ausgefuehrt werden.

### Empfohlenes Vergleichsdesign

Es sollen mindestens drei Fits auf exakt derselben raeumlichen und zeitlichen Auswahl gegenuebergestellt werden:

| Fit | Eingabe | Zielgroesse |
|---|---|---|
| Julia ETAS-SPDE | alle Ereignisse | gemeinsam geschaetzte raeumliche Hintergrundrate plus Triggering |
| PG all | Counts aller Ereignisse | gesamte beobachtete raeumliche Ereignisdichte inklusive Cluster |
| PG NND | Counts der durch NND behaltenen Ereignisse | zweistufige Approximation der Hintergrunddichte |

Optional kommt ein vierter Julia-Sensitivitaetsfit nur auf NND-behaltenen Ereignissen hinzu. Dieser ist kein gleichwertiger ETAS-Hauptfit, weil Triggering bereits vorab entfernt wurde. Er zeigt aber, wie stark der raeumliche Julia-Teil auf den vorherigen Declustering-Schritt reagiert.

Als erster Pilot wird empfohlen:

```text
Magnitude >= 3.0
Zeitraum 2015 bis zum gemeinsamen festen Enddatum
alle Ereignisse innerhalb eines vorab festgeschriebenen Italien-Fensters
Zeit in Tagen seit dem ersten Ereignis
Raum in isotrop skalierten rotierten Kilometerkoordinaten
```

`Mc=3.0` reduziert den Katalog auf 3.622 Ereignisse im aktuellen Fenster und entspricht zugleich eher der in der lokalen Sofiane-Dokumentation genannten Italien-Auswahl. Vor Verwendung muss noch geklaert werden, ob diese Auswahl wirklich dem fuer den wissenschaftlichen Vergleich vorgesehenen Katalog und Paperstand entspricht.

### Gemeinsamer Datenvertrag

Fuer Julia wird ein eigener, reproduzierbarer Adapter verwendet. Er ueberschreibt weder `Main_script.jl` noch die Originaldaten. Numerische Ausgabespalten:

```text
time_days  magnitude_excess  x_model  y_model
```

Diese vier Spalten entsprechen Sofianes numerischem Loader. `event_id`, Originalzeit, absolute Magnitude und Originalkoordinaten stehen zur Rueckverfolgbarkeit in einer separaten Mapping-Datei.

Konventionen:

- nach `datetime` aufsteigend sortieren;
- `time_days = (datetime - erster Zeitstempel)` in Tagen;
- fuer Sofianes Rechenweg `magnitude_excess = magnitude - Mc` schreiben und im Julia-`Catalog` `M0 = 0` setzen;
- `x_rot_km` und `y_rot_km` nur verschieben und isotrop skalieren, niemals x und y getrennt auf `[0,5]` pressen;
- vorgeschlagene einheitliche Skalierung: `SPACE_SCALE_KM = 100`, also eine Julia-Raumeinheit gleich 100 km;
- `x_model = (x_rot_km - x_min_km)/SPACE_SCALE_KM` und analog fuer y;
- Domain-Ecken aus den festgeschriebenen Grenzen erzeugen, nicht aus den zufaelligen Extrema eines einzelnen Samples;
- Auswahlparameter, Transformation, Einheit, Eventzahl und Checks in einer JSON-Metadatei speichern.

Die isotrope Skalierung erhaelt Entfernungsverhaeltnisse. Eine getrennte Normierung beider Achsen auf das synthetische Quadrat `[0,5] x [0,5]` wuerde Italien anisotrop verzerren und damit den raeumlichen ETAS-Kern veraendern.

### Pflichtpruefungen vor dem realen Fit

Vor einer Interpretation muessen folgende Punkte im Julia-Code geklaert oder korrigiert werden:

1. **Magnitudenkonvention:** Der Adapter liefert wie Sofianes Beispieldaten bereits `M - Mc`; deshalb wird `Catalog(..., M0=0)` verwendet. Am gemeinsamen Sampler wurde fuer diesen Lauf nichts geaendert. Vor einer wissenschaftlichen Interpretation muss anhand von Sofianes Modellgleichungen bestaetigt werden, dass diese Konvention beabsichtigt ist.
2. **Raumparameter:** `D`, `rho`, Meshflaechen und Intensitaet haengen von der gewaehlten Raumeinheit ab. Startwerte, Bounds und Priors aus dem synthetischen `[0,5]`-Beispiel duerfen nicht ungeprueft fuer Italien uebernommen werden.
3. **Bedeutung von `rho`:** In `spatialSPDE.jl` wird der uebergebene Wert direkt als `kappa` verwendet. Er ist daher nicht ohne Herleitung als gewohnte Korrelationsreichweite interpretierbar.
4. **Zeiteinheit:** Bei Zeit in Tagen muessen insbesondere `c`, der zeitliche Kernel und dessen Priors/Bounds als Tagesgroessen interpretiert und geprueft werden.
5. **Raumrand und Beobachtungsfenster:** Rechteck, Integrationsgewichte und Mesh muessen genau dieselbe feste Analyseflaeche beschreiben. Randereignisse vor Beginn des Zeitfensters fehlen als moegliche Eltern; deshalb ist eine zeitliche Vorlaufperiode beziehungsweise eine Sensitivitaetsanalyse zum Startdatum zu erwägen.
6. **Reproduzierbarkeit:** Der Sampler benoetigt einen expliziten RNG/Seed und der Lauf muss Seed, Git-Commit, Julia-/Paketversionen und alle Konfigurationswerte speichern.
7. **Burn-in:** Der aktuelle Export `BURNIN:end` ist inklusiv und behaelt bei `BURNIN=10` insgesamt 91 von 100 Werten. Die gewuenschte Semantik muss vor Produktionslaeufen festgelegt werden.
8. **Declustering-Output:** Aktuell wird nur die Anzahl der Hintergrundereignisse gespeichert. Fuer einen Ereignisvergleich mit NND sollten posteriorer Background-Anteil oder Background-Wahrscheinlichkeit je Ereignis exportiert werden.
9. **Flaecheneinheit der Intensitaet:** Wegen des Integrals `sum(w * Tmax * intensity)` ist die Julia-Intensitaet als Rate pro Modellflaeche und Modellzeit zu behandeln. Diese Einheit muss mit einem kleinen kontrollierten Test bestaetigt und fuer Vergleiche in Ereignisse pro km² und Jahr beziehungsweise erwartete Counts pro gemeinsamer Rasterzelle umgerechnet werden.

Diese Punkte sind keine kosmetischen Verbesserungen. Ohne sie koennen numerisch erfolgreiche Laeufe wissenschaftlich falsch skaliert oder zwischen den Methoden nicht vergleichbar sein.

### Mesh- und Laufstrategie

Der Elternschritt des aktuellen Julia-Codes speichert fuer Ereignis `j` Gewichte zu allen frueheren Ereignissen. Speicher- und Rechenaufwand wachsen daher ungefaehr quadratisch mit der Ereigniszahl. Ein unkontrollierter Start mit allen 11.327 Mc-2.5-Ereignissen ist nicht der richtige erste Schritt.

Empfohlene Stufen:

1. **Adaptertest:** 50 bis 200 fruehe Ereignisse, 2 bis 5 Iterationen; nur Format, Sortierung, Einheiten, Domain und Output testen.
2. **Kleiner Pilot:** etwa ein ruhigeres Jahr oder eine vorab definierte Teilperiode mit Mc 3,0; 10 bis 30 Iterationen und grobes Netz. Zweck ist Laufzeit- und Speicherprofiling, nicht Inferenz.
3. **Gesamter Mc-3.0-Pilot:** 3.622 Ereignisse im festen Fenster; mehrere kurze Seeds und ein grobes bis mittleres Netz. Erst danach Laufzeit fuer Produktionsketten hochrechnen.
4. **Produktionsfit:** mehrere unabhaengige Ketten, ausreichendes Burn-in, gespeicherte Diagnostik und vorab festgelegte Abbruch-/Konvergenzkriterien.
5. **Sensitivitaet:** mindestens Magnitudenschwelle, Meshaufloesung, Zeitfenster, SPDE-Prior und Startwerte variieren. Mc 2,5 erst angehen, wenn Mc 3,0 technisch und statistisch stabil ist.

Bei `SPACE_SCALE_KM = 100` entsprechen ungefaehr:

| typische Zielskala | maximale Dreiecksflaeche in Modellkoordinaten | Rolle |
|---:|---:|---|
| 100 km | etwa 0,5 | grober Laufzeitpilot |
| 50 km | etwa 0,125 | mittlere Sensitivitaet |
| 20 km | etwa 0,02 | ungefaehr mit dem 20-km-PG-Raster vergleichbar, wahrscheinlich teuer |

Die Werte sind Startpunkte, keine fachlich validierten Produktionswerte. Die tatsaechliche Zahl der Meshknoten und die Laufzeit muessen fuer die rechteckige Italien-Domain protokolliert werden.

### Vergleich der Ergebnisse

Julia liefert die Hintergrundintensitaet an unregelmaessigen Meshknoten, PG liefert erwartete Counts auf regulaeren Rasterzellen. Direkte Farbskalenvergleiche waeren irrefuehrend. Beide Ergebnisse muessen zuerst auf eine gemeinsame Zielgroesse gebracht werden, vorzugsweise erwartete Hintergrundereignisse je 20-km-Zelle ueber denselben Beobachtungszeitraum.

Vorgesehener Vergleich:

1. Julia-Intensitaet innerhalb der Dreiecke interpolieren und ueber jede gemeinsame PG-Zelle integrieren.
2. Raumeinheit und Zeitspanne explizit in erwartete Counts umrechnen.
3. Fuer alle Karten dieselbe Maske, Ausdehnung und Farbskala verwenden.
4. Posterior-Mittel beziehungsweise Median und Unsicherheit getrennt darstellen.
5. Neben Karten auch numerische Masse berichten:
   - Gesamtzahl geschaetzter Hintergrundereignisse und Background-Anteil;
   - Korrelation beziehungsweise Rangkorrelation der gemeinsamen Zellraten;
   - Differenz- und Quotientenkarten;
   - Hotspot-Ueberlappung fuer vorab festgelegte Quantile;
   - raeumlich blockierte Vorhersage- oder Log-Score-Pruefung, falls ein sauberes Train/Test-Design implementiert wird;
   - Stabilitaet dieser Groessen ueber Ketten und Sensitivitaetslaeufe.

Ein besonders informativer Vergleich ist:

```text
PG all minus Julia ETAS-Background
```

Diese Differenz sollte vor allem dort gross sein, wo zeitlich konzentrierte Sequenzen liegen. Zusaetzlich zeigt

```text
PG NND-Background minus Julia ETAS-Background
```

die Unterschiede zwischen vorgelagertem deterministischem NND-Declustering und gemeinsamer probabilistischer ETAS-Zuordnung.

### Wissenschaftlich moegliche Erkenntnisse

Bei stabilen und diagnostisch unauffaelligen Laeufen koennen wir vorsichtig untersuchen:

- welche Regionen eine robuste langfristige Hintergrundseismizitaet zeigen;
- welche scheinbaren Hotspots hauptsaechlich durch einzelne Nachbebensequenzen entstehen;
- ob NND mehr oder weniger Ereignisse als Hintergrund klassifiziert als das ETAS-Modell und wo diese Unterschiede liegen;
- wie stark raeumliche Hintergrundkarten von Declustering, Glattung und Meshaufloesung abhaengen;
- welche zeitlichen, produktivitaetsbezogenen und raeumlichen Triggerparameter mit dem Katalog vereinbar sind;
- ob ein gemeinsames zeitlich-raeumliches Modell gegenueber der zweistufigen Pipeline konsistentere Unsicherheiten oder bessere out-of-sample Vorhersagen liefert.

Nicht ohne weitere Evidenz ableitbar sind tektonische Kausalitaet, die "wahre" Background-Zuordnung einzelner Ereignisse, generelle Ueberlegenheit einer Methode oder belastbare Gefahrenprognosen. Auch optisch aehnliche Karten reichen nicht als Modellvalidierung.

### Konkrete Arbeitspakete

- **WP1 – Analysevertrag festschreiben:** Katalogquelle, Mc, Zeitraum, festes Raumfenster, Zeiteinheit und gemeinsame Zielgroesse mit Sofiane und dem Team bestaetigen.
- **WP2 – Julia-Korrektheitsaudit:** Magnitudeninkonsistenz, Parameter-Einheiten, SPDE-Parametrisierung, Randbehandlung und RNG klaeren; kleine synthetische Regressionstests hinzufuegen.
- **WP3 – Italien-Adapter:** Eventdatei und Mapping/Metadaten reproduzierbar aus dem gemeinsamen Preprocessing erzeugen, ohne Originaldateien zu ueberschreiben.
- **WP4 – Technischer Pilot:** kleine Teilmenge und anschliessend Mc-3.0-Gesamtkatalog mit Laufzeit-/Speicherprotokoll testen.
- **WP5 – Produktionsketten:** mehrere Seeds/Ketten, laengerer Burn-in, Thin nur bei begruendetem Speicherbedarf und vollstaendige Diagnostik.
- **WP6 – Gemeinsame Auswertung:** Julia-Mesh auf PG-Zellen integrieren, Karten und numerische Vergleichsmasse mit identischen Einheiten erzeugen.
- **WP7 – Sensitivitaet und Bericht:** Ergebnisse ueber Mc, Mesh, Priors und Zeitfenster vergleichen; bestaetigte Befunde, Unsicherheit und offene Modellunterschiede getrennt dokumentieren.

### Entscheidung vor der Implementierung

Empfohlener naechster konkreter Schritt ist WP1 plus WP2, danach ein eigener Italien-Adapter und ein separates Julia-Laufskript. `Main_script.jl` sollte als weiterhin lauffaehiges synthetisches Referenzbeispiel erhalten bleiben. Der reale Lauf sollte eigene Konfigurationen und eigene Ergebnisordner verwenden, damit synthetische und reale Ergebnisse nicht gegenseitig ueberschrieben werden.

### Bevorzugter Ablauf: Sofianes Main-Skript fuer Italien

Der neue Ablauf bleibt bewusst nahe an Sofianes Original. `Main_script_italy.jl` ist eine eigenstaendige Kopie des Einstiegsablaufs. Die gemeinsamen Dateien unter `src/` und das synthetische `Main_script.jl` bleiben dabei unveraendert. Angepasst sind Datenpfad, rechteckige Italien-Domain, Seed, Testlaenge und eigener Ausgabeordner.

`scripts/prepare_italy_mc3_comparison.py` liest den Rohkatalog aus dem uebergeordneten `data/`-Ordner, wendet den festgeschriebenen Analysevertrag an und berechnet NND eigens fuer `Mc=3.0` neu. Danach erzeugt `scripts/prepare_italy_catalog.py` aus diesem Bestand Sofianes headerloses numerisches Vier-Spalten-Format:

```text
Tage seit dem ersten Ereignis | Magnitude - Mc | skaliertes x | skaliertes y
```

Erzeugte Eingaben und Begleitdateien:

| Datei | Inhalt |
|---|---|
| `data/italy_mc3_sofiane.txt` | kompletter Mc-3.0-Katalog mit 3.622 Zeilen |
| `data/italy_mc3_sofiane_first_200.txt` | erste 200 Ereignisse fuer technische Tests |
| `data/italy_mc3_sofiane_first_500.txt` | erste 500 Ereignisse fuer Laufzeittests |
| `data/italy_mc3_sofiane_first_1000.txt` | erste 1.000 Ereignisse fuer Laufzeittests |
| `data/italy_mc3_sofiane_mapping.csv` | Rueckbezug zu Event-ID, Zeit, Magnitude und Originalkoordinaten |
| `data/italy_mc3_sofiane_meta.json` | Auswahl, Transformationen, Einheiten und Eventzahlen |

Start aus `SPDE-ETAS-main`:

```bash
/Users/toni/Documents/CodeProjects/VirtualEnvs/PolyaGamma/bin/python scripts/prepare_italy_mc3_comparison.py
python3 scripts/prepare_italy_catalog.py
julia --project=. Main_script_italy.jl
```

Der validierte Smoke-Test verwendet standardmaessig 200 Ereignisse, zwei MCMC-Iterationen, Seed `20260908`, eine Raumeinheit von 100 km und eine maximale Dreiecksflaeche von 0,5. Er ergab eine Zeitspanne von 369,81 Tagen sowie ein Mesh mit 240 Knoten und 423 Dreiecken. Beide MCMC-Schritte wurden erfolgreich beendet; die Ergebnisse liegen getrennt unter `mcmc_results/italy/`.

Fuer den kompletten Katalog muss in `Main_script_italy.jl` nur `DATA_FILE` auf `data/italy_mc3_sofiane.txt` umgestellt werden. `NITER`, Burn-in, Priors, Startwerte und Mesh sollten jedoch vor diesem Lauf fachlich und hinsichtlich Laufzeit/Speicher festgelegt werden. Ein blosses Hochsetzen der Iterationszahl waere noch kein belastbarer Produktionsfit.

Alternativ akzeptiert das Skript Datendatei, Iterationszahl und Ausgabeordner als Argumente. Damit koennen Pilotlaeufe ohne Codeaenderung getrennt gestartet werden:

```bash
julia --project=. Main_script_italy.jl data/italy_mc3_sofiane_first_500.txt 2 mcmc_results/italy_pilot_500
julia --project=. Main_script_italy.jl data/italy_mc3_sofiane_first_1000.txt 2 mcmc_results/italy_pilot_1000
```

Beide Skalierungstests wurden am 15. September 2026 erfolgreich abgeschlossen:

| Ereignisse | Zeitspanne | Iterationen | reale Prozesszeit | max. Resident-Memory | Peak-Memory-Footprint |
|---:|---:|---:|---:|---:|---:|
| 500 | 610,41 Tage | 2 | 10,67 s | 781 MB | 561 MB |
| 1.000 | 668,61 Tage | 2 | 10,68 s | 775 MB | 567 MB |

Die Messwerte stammen aus separaten Prozessen mit `/usr/bin/time -l`. Dass sich die Zeiten und Speicherwerte kaum unterscheiden, zeigt, dass bei nur zwei Iterationen Julia-Kompilierung, Paketinitialisierung und die feste Mesh-Arbeit dominieren. Diese Messung belegt technische Stabilitaet bis 1.000 Ereignisse, erlaubt aber noch keine belastbare Hochrechnung fuer lange Ketten oder alle 3.622 Ereignisse. Fuer ein echtes Laufzeitprofil sollten mehr Iterationen in einer Julia-Sitzung und getrennte Messungen von Initialisierung und MCMC verwendet werden.

### Festgelegter Mc-3.0-Vergleichsbestand und NND-Neuberechnung

Der technische Analysevertrag ist nun explizit in `data/italy_mc3_comparison/metadata.json` gespeichert:

```text
Magnitude: M >= 3.0
Zeitraum: decimal year 2015.0 bis 2026.5
Projektion: gamma = 41.75252483646699 Grad
Rotation: -45 Grad
festes Raumfenster: x = 265.707... bis 1269.757... km
                     y = -1414.197... bis -34.860... km
gemeinsames Raster: 51 x 69 Zellen, nominal 20 km
```

NND wurde mit `D=1.6`, `b=1.0`, `Mc=3.0`, `log10(eta)=-4.6` und Seed 0 neu ausgefuehrt. Damit werden nicht die methodisch unpassenden Mc-2.5-Labels wiederverwendet:

| Auswahl | Ereignisse |
|---|---:|
| nach Magnituden- und Zeitfilter | 3.696 |
| innerhalb des festen Raumfensters | 3.622 |
| NND-kept innerhalb des Fensters | 1.694 |
| NND-triggered innerhalb des Fensters | 1.928 |

`data/italy_mc3_comparison/count_grids.npz` enthaelt auf demselben Raster bereits getrennte beobachtete Counts fuer alle 3.622 Ereignisse und fuer die 1.694 NND-kept-Ereignisse. Die letzte Zeile und Spalte des nominellen 20-km-Rasters ragen ueber das Analysefenster hinaus; bei Ratenvergleichen werden diese Zellen deshalb auf die tatsaechliche Schnittflaeche beschnitten.

### TODO fuer die beiden PG-Laeufe

Vor den vergleichenden PG-Laeufen muss in `experiments/exp_italy_preprocess_nnd_rot_cut_pg.py` folgende Funktion korrigiert werden:

```python
def get_cell_size_km_from_filename(prefix: Path) -> float:
    # ... liest cell_size_km korrekt aus dem Dateinamen ...
    return 2  # cell_size_km
```

Bei einem Dateinamen mit `_dkm_20` wird zwar `cell_size_km=20.0` geparst, anschliessend aber konstant `2` zurueckgegeben. Mit `RHO_KM=4.0` berechnet der Code dadurch `rho_cells=4/2=2` statt `4/20=0.2`. Falls `rho` tatsaechlich in Rasterzellen erwartet wird, entspricht die verwendete raeumliche Skala damit effektiv etwa 40 km statt 4 km, also Faktor 10 zu gross. Das veraendert die Prior-Praezisionsmatrix und damit die Glattung und Posteriorraten. Vor einer Ergebnisinterpretation muss `return cell_size_km` verwendet und die Bedeutung von `rho` in `precision_matern` bestaetigt werden.

Danach sind zwei getrennte PG-Laeufe mit `data/italy_mc3_comparison/count_grids.npz` vorgesehen:

1. `counts_all`: alle 3.622 Ereignisse;
2. `counts_nnd_kept`: 1.694 mit Mc=3.0 neu bestimmte NND-Background-Ereignisse.

Das bestehende PG-Skript zeigt die Ergebnisse derzeit nur als Plots und speichert Posteriorfelder nicht dauerhaft. Fuer den Vergleich muessen Mean-, SD- und moeglichst die Raster-Samples zusammen mit Seed und Konfiguration geschrieben werden. Diese PG-Aenderungen und Laeufe sind noch nicht umgesetzt.

### Technischer Vollkataloglauf und gemeinsames Vergleichsraster

Der komplette Julia-Katalog wurde technisch mit zwei Iterationen ausgefuehrt:

```bash
julia --project=. Main_script_italy.jl data/italy_mc3_sofiane.txt 2 mcmc_results/italy_full_technical
```

| Groesse | Ergebnis |
|---|---:|
| Ereignisse | 3.622 |
| Zeitspanne | 4.142,14 Tage |
| Mesh | 240 Knoten, 423 Dreiecke |
| reale Prozesszeit | 12,22 s |
| maximale Resident-Memory | 1,06 GB |
| Peak-Memory-Footprint | 820 MB |
| technische Background-Counts | 1.337 und 1.262 |

Die Background-Counts und Felder sind mit zwei Iterationen nicht wissenschaftlich interpretierbar.

`scripts/project_julia_to_pg_grid.py` integriert jedes stueckweise lineare Julia-Intensitaetsfeld exakt ueber die Schnittflaechen der SPDE-Dreiecke mit den gemeinsamen 20-km-Zellen. Anschliessend wird mit der Beobachtungsdauer multipliziert. Zielgroesse sind damit erwartete Julia-Background-Ereignisse je gemeinsamer Rasterzelle und Zeitraum. Ausgaben:

| Datei | Inhalt |
|---|---|
| `mcmc_results/italy_full_technical/julia_expected_counts_on_pg_grid.npz` | Julia-Rastersamples, Mittelwert/SD und beobachtete All-/NND-Counts |
| `mcmc_results/italy_full_technical/julia_pg_grid_summary.csv` | Zellgrenzen, beschnittene Flaeche und gemeinsame Vergleichswerte |

Die technischen Julia-Rastersummen betragen 1.450,53 und 1.340,66 erwartete Background-Ereignisse. Dass sie nicht exakt den gezogenen Background-Counts 1.337 und 1.262 entsprechen, muss vor Produktionsauswertungen untersucht werden; moegliche Ursachen sind die bedingte MCMC-Zuordnung gegenueber dem integrierten Intensitaetsfeld sowie Modell-/Integrationsdetails. Die neue Rasterintegration selbst verwendet keine Mittelpunktnaeherung.

Wichtig: Die im Smoke-Test ausgegebenen Background-Counts 159 und 139 sowie alle Parameter- und Intensitaetswerte sind **keine wissenschaftlichen Ergebnisse**. Zwei Iterationen, Sofianes synthetische Startwerte/Priors und eine chronologisch abgeschnittene Teilmenge pruefen nur den Daten- und Rechenweg.

### Frueherer Prototyp: `experiments/exp_italy.jl`

Am 10. September 2026 wurde ein separates technisches Julia-Experiment angelegt:

```text
experiments/exp_italy.jl
```

Start aus `SPDE-ETAS-main`:

```bash
julia --project=. experiments/exp_italy.jl
```

Das Skript veraendert `Main_script.jl` nicht und verwendet einen eigenen Ausgabeordner:

```text
mcmc_results/italy_smoke/
```

Der aktuelle Smoke-Test ist absichtlich klein konfiguriert:

```text
Mc = 3.0
erste 200 geeignete Ereignisse in chronologischer Reihenfolge
2 MCMC-Iterationen
Seed = 20260908
1 Modell-Raumeinheit = 100 km
maximale Dreiecksflaeche = 0.5 Modellflaecheneinheiten
```

Der Loader liest direkt den vorhandenen Pipe-Katalog
`../data/preprocess_nnd_rot_cut_bin_Mc_2.5_eta_-4.60_dkm_20_events.csv`. Er prueft benoetigte Spalten und Feldzahlen, verwendet nur Ereignisse mit `inside_final_cut=True` und `Magnitude >= 3.0`, sortiert nach Zeit und wandelt die Zeit in Tage seit dem ersten ausgewaehlten Ereignis um. Die rotierten Kilometerkoordinaten werden mit derselben Skala in beiden Achsen verschoben und durch 100 km geteilt.

Der validierte Lauf ergab:

```text
geeignete Mc-3.0-Ereignisse im Fenster: 3622
fuer den Smoke-Test verwendet: 200
Zeitraum der Teilmenge: 2015-01-01 bis 2016-01-06
Zeitspanne: 369.81 Tage
Domain: 10.041 x 13.793 Modell-Raumeinheiten
Mesh: 240 Knoten, 423 Dreiecke
MCMC: beide Schritte erfolgreich abgeschlossen
```

Erzeugte Dateien:

| Datei | Inhalt |
|---|---|
| `run_metadata.txt` | Auswahl, Einheiten, Seed, Netz- und Laufparameter |
| `selected_catalog.tsv` | Event-ID, relative Zeit, Magnitude und Modellkoordinaten fuer 200 Ereignisse |
| `mesh_points.tsv` | IDs und Koordinaten der 240 Meshknoten |
| `parameter_chain.csv` | zwei technische Parametersaetze |
| `intensity_chain.csv` | zwei Intensitaetsfelder mit je 240 Knotenwerten |
| `background_count_chain.txt` | technische Background-Counts je Iteration |

Mit demselben Seed wurden in zwei erfolgreichen Testlaeufen dieselben beiden Iterationswerte beobachtet. Der technische Daten- und Rechenpfad ist damit fuer diese Teilmenge reproduzierbar.

Wichtig: Die Background-Counts 159 und 139 sowie alle Parameter- und Intensitaetswerte dieses Laufs sind **keine wissenschaftlichen Ergebnisse**. Zwei Iterationen, synthetische Startwerte/Priors und die chronologisch erste Teilmenge dienen ausschliesslich als Integrations- und Laufzeittest.

Der Prototyp und dessen Ausgaben unter `mcmc_results/italy_smoke/` sind noch unversioniert vorhanden. Sie werden im bevorzugten Ablauf nicht mehr benoetigt, wurden aber nicht ohne ausdrueckliche Freigabe geloescht. Der gemeinsame Sampler steht wieder auf dem bisherigen Projektstand.

Der naechste Schritt ist eine wissenschaftlich konfigurierte Produktionskette, nicht die Interpretation des Zwei-Iterations-Laufs. Zuerst sollten wir:

1. die Magnituden-, Zeit- und Raumparametrisierung mit Sofiane bestaetigen;
2. Iterationszahl, Burn-in, Priors, Startwerte und mehrere Seeds festlegen;
3. die oben notierten PG-Korrekturen und beide Mc-3.0-PG-Laeufe umsetzen;
4. zusaetzlich Event-weise Background-Wahrscheinlichkeiten und ausreichende Diagnostik exportieren;
5. erst danach mehrere laengere Mc-3.0-Ketten planen.

## Aktueller Status

- **Erledigt:** Julia-1.12-/Optim-2.x-Kompatibilitaet fuer den beobachteten Lauf hergestellt.
- **Erledigt:** Nichtinteraktive, reproduzierbare Netzerzeugung mit 45 Knoten und 71 Dreiecken.
- **Erledigt:** Vollstaendiger technischer Testlauf ueber 100 MCMC-Schritte.
- **Erledigt:** Reproduzierbarer Adapter in Sofianes numerisches Vier-Spalten-Format samt Mapping und Metadaten.
- **Erledigt:** Separate Kopie `Main_script_italy.jl` mit rechteckigem Mesh, Seed und eigenen Ausgaben; Smoke-Test mit 200 Ereignissen erfolgreich.
- **Erledigt:** Technische Zwei-Iterations-Piloten mit 500 und 1.000 Ereignissen erfolgreich; getrennte Ausgaben und erste Ressourcenmessung dokumentiert.
- **Erledigt:** Gemeinsamen Mc-3.0-Analysevertrag festgeschrieben und NND fuer Mc=3.0 neu berechnet.
- **Erledigt:** Technischer Julia-Vollkataloglauf mit allen 3.622 Ereignissen erfolgreich.
- **Erledigt:** Julia-SPDE-Intensitaeten exakt auf das gemeinsame beschnittene 20-km-Raster integriert.
- **Offen:** PG-Zellgroessenfehler korrigieren, `rho` fachlich bestaetigen und PG-all/PG-NND mit gespeicherten Posteriorfeldern ausfuehren.
- **Offen:** Magnitudenkonvention und Modellgleichungen mit Sofiane fachlich bestaetigen; der gemeinsame Sampler wurde dafuer nicht veraendert.
- **Offen:** Plotting und systematische MCMC-Diagnostik.
- **Offen:** Fachliche Freigabe der Netzaufloesung und Produktionsparameter.

Hinweis: `mcmc_results/` und die aktualisierte Datei `mesh_points_case01_patches.txt` stammen aus dem Validierungslauf. Andere bereits im uebergeordneten Repository vorhandene Aenderungen wurden bei dieser Arbeit nicht bearbeitet.
