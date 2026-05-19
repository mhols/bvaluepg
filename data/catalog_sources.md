# Earthquake Catalog Sources

## Italy / INGV

File: `earthquakes_2point5_ingv_italy_2015-2026.json`

Raw download: `italy_ingv_m2point5_2015-2026.txt`

Source query:

```text
https://webservices.ingv.it/fdsnws/event/1/query?format=text&starttime=2015-01-01&endtime=2026-05-19&minmagnitude=2.5&minlatitude=35&maxlatitude=49&minlongitude=5&maxlongitude=20
```

Selection:

- Time range: 2015-01-01 to 2026-05-19
- Bounding box: lat 35 to 49, lon 5 to 20
- Magnitude threshold: M >= 2.5
- Events in converted GeoJSON: 11,435

Completeness note: this is a modern INGV/ISIDe subset. Literature notes that ISIDe has significant incompleteness before 2005, while the post-2005 catalog is much more homogeneous. Starting in 2015 and using M >= 2.5 is a conservative working choice close to the requested 10,000 events.

Sources:

- INGV FDSN data center: https://www.fdsn.org/datacenters/detail/INGV/
- ISIDe completeness paper: https://link.springer.com/article/10.1007/s11600-024-01395-3

## France / BCSF-Renass

File: `earthquakes_mw2point5_bcsf_renass_france_1962-2021.json`

Raw download: `france_bcsf_renass_1962-2021.json.zip`

Source dataset:

```text
https://renass.unistra.fr/products/instrumental-seismicity-in-mainland-france/instrumental-seismicity-in-mainland-france_dataset_1962-2021.json.zip
```

Selection:

- Time range: 1962-2021
- Region: mainland France dataset
- Magnitude threshold: Mw >= 2.5
- Events in converted GeoJSON: 3,070

Completeness note: this is intended as a weak-seismicity comparison catalog. The Mw >= 2.5 threshold is a conservative filter, but local completeness should still be checked by period and subregion before formal inference.

Sources:

- Dataset page: https://renass.unistra.fr/fr/produits/sismicite-instrumentale-de-la-france-hexagonale/
- DOI: https://doi.org/10.25577/fv3f-sq09
- Method note: https://renass.unistra.fr/products/instrumental-seismicity-in-mainland-france/instrumental-seismicity-in-mainland-france_note.pdf


## California

The SCEDC hosts files that document the curation status of individual days in the earthquake catalog. Historically, SCSN analysts attempted to review every event in the catalog. However, with the growth of both the seismic network and the catalog itself, complete manual review has become increasingly difficult with limited staff resources. This is particularly true for the 2019 Ridgecrest sequence, which remains largely unreviewed below M2.5. The corresponding catalog status files are updated daily.

Source: Southern California Earthquake Data Center (SCEDC), "Earthquake Catalogs", section "Catalog Processing Status":

```text
https://scedc.caltech.edu/data/eq-catalogs.html
```
