# F2_extreme misclassification diagnostic

Analysis window: 2026-05-14 to 2026-05-21 (UTC); source: /var/lib/codar-sounder/ac0g-bee1-rx888/SEAB

Total records: 169,367

## mode_layer distribution

| mode_layer | n | %% |
|---|---|---|
| F2 | 66,367 | 39.2 |
| F2_extreme | 62,336 | 36.8 |
| F1 | 26,044 | 15.4 |
| E | 14,620 | 8.6 |

## Multi-hop reinterpretation of F2_extreme records

For each F2_extreme record (n = 62336), compute ``h'_apparent = sqrt(P² - D²) / (2N)`` for N = 1, 2, 3, 4.  Count how many records land in:

  - **normal F2 band** (h' = 150–450 km)
  - **F1 / borderline** (h' = 450–600 km)
  - **F2_extreme** (h' > 600 km)
  - **below normal F** (h' < 150 km)

| N | normal F2 | F1/borderline | F2_extreme | below F | mean h' | median h' |
|---|---|---|---|---|---|---|
| 1 | 0 (0.0%) | 10,431 (16.7%) | 51,905 (83.3%) | 0 (0.0%) | 780 km | 789 km |
| 2 | 45,163 (72.5%) | 17,173 (27.5%) | 0 (0.0%) | 0 (0.0%) | 390 km | 394 km |
| 3 | 62,336 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 260 km | 263 km |
| 4 | 51,905 (83.3%) | 0 (0.0%) | 0 (0.0%) | 10,431 (16.7%) | 195 km | 197 km |

## F2_extreme group_range histogram (100 km bins)

| P range (km) | count | %% |
|---|---|---|
| 1700–1799 | 5,301 | 8.5 |
| 1800–1899 | 8,740 | 14.0 |
| 1900–1999 | 8,168 | 13.1 |
| 2000–2099 | 7,375 | 11.8 |
| 2100–2199 | 8,168 | 13.1 |
| 2200–2299 | 7,911 | 12.7 |
| 2300–2399 | 8,345 | 13.4 |
| 2400–2499 | 8,328 | 13.4 |

## Climatological fit

- 45,163 / 62,336 (72.5%) F2_extreme records have a clean **2-hop** reinterpretation in the normal F2 band (150–450 km).

- 62,336 / 62,336 (100.0%) have a clean **3-hop** reinterpretation in the normal F2 band.

- 62,336 / 62,336 (100.0%) have **either** 2-hop OR 3-hop reinterpretation in normal F2.


**Verdict: multi-hop misclassification hypothesis is strongly supported.** A majority of records flagged as F2_extreme have a climatologically plausible multi-hop interpretation at normal F2 heights — far more likely than persistent F2_extreme conditions across 35% of CPIs.