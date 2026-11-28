
# 5.1.1 – Logistic Regression implementáció és optimalizálás

## 1. Feladat adatai

| Megnevezés | Érték |
|---|---|
| Felelős | Anna |
| Feature branch | `feature/05-01-01-logistic-regression` |
| Kapcsolódó kutatás | 4.1.1 – Logistic Regression Research |
| Notebook | `notebooks/02_logistic_regression.ipynb` |
| Modellmodul | `src/models/ml/logistic_regression.py` |
| Unit tesztek | `tests/test_logistic_regression.py` |

## 2. Feladat célja

A multinomiális Logistic Regression modell implementációja, baseline konfigurációjának értékelése, valamint a regularizáció és az alternatív adatreprezentációk hatásának vizsgálata a Fashion-MNIST adathalmazon.

A részletes implementáció, mérési eredmények, vizualizációk és szakmai értékelések a kapcsolódó notebookban találhatók.

## 3. Megvalósítási lépések

A cellaszámok a notebook első cellájától számított, 1-től kezdődő sorszámok.

| Lépés | Megvalósítás és cél | Notebook cellák |
|---|---|---|
| 1. Környezet konfigurálása | Függőségek, hardverinformációk, random seed és projekt színpaletta rögzítése a reprodukálhatóság érdekében. | 1–5 |
| 2. Adatbetöltés | Közös adatbetöltő és stratifikált train/validation split alkalmazása az egységes modellösszehasonlításhoz. | 6–7 |
| 3. Kísérleti infrastruktúra | Újrafelhasználható tanítási, értékelési, időmérési és eredménymentési függvények kialakítása. | 8–14 |
| 4. Baseline modell | LR-00 tanítása és részletes értékelése referenciaként a további kísérletekhez. | 15–32 |
| 5. Regularizáció | A C hiperparaméter hatásának vizsgálata a prediktív teljesítményre és konvergenciára. | 33–43 |
| 6. Standardizálás | A StandardScaler hatásának vizsgálata, szükség esetén konvergenciakontroll. | 44–48 |
| 7. PCA | Az EDA alapján kiválasztott 84 és 187 komponenses dimenziócsökkentés vizsgálata. | 49–52 |
| 8. HOG | Lokális gradiensinformációkon alapuló jellemzőreprezentáció értékelése. | 53–56 |
| 9. Kombinált optimalizálás | A legjobb adatreprezentáció kombinálása különböző regularizációs konfigurációkkal. | 57–61 |
| 10. Modellválasztás | Validációs rangsorolás, a kiválasztott modell és preprocessor visszatöltése, metrikák ellenőrzése. | 62–66 |
| 11. Optimalizált modell értékelése | Confusion matrix, osztályonkénti metrikák, HOG súlyvizualizáció és EDA hibaanalízis. | 67–76 |
| 12. Összehasonlítás | A baseline és optimalizált modell teljesítményének és számítási költségeinek összevetése. | 77–81 |
| 13. Eredmények exportálása | Végleges metrikák, osztályonkénti eredmények és következtetések. | 82–85 |

## 4. Eltérések az eredeti kísérleti tervtől

- Az eredetileg tervezett 100 és 200 PCA-komponens helyett az EDA eredményei alapján 84 és 187 komponenst alkalmaztunk Beni javaslatára az EDA kapcsán.
- Az LR-03 és LR-04 konfigurációk nem konvergáltak az 1000 iterációs kereten belül, ezért LR-03B és LR-04B azonosítóval, 3000-es iterációs limittel kontrollfuttatásokat végeztünk.
- A HOG reprezentáció biztosította a legjobb egyedi validációs eredményt, ezért ezt kombináltuk a további C értékekkel.

Összesen 13 dokumentált kísérlet készült.

## 5. Főbb eredmények

| Metrika | Baseline (LR-00) | Optimized (LR-07) |
|---|---:|---:|
| Adatreprezentáció | Pixelvektor | HOG |
| C | 1.0 | 1.0 |
| Validation accuracy | 86,15% | 88,67% |
| Validation Macro F1 | 86,10% | 88,66% |
| Validation log loss | 0,4062 | 0,3096 |
| Tanítási idő | 68,07 s | 38,23 s |
| Modellméret | 0,0608 MiB | 0,0998 MiB |

A HOG reprezentáció 2,52 százalékpontos validációs pontosságjavulást eredményezett, magasabb előfeldolgozási költség mellett.

A részletes kísérleti eredmények és következtetések a notebook 5–10. fejezeteiben találhatók.

## 6. Eredményfájlok

| Fájl / mappa | Tartalom |
|---|---|
| `results/experiments/logistic_regression_experiments.csv` | Összes kísérlet konfigurációja és metrikái |
| `results/experiments/logistic_regression_final_metrics.csv` | Baseline és optimalizált modell eredményei |
| `results/experiments/logistic_regression_classification_report.csv` | Osztályonkénti validációs metrikák |
| `results/figures/logistic_regression/` | Eredményábrák |
| `artifacts/logistic_regression/` | Betanított modellek, preprocesszorok és konfigurációk |

A modellállományok lokálisan tárolódnak, nem kerülnek Git-verziókezelésbe.

## 7. Validáció és reprodukálhatóság

- Közös adatfelosztás: 50 000 training, 10 000 validation és 10 000 official test minta.
- Közös random seed: 42.
- A preprocessing illesztése kizárólag a training halmazon történik.
- A modellválasztás kizárólag validációs metrikák alapján történt.
- A kiválasztott modell visszatöltése sikeres volt, a mentett validációs metrikák reprodukálhatók.
- A hivatalos teszthalmaz értékelése a projekt végső modellösszehasonlítási szakaszában történik.

A futásidők lokális CPU-környezetben mért értékek. Az inferenciaidő nem tartalmazza az előfeldolgozás költségét.
