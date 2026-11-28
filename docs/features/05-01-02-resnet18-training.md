
# 5.1.2 – ResNet-18 implementáció és optimalizálás

## 1. Feladat adatai

| Megnevezés | Érték |
|---|---|
| Felelős | Anna |
| Feature branch | `feature/05-01-02-resnet18` |
| Kapcsolódó kutatás | 4.1.2 – ResNet-18 modellkutatás és kísérlettervezés |
| Notebook | `notebooks/04_resnet18.ipynb` |
| Modellmodul | `src/models/dl/resnet18.py` |
| Training modul | `src/training/dl.py` |
| Unit tesztek | `tests/test_resnet18.py` |

## 2. Feladat célja

A ResNet-18 architektúra adaptálása a Fashion-MNIST adathalmazhoz, baseline tanítása és optimalizálása, valamint a mélyebb reziduális architektúra prediktív teljesítményének és általánosítási képességének vizsgálata.

A részletes implementáció, mérési eredmények, vizualizációk és szakmai értékelések a kapcsolódó notebookban találhatók.

## 3. Megvalósítási lépések

A cellaszámok a notebook első cellájától számított, 1-től kezdődő sorszámok.

| Lépés | Megvalósítás és cél | Notebook cellák |
|---|---|---|
| 1. Környezet konfigurálása | GitHub–Colab integráció, GPU, függőségek, random seed és futtatási környezet rögzítése. | 1–12 |
| 2. Adatbetöltés | Közös adatfelosztás, DL preprocessing és baseline konfiguráció alkalmazása. | 13–17 |
| 3. Modellarchitektúra | Adaptált ResNet-18 inicializálása, paraméterszám, rétegdimenziók és forward pass ellenőrzése. | 18–21 |
| 4. Kísérleti infrastruktúra | Közös training framework integrálása, újrafelhasználható experiment runner, CSV- és checkpointmentés. | 22–32 |
| 5. Baseline modell | RN-00 tanítása és értékelése referenciaként. | 33–37 |
| 6. Dropout | Az osztályozófej előtti dropout hatásának vizsgálata. | 38–41 |
| 7. Learning rate | Alacsonyabb learning rate hatásának vizsgálata. | 42–45 |
| 8. Standardizálás | Training-only normalizálási statisztikák alkalmazása. | 46–49 |
| 9. Augmentáció | RandomAffine transzformáció hatásának vizsgálata. | 50–53 |
| 10. Weight decay | Súlyregularizáció hatásának vizsgálata. | 54–57 |
| 11. Scheduler | A közös training modul kiegészítése és CosineAnnealingLR alkalmazása. | 58–64 |
| 12. Kombinált optimalizálás | Korábbi eredmények visszaállítása, a legígéretesebb technikák kombinálása és értékelése. | 65–73 |
| 13. Modellválasztás | Validációs rangsorolás, checkpointok visszatöltése és metrikák reprodukálása. | 74–81 |
| 14. Learning curves | Konvergencia és túlilleszkedés vizsgálata. | 82–84 |
| 15. Osztályonkénti értékelés | Confusion matrixok és class-level F1-score összehasonlítása. | 85–90 |
| 16. Kísérletek összehasonlítása | Baseline-referenciás pontosságjavulás és számítási költségek értékelése. | 91–94 |
| 17. Feature mapek | A négy reziduális szakasz aktivációinak kvalitatív vizualizációja. | 95–99 |
| 18. Eredmények exportálása | Végleges metrikák, classification report és fájlellenőrzés. | 100–103 |
| 19. Lezárás | Következtetések és végleges ZIP-backup. | 104–107 |

## 4. Eltérések az eredeti kísérleti tervtől

- A modell tanítása lokális CPU helyett Google Colab Tesla T4 GPU-n történt.
- A közös DL training framework opcionális CosineAnnealingLR támogatással egészült ki.
- A Colab munkamenet megszakadása miatt a korábbi kísérleteket mentett ZIP-fájlokból állítottuk vissza, újratanítás nélkül.
- Az opcionális RN-08 (20 epoch) és RN-09 (alternatív stem) kísérleteket a korlátozott GPU-kapacitás és tanítási idő miatt elhagytuk.

Összesen 8 modellkísérlet és 80 tanítási epoch készült.

## 5. Főbb eredmények

| Metrika | Baseline (RN-00) | Optimized (RN-07) |
|---|---:|---:|
| Training accuracy | 98,80% | 96,75% |
| Validation accuracy | 93,39% | 94,94% |
| Validation Macro F1 | 93,39% | 94,94% |
| Validation log loss | 0,232670 | 0,144221 |
| Accuracy gap | 5,41 pp | 1,81 pp |
| Best epoch | 9 | 10 |
| Tanítási idő | 643,34 s | 625,85 s |
| Tanítható paraméterek | 11 172 810 | 11 172 810 |
| Checkpointméret | 42,69 MiB | 42,69 MiB |

A dropout, RandomAffine augmentáció és CosineAnnealingLR kombinációja 1,55 százalékpontos validációs pontosságjavulást eredményezett, változatlan paraméterszám mellett.

Az optimalizált modell mind a tíz kategóriában magasabb F1-score-t ért el. A legnagyobb javulás a Shirt kategóriában figyelhető meg (+4,25 pp).

## 6. Eredményfájlok

| Fájl / mappa | Tartalom |
|---|---|
| `results/experiments/resnet18_experiments.csv` | Összes kísérlet konfigurációja és metrikái |
| `results/experiments/resnet18_epoch_history.csv` | Epochonkénti tanítási és validációs metrikák |
| `results/experiments/resnet18_final_metrics.csv` | Baseline és optimalizált modell eredményei |
| `results/experiments/resnet18_classification_report.csv` | Osztályonkénti validációs metrikák |
| `results/figures/resnet18/` | Öt eredményábra |
| `artifacts/resnet18/` | Modell-checkpointok és konfigurációk |

A modellállományok és ZIP-backupok lokálisan tárolódnak, nem kerülnek Git-verziókezelésbe.

## 7. Validáció és reprodukálhatóság

- Közös adatfelosztás: 50 000 training, 10 000 validation és 10 000 official test minta.
- Közös random seed: 42.
- Tanítás előtanított súlyok nélkül.
- Training-only standardizálási statisztikák és augmentáció.
- Validációs alapú best epoch és végleges modellválasztás.
- A baseline és optimalizált checkpointok visszatöltése és a validációs metrikák reprodukálása sikeres.
- A hivatalos teszthalmaz értékelése a projekt végső modellösszehasonlítási szakaszában történik.

A tanítás Google Colab Tesla T4 GPU-n, PyTorch 2.14.1+cu130 és Torchvision 0.29.1+cu130 környezetben történt.

A különböző Colab munkamenetek és a többi modell eltérő hardverkörnyezete korlátozza a futásidők közvetlen összehasonlíthatóságát. A következtetések egyetlen adatfelosztáson és random seeden alapulnak.
