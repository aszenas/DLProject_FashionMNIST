
# 4.1.1 – Logistic Regression: modellkutatás és kísérlettervezés

## 1. Feladat adatai

| Megnevezés | Érték |
|---|---|
| Felelős | Anna |
| Feature branch | `feature/04-01-01-logistic-regression-research` |
| Modell | Multinomial Logistic Regression |

## 2. Feladat célja

A multinomiális logisztikus regresszió elméleti hátterének bemutatása, baseline konfigurációjának és optimalizálási kísérleteinek megtervezése a Fashion-MNIST adathalmazon. A modell lineáris referenciaként szolgál a nemlineáris ML- és DL-modellek pontosságának, komplexitásának és számítási igényének összehasonlításához.

## 3. Elméleti háttér és modellválasztás

A logisztikus regresszió felügyelt, valószínűségi osztályozási algoritmus, amely a bemeneti jellemzők súlyozott lineáris kombinációjára épül [1]. Mivel a Fashion-MNIST tíz egymást kizáró kategóriát tartalmaz (lásd 2.3. fejezet), a többosztályos kiterjesztést, a multinomiális logisztikus regressziót alkalmazzuk.

A modell minden osztályhoz lineáris pontszámot (logitot) rendel:

$$
z_k = w_k^Tx+b_k
$$

A softmax függvény a pontszámokat osztályvalószínűségekké alakítja:

$$
P(y=k|x)=\frac{e^{z_k}}{\sum_{j=0}^{K-1}e^{z_j}}
$$

A valószínűségek összege 1, a predikció pedig a legmagasabb valószínűségű osztály. A modell tanítása a cross-entropy veszteség és az L2 regularizációs büntetőtag összegének minimalizálásával történik. A regularizáció inverz erősségét a `C` hiperparaméter szabályozza. Alacsonyabb C erősebb, magasabb C gyengébb regularizációt jelent [1, 4].

| Előnyök | Korlátok |
|---|---|
| Egyszerű modellstruktúra, kevés paraméter | Lineáris döntési határok a jellemzőtérben |
| Konvex optimalizálási probléma | Korlátozott reprezentációs képesség |
| Várhatóan gyors inferencia | Nem kezeli explicit módon a kép térbeli struktúráját |
| Értelmezhető súlyvektorok | Érzékeny lehet az adatreprezentációra |
| Közvetlen osztályvalószínűségek | Hasonló ruházati kategóriák elkülönítése nehéz lehet |

A szakirodalomban a logisztikus regresszió megközelítőleg 84,2%-os pontosságot ért el a Fashion-MNIST adathalmazon [2]. Egyszerűsége miatt megfelelő referenciamodell, amelyhez képest vizsgálható az összetettebb modellek és adatreprezentációk által biztosított teljesítményjavulás. A publikált pontosság nem tekinthető garantált saját eredménynek.

## 4. Bemeneti adatok és előfeltételek

- Közös adatbetöltés: `load_dataset()`.
- Közös stratifikált adatfelosztás: `load_split()`, `data/splits.npz`, seed 42.
- Training: 50 000; validation: 10 000; official test: 10 000 kép.
- Közös ML preprocessing: `build_ml_datasets()`.
- Baseline bemenet: 784 elemű, [0, 1] tartományú float32 pixelvektorok.
- Kimenet: 0–9 közötti osztálycímke, illetve tíz osztályvalószínűség.
- Opcionális preprocessing: StandardScaler, PCA és HOG; az illesztést igénylő transzformációk kizárólag a training halmazon illeszthetők.
- Könyvtárak: scikit-learn 1.6.1, NumPy, scikit-image, Matplotlib.

A baseline modell 7840 súlyt és 10 bias paramétert, összesen 7850 értéket tárol. A tárolt paraméterek száma a bemeneti dimenzió függvényében változik, és nem tartalmazza a preprocessing objektumok paramétereit.

## 5. Megvalósítási terv

A modellimplementáció külön Python-modulban, a tanítás és optimalizálás notebookban történik. A kísérleti konfigurációk és eredmények CSV-ben, az ábrák a `results/figures/`, a mentett modellek és preprocessing objektumok az `artifacts/` mappában kapnak helyet.

Tervezett fájlok: `src/models/ml/logistic_regression.py`, `notebooks/02_logistic_regression.ipynb`, `results/experiments/logistic_regression.csv`, valamint a modelltréninghez tartozó feature-dokumentáció.

## 6. Baseline konfiguráció

| Paraméter | Érték | Indoklás |
|---|---|---|
| solver | lbfgs | Multinomiális optimalizáció támogatása |
| penalty | l2 | Súlyok regularizációja |
| C | 1.0 | Kiinduló regularizációs erősség |
| max_iter | 1000 | Konvergencia biztosítására választott iterációs keret |
| tol | 0.0001 | Optimalizáció leállási toleranciája |
| fit_intercept | True | Bias paraméter tanulása |
| class_weight | None | Kiegyensúlyozott osztályeloszlás |
| random_state | 42 | Az ezt használó solverek reprodukálhatósága |
| standardize | False | Közös ML baseline |
| pca_components | None | Nincs dimenziócsökkentés |
| use_hog | False | Nyers pixelreprezentáció |

A scikit-learn 1.6.1 verziójában az L-BFGS solver automatikusan multinomiális veszteségfüggvényt alkalmaz tíz osztály esetén. A `multi_class` paramétert nem szükséges explicit megadni. A `random_state` az L-BFGS működését nem vezérli [4].

## 7. Kísérleti terv

| Run ID | Eltérés a baseline-tól | Hipotézis |
|---|---|---|
| LR-00 | Nincs | Lineáris referenciamodell |
| LR-01 | C=0.01 | Erősebb regularizáció, kisebb súlyok, potenciális underfitting |
| LR-02 | C=0.1 | Erősebb regularizáció, javulhat az általánosítás |
| LR-03 | C=10.0 | Gyengébb regularizáció, jobb illeszkedés, potenciális overfitting |
| LR-04 | standardize=True | Változhat a konvergencia és a regularizáció hatása |
| LR-05 | pca_components=100 | Kisebb modellillesztési költség, potenciális információvesztés |
| LR-06 | pca_components=200 | Nagyobb információmegőrzés, mint LR-05 esetén |
| LR-07 | use_hog=True | A lokális gradiensinformáció javíthatja az osztályozást |
| LR-08+ | Legjobb preprocessing és C optimalizálása | További teljesítményjavulás lehetséges |

Az egyedi kísérletekben egy tényezőt változtatunk, a többi baseline paraméter változatlan marad. A PCA komponensszámai az EDA cumulative explained variance elemzése alapján pontosíthatók. A kiválasztott preprocessing konfiguráció esetén ismét megvizsgáljuk a C értékének hatását.

| Adatreprezentáció | Jellemzők száma | Tárolt modellparaméterek |
|---|---:|---:|
| Baseline / StandardScaler | 784 | 7850 |
| PCA 100 | 100 | 1010 |
| PCA 200 | 200 | 2010 |
| HOG | 1296 | 12970 |

A PCA alkalmazásával várhatóan csökkenthető a modellillesztés számítási költsége. A HOG a képek lokális gradiensinformációinak reprezentálására alkalmas jellemzőkinyerési módszer [3], amelytől a lineáris osztályozó teljesítményének javulását várjuk. Mindkét esetben figyelembe vesszük az előfeldolgozás többletköltségét.

Opcionális kísérletek: L1 és Elastic Net regularizáció SAGA solverrel (azonos solverrel végzett L2 kontroll mellett), StandardScaler + PCA, illetve HOG + PCA kombinációja.

## 8. Értékelési protokoll és vizualizációk

- Minden futtatás azonos mentett train/validation splitet használ.
- Rögzítendő: training és validation accuracy, macro F1-score, log loss, preprocessing time, training time, inference time, modellméret, konfiguráció és futtatási környezet.
- Modellválasztás: legmagasabb validation accuracy, egyezéskor macro F1-score, majd a közös csapatprotokoll szerinti döntési szempont.
- Konvergencia ellenőrzése: `n_iter_` és `ConvergenceWarning`; szükség esetén a max_iter növelése.
- A preprocessing és modellillesztés idejét külön mérjük, a modell és preprocessing objektumok tárolási igényét külön is rögzítjük.
- A teszthalmazon kizárólag a véglegesített baseline és optimalizált konfiguráció értékelése történik, tesztalapú tuning nélkül.

Tervezett vizualizációk:

- Baseline és optimalizált modell confusion matrixa.
- Training és validation metrikák összehasonlítása.
- C hiperparaméter és validation accuracy kapcsolata.
- Preprocessing konfigurációk pontosságának és futásidejének összehasonlítása.
- Tanítási idő, inferenciaidő és modellméret összehasonlítása.
- Osztályonkénti pixelsúlyok vizualizálása 28×28-as heatmapeken.
- Opcionálisan hibásan osztályozott validációs minták.

Az ábrák a TERV.xlsx fájlban meghatározott közös színpaletta alapján készülnek.

## 9. Korlátok és következő lépések

A dokumentáció előzetes kísérleti tervet tartalmaz. A kísérletek köre a baseline futásideje, konvergenciája és a rendelkezésre álló számítási kapacitás alapján módosítható. A tényleges modellimplementáció, mérési eredmények és kutatási hipotézisek értékelése külön dokumentációban készül el.

## 10. Felhasznált források

1. Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. Journal of Statistical Software, 33(1), 1–22. https://doi.org/10.18637/jss.v033.i01
2. Mukhamediev, R. I. (2024). State-of-the-Art Results with the Fashion-MNIST Dataset. Mathematics, 12(20), 3174. https://doi.org/10.3390/math12203174
3. Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. IEEE CVPR, 886–893. https://doi.org/10.1109/CVPR.2005.177
4. Scikit-learn 1.6 – LogisticRegression. https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html
