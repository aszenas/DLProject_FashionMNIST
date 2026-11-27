# 5.3/A – Random Forest: implementáció, optimalizálás és értékelés

## 1. Feladat adatai

| Megnevezés | Érték |
|---|---|
| Felelős | Beni |
| Feature branch | `feature/05-03-random-forest` |
| Dátum | 2026.11.26. |
| Kapcsolódó feladatok | 2.2, 2.3, 3.1–3.4, 4.3 |
| Kapcsolódó notebook | `notebooks/02_random_forest.ipynb` |
| Modellmodul | `src/models/ml/random_forest.py` |

## 2. Feladat célja

A feladat célja egy reprodukálható Random Forest osztályozó kialakítása a Fashion-MNIST adathalmazhoz, a baseline és a célzott hiperparaméter- és előfeldolgozási kísérletek összehasonlítása, valamint a validációs eredmények alapján kiválasztott konfiguráció végső tesztelése. A prediktív teljesítmény mellett a modellméretet, a tanítási és predikciós időt, továbbá a képi jellemzők szerepét is vizsgáljuk.

## 3. Elméleti háttér

A Random Forest bootstrap mintákon tanított döntési fákból álló ensemble módszer. A csomópontok felosztásakor véletlenszerűen választott jellemzőrészhalmazok mérséklik a fák közötti függőséget. A scikit-learn osztályozó a fák által becsült osztályvalószínűségek átlagát használja a predikcióhoz [1, 2]. A modell nemlineáris döntési határokat képes kialakítani, és nem igényel jellemzőnkénti standardizálást. Ugyanakkor a kilapított pixelvektor térbeli szerkezetét nem kezeli explicit módon, a nagy és mély ensemble pedig jelentős tárolási költséget okozhat.

A PCA lineáris dimenziócsökkentés, míg a HOG lokális gradiensirányokat reprezentáló jellemzőkinyerési eljárás. Hatásukat azonos Random Forest hiperparaméterek mellett vizsgáljuk. A módszertani alapokat és az előzetes hipotéziseket a 4.3-as dokumentáció tartalmazza.

## 4. Bemeneti adatok és előfeltételek

| Adathalmaz | Mintaszám | Baseline jellemzőszám |
|---|---:|---:|
| Training | 50 000 | 784 |
| Validation | 10 000 | 784 |
| Official test | 10 000 | 784 |

A közös `load_dataset()`, `load_split()` és `build_ml_datasets()` függvényeket használjuk. A mentett, 42-es seed alapján stratifikált split azonos a többi modellével. A baseline 28 × 28 pixeles képeit 0–1 tartományba skálázzuk és 784 elemű float32 vektorokká alakítjuk. Nincs további standardizálás, osztálysúlyozás vagy újramintavételezés. A PCA illesztése kizárólag a training halmazon történik; a validation és test mintákra a tanult transzformációt alkalmazzuk. A közös preprocessing a test tömböt is előállítja, de tesztpredikciót a konfiguráció rögzítése előtt nem végzünk.

Fő függőségek: scikit-learn, scikit-image, NumPy, PyTorch, Torchvision, Pandas, Matplotlib, Seaborn, Joblib és IPykernel. A pontos csomagverziók a repository `requirements.txt` fájljában találhatók. A lokális fejlesztés Windows, Python 3.12.10 és CPU-alapú környezetben zajlott; a pontos CPU-típus és RAM-kapacitás nincs rögzítve ebben a dokumentumban, a csapatszintű futásidő-összehasonlítás előtt pótolandó.

## 5. Implementáció

### 5.1. Megvalósítási lépések

1. Újrafelhasználható modellkonstruktor és baseline konfiguráció létrehozása.
2. A közös adatbetöltés, split és preprocessing integrálása.
3. A baseline tanítása, validációs értékelése, confusion matrix és pixel fontossági ábra előállítása.
4. Általános kísérletfuttató és mérési protokoll kialakítása.
5. Egyedi hiperparaméter-, PCA-, HOG- és kombinált kísérletek elvégzése, eredmények CSV-exportja.
6. A konfigurációk összehasonlítása és a végső modell kiválasztása kizárólag validációs eredmények alapján.
7. A kiválasztott konfiguráció újratanítása az 50 000 elemű training halmazon, majd a hivatalos teszthalmaz értékelése.
8. A modell, az illesztett preprocessor, a metrikák és az ábrák exportja.

### 5.2. Technikai döntések

| Döntés | Indoklás | Alternatíva |
|---|---|---|
| `build_random_forest()` modellkonstruktor | Azonos baseline, konfigurálható felülírások | Külön modellkód futásonként |
| Egy tényező változtatása az első kísérletekben | Az egyedi hatások értelmezhetősége | Teljes grid search |
| Közös split és seed 42 | Reprodukálhatóság és összehasonlíthatóság | Több seed vagy keresztvalidáció |
| `n_jobs=-1` | A rendelkezésre álló CPU-magok kihasználása | Rögzített kisebb magszám |
| Validációs accuracy szerinti rangsor | Előre rögzített modellválasztás | Hatékonyságorientált célfüggvény |
| Öt ismétlés mediánja, bemelegítéssel | Predikciós idő ingadozásának mérséklése | Egyszeri időmérés |
| Tömörítetlen Joblib-szerializálás | Konzisztens modellfájlméret | Tömörített mentés |
| Projekt színpaletta | Vizuális konzisztencia | Egyedi modellpaletta |

### 5.3. Létrehozott fájlok

| Fájl | Funkció |
|---|---|
| `src/models/ml/random_forest.py` | Random Forest modellkonstruktor |
| `notebooks/02_random_forest.ipynb` | Tanítás, kísérletek, értékelés és vizualizáció |
| `results/experiments/random_forest_experiments.csv` | 13 futás konfigurációi és metrikái |
| `results/experiments/random_forest_final_metrics.csv` | A kiválasztott modell végleges mutatói |
| `results/experiments/random_forest_classification_report.csv` | Tesztosztályonkénti metrikák |
| `results/figures/random_forest/` | Eredményábrák |
| `artifacts/ml/random_forest.joblib` | Lokálisan tárolt betanított modell |
| `artifacts/ml/random_forest_preprocessor.joblib` | Lokálisan tárolt illesztett preprocessor |
| `artifacts/ml/random_forest_baseline.joblib` | Lokálisan tárolt betanított baseline modell |
| `artifacts/ml/random_forest_baseline_preprocessor.joblib` | Lokálisan tárolt baseline preprocessor |
| `docs/features/05-03-random-forest.md` | Szakmai dokumentáció |

A modellállományok nem kerülnek Git-verziókezelésbe.

A mentés ellenőrzése során a baseline és a végleges optimalizált modell,
valamint a hozzájuk tartozó preprocessor visszatöltése is megtörténik. A
baseline preprocessor első 100 nyers validációs kép transzformációját az
eredeti előfeldolgozott jellemzőkkel, a baseline modell predikcióit pedig az
eredeti modell predikcióival vetjük össze.

### 5.4. Futtatási útmutató

A repository gyökérmappájában, a projekt virtuális környezetében:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

A `notebooks/02_random_forest.ipynb` notebookot a projekt kernelével kell megnyitni, majd a cellákat sorrendben futtatni. A közös adatbetöltő szükség esetén letölti az adatokat, a splitet a `data/splits.npz` fájlból kell betölteni. A futás során a CSV-k, ábrák és lokális modellállományok automatikusan létrejönnek. A notebook első teljes újrafuttatása előtt a végleges konfigurációt nem szabad a már ismert teszteredmények alapján módosítani.

## 6. Baseline modell

### 6.1. Konfiguráció

| Paraméter | Érték |
|---|---|
| n_estimators | 100 |
| criterion | gini |
| max_depth | None |
| min_samples_split | 2 |
| min_samples_leaf | 1 |
| max_features | sqrt |
| bootstrap | True |
| class_weight | None |
| random_state | 42 |
| n_jobs | -1 |
| standardize | False |
| pca_components | None |
| use_hog | False |

A modell bemenete 784 pixeljellemző, kimenete a tíz osztály valamelyike. Neurális hálózati paraméterszám, loss function és optimizer nem releváns.

### 6.2. Baseline eredmények (RF-00)

| Metrika | Training | Validation |
|---|---:|---:|
| Accuracy | 1,0000 | 0,8836 |
| Macro F1-score | 1,0000 | 0,8823 |

A train–validation accuracy gap 11,64 százalékpont. Az előfeldolgozás ideje 1,3763 s, a tanítás 9,2526 s, a 10 000 validációs mintán mért predikciós idő mediánja 0,0858 s, a tömörítetlen modellméret 116,1264 MiB. A perfect training accuracy és a generalizációs rés túlillesztésre utal. A Shirt kategória recallja 60,7%, F1-score-ja 67,37%.

## 7. Hiperparaméter-optimalizálás és kísérletek

### 7.1. Kísérleti terv

A baseline mellett először egyedi hiperparaméterek, majd azonos baseline modell mellett alternatív adatreprezentációk hatását vizsgáljuk. Az EDA alapján a korábban tervezett 100 PCA-komponens helyett 84 és 187 komponenst választottunk (legalább 90%, illetve 95% megőrzött training variancia). A regularizáció vizsgálatához a 10-es maximális mélységet is hozzáadtuk, az egyetlen tervezett kombinált futás helyett három célzott kombinációt vizsgáltunk. Ezek az eltérések a validációs eredmények és az erőforrásigény alapján születtek, a teszthalmaz értékelése előtt.

### 7.2. Kísérleti napló

Az accuracy és F1 értékei 0–1 skálán szerepelnek; a táblázat az eredeti kísérleti napló kerekített értékeit közli. Az inferenciaidő a 10 000 validációs minta már előfeldolgozott jellemzőin mért predikció mediánja. A modellméret a preprocessor nélküli, tömörítetlen fájlméret.

| Run ID | Eltérés a baseline-tól | Train Acc. | Val Acc. | Val Macro F1 | Gap | Prep. (s) | Train (s) | Inference (s) | Méret (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RF-00 | Baseline | 1,0000 | 0,8836 | 0,8823 | 0,1164 | 1,3763 | 9,2526 | 0,0858 | 116,1264 |
| RF-01 | 50 fa | 1,0000 | 0,8826 | 0,8813 | 0,1174 | 1,3763 | 4,3877 | 0,0548 | 57,9888 |
| RF-02 | 200 fa | 1,0000 | 0,8842 | 0,8830 | 0,1158 | 1,3763 | 16,6308 | 0,1412 | 232,4203 |
| RF-03 | max_depth=20 | 0,9942 | 0,8843 | 0,8831 | 0,1099 | 1,3763 | 7,8245 | 0,0751 | 100,9345 |
| RF-04 | min_samples_leaf=3 | 0,9890 | 0,8833 | 0,8822 | 0,1057 | 1,3763 | 8,0013 | 0,0982 | 78,5227 |
| RF-05 | max_features=log2 | 1,0000 | 0,8807 | 0,8792 | 0,1193 | 1,3763 | 3,0180 | 0,0883 | 149,2172 |
| RF-06 | PCA, 84 komponens | 1,0000 | 0,8698 | 0,8683 | 0,1302 | 1,9343 | 8,7650 | 0,0488 | 153,7465 |
| RF-07 | PCA, 187 komponens | 1,0000 | 0,8679 | 0,8659 | 0,1321 | 2,0388 | 14,3075 | 0,0609 | 170,5223 |
| RF-08 | HOG, 1296 jellemző | 1,0000 | 0,8724 | 0,8710 | 0,1276 | 32,9073 | 20,9137 | 0,1090 | 130,3490 |
| RF-09 | max_depth=10 | 0,8843 | 0,8557 | 0,8534 | 0,0286 | 1,3763 | 4,7746 | 0,0629 | 16,4752 |
| RF-10 | 200 fa, depth=20 | 0,9942 | 0,8844 | 0,8832 | 0,1098 | 1,3763 | 16,0133 | 0,1607 | 201,9181 |
| RF-11 | 100 fa, depth=20, leaf=3 | 0,9827 | 0,8808 | 0,8795 | 0,1019 | 1,3763 | 8,5706 | 0,1001 | 71,7328 |
| RF-12 | 50 fa, depth=20, leaf=3 | 0,9813 | 0,8814 | 0,8802 | 0,0999 | 1,3763 | 4,7687 | 0,0587 | 35,8706 |

### 7.3. Kísérletek értékelése

**Ensemble mérete és regularizáció.** A 200 fa csak 0,06 százalékponttal javította a baseline validációs pontosságát, miközben a modellméret megduplázódott. Az 50 fás modell közel azonos teljesítmény mellett nagyjából feleakkora méretű. A 20-as maximális mélység javította a mért validációs accuracyt és csökkentette a modellméretet, míg a 10-es korlát már 2,79 százalékpontos pontosságromlást okozott a baseline-hoz képest. A kisebb generalizációs rés önmagában nem jelent jobb modellt. A minimum leaf size növelése csökkentette a modellméretet és a generalizációs rést.

**Adatreprezentáció.** A PCA 84 és 187 komponenses futásai rendre 1,38 és 1,57 százalékponttal alacsonyabb accuracyt adtak. A jellemzőszám csökkenése ellenére mindkét modell fájlmérete növekedett, a már transzformált bemeneteken végzett predikció viszont gyorsabb volt. A növekvő modellméret lehetséges magyarázata a döntési fák összetettebb szerkezete; ennek ellenőrzése külön csomópontszám-elemzést igényelne. A HOG 1,12 százalékpontos accuracy-romlással és számottevő preprocessing- és tanításiidő-növekedéssel járt. A vizsgált beállítások mellett az eredeti pixelvektorok biztosították a legjobb prediktív teljesítményt.

**Kombinált kísérletek.** RF-10 érte el a legjobb validációs pontosságot (88,44%) és macro F1-et (88,32%). RF-03 egyetlen helyesen osztályozott validációs kép különbségével maradt el, nagyjából feleakkora modellméret és tanítási idő mellett. RF-12 a baseline-hoz képest 69,1%-kal kisebb modellméretet, 48,5%-kal rövidebb tanítást és 31,6%-kal rövidebb modellpredikciós időt eredményezett, 0,22 százalékpontos accuracy-romlás mellett. A futásidő-megállapítások az adott mérési környezetre vonatkoznak.

### 7.4. Legjobb konfiguráció

| Hiperparaméter | Baseline | Kiválasztott (RF-10) |
|---|---|---|
| n_estimators | 100 | 200 |
| max_depth | None | 20 |
| min_samples_leaf | 1 | 1 |
| min_samples_split | 2 | 2 |
| max_features | sqrt | sqrt |
| criterion | gini | gini |
| bootstrap | True | True |
| random_state | 42 | 42 |
| Adatreprezentáció | 784 pixel | 784 pixel |

A kiválasztás elsődleges kritériuma a validációs accuracy, egyezéskor a macro F1, majd a kisebb modellméret. E szabály alapján RF-10 a kiválasztott konfiguráció. A baseline-hoz képest a validációs javulás mindössze 0,08 százalékpont, ezért nem tekinthető bizonyított általánosítási előnynek. RF-03 és RF-12 a számítási hatékonyság szempontjából ígéretes alternatívák, de tesztteljesítményüket nem mértük.

## 8. Vizualizációk és végső eredmények

### 8.1. Végső tesztértékelés

A kiválasztott RF-10 konfigurációt az eredeti 50 000 elemű training halmazon újratanítottuk, a validációs accuracy megismétlődött (88,44%). Az ezt követő tesztértékelés nem szolgált további optimalizálásra.

| Mutató | Training | Validation | Test |
|---|---:|---:|---:|
| Accuracy | 0,99422 | 0,8844 | 0,8727 |
| Macro F1 | 0,994219 | 0,883249 | 0,87117 |
| Mintaszám | 50 000 | 10 000 | 10 000 |

| Számítási mutató | Érték |
|---|---:|
| Előfeldolgozás | 1,376308 s |
| Végső újratanítás | 21,718126 s |
| Tesztpredikció mediánja (10 000 minta) | 0,16582 s |
| Modellméret | 201,918099 MiB |
| Preprocessor mérete | 0,00046 MiB |

A végső újratanítás ideje eltér a korábbi RF-10 kísérleti futás 16,0133 s értékétől; a két mérés külön futáshoz tartozik. A tesztpontosság 1,17 százalékponttal alacsonyabb a validációs pontosságnál. A modell 8727 képet osztályozott helyesen és 1273-at tévesen. A training–test pontosságkülönbség 12,152 százalékpont. A regularizáció mellett továbbra is jelentős generalizációs rés figyelhető meg.

### 8.2. Osztályonkénti eredmények

| Kategória | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| T-shirt/top | 0,8171 | 0,8580 | 0,8371 | 1000 |
| Trouser | 0,9917 | 0,9560 | 0,9735 | 1000 |
| Pullover | 0,7610 | 0,7930 | 0,7767 | 1000 |
| Dress | 0,8728 | 0,9060 | 0,8891 | 1000 |
| Coat | 0,7532 | 0,8120 | 0,7815 | 1000 |
| Sandal | 0,9785 | 0,9550 | 0,9666 | 1000 |
| Shirt | 0,7116 | 0,5750 | 0,6361 | 1000 |
| Sneaker | 0,9279 | 0,9520 | 0,9398 | 1000 |
| Bag | 0,9548 | 0,9710 | 0,9628 | 1000 |
| Ankle boot | 0,9481 | 0,9490 | 0,9485 | 1000 |
| Macro avg | 0,8717 | 0,8727 | 0,8712 | 10 000 |

A Shirt a legnehezebben osztályozható kategória (recall 57,5%, F1 63,61%). A Trouser, Sandal és Bag F1-score-ja 96% feletti. A kiegyensúlyozott support miatt a macro és weighted átlagok megegyeznek, a macro recall az accuracyvel azonos.

### 8.3. EDA és tévesztési mintázatok kapcsolata

| Kategóriapár | Átlagképek cosine similarity | Baseline validation tévesztések | RF-10 test tévesztések |
|---|---:|---:|---:|
| T-shirt/top – Shirt | 0,9489 | 219 | 243 |
| Pullover – Coat | 0,9841 | 216 | 221 |
| Pullover – Shirt | 0,9832 | 160 | 176 |
| Coat – Shirt | 0,9874 | 143 | 146 |
| Összesen | – | 738 | 786 |

A tévesztések mindkét irányban összegezve szerepelnek. A négy kategóriapár a baseline 1164 validációs hibájának 63,4%-át, a kiválasztott modell 1273 teszthibájának 61,7%-át adja. Ez összhangban áll az EDA vizuális hasonlóságra vonatkozó megállapításaival, de nem bizonyít oksági összefüggést, illetve nem helyettesíti a teljes hasonlósági mátrix statisztikai elemzését. A két oszlop eltérő modellen és eltérő adathalmazon készült; közvetlen javulás vagy romlás nem vezethető le belőlük.

### 8.4. Pixel-level feature importance

A nyers pixelvektorokon betanított Random Forest `feature_importances_` attribútumát az eredeti 28 × 28 képméretre alakítjuk vissza. A baseline ábrán a középső függőleges tengely és egyes kontúrhoz közeli területek fontossága kiemelkedő, a képszélek és sarkok szerepe kisebb. A végleges modellre is elkészül ugyanez az ábra. Az impurity-based fontosság globális, a fák felosztásaiban mért normalizált impurity-csökkenéshez kapcsolódó mutató; nem egyedi predikciós magyarázat és nem oksági mérőszám. Korrelált jellemzők esetén a fontosság megoszolhat. PCA vagy HOG reprezentációra a pixelrácsba való közvetlen visszaalakítás nem lenne értelmezhető.

### 8.5. Kapcsolódó ábrák

A `results/figures/random_forest/` mappában:

- `baseline_confusion_matrix.png`
- `baseline_feature_importance.png`
- `experiment_accuracy_comparison.png`
- `experiment_efficiency_comparison.png`
- `accuracy_model_size_tradeoff.png`
- `final_confusion_matrix.png`
- `final_feature_importance.png`

A kísérleti és végleges CSV-állományok a `results/experiments/` mappában találhatók.

## 9. Tesztelés és validáció

A közös preprocessing modulok kilenc automatizált tesztje a fejlesztés előtt sikeresen lefutott. A notebook ellenőrzi a közös adathalmazok méreteit és a stratifikált training/validation osztályeloszlást. A végleges konfiguráció újratanítása reprodukálta a kiválasztási accuracyt. A modellmentés és visszatöltés ellenőrzése a notebook része; a végső teljes notebook-újrafuttatást, a Ruff linting és formázási ellenőrzést, valamint a Git-státuszt a PR előtt lokálisan kell ellenőrizni. Ezek sikeressége itt még nincs külön igazolva.

## 10. Felmerült problémák és korlátok

| Megfigyelés | Lehetséges ok / kezelés |
|---|---|
| Jelentős generalizációs rés | A faensemble nagy kapacitása; mélység és levélméret vizsgálata |
| PCA mellett nagyobb modellfájl | Lehetséges összetettebb faalak; külön szerkezeti mérés szükséges a magyarázat ellenőrzéséhez |
| HOG számítási többletköltsége | Képenkénti jellemzőkinyerés |
| Futásidő ingadozása | Hardver- és terhelésfüggés; közös mérési feltételek szükségesek |

Egyetlen split és seed alapján vizsgáltuk a konfigurációkat; nincs több seedes bizonytalansági becslés. A kis validációs különbségek nem jelentenek igazolt statisztikai előnyt. Az inferenciaidő a már előfeldolgozott bemeneten végzett modellpredikció idejét mutatja, nem end-to-end késleltetést. A preprocessing idő mindhárom részhalmaz előállítását tartalmazza; a baseline időt a nyers pixeles futásoknál újrafelhasználtuk. A modellméret nem azonos a futás közbeni RAM-igénnyel. A végső összehasonlításban az eltérő hardverek és szerializálási módszerek korlátozzák a közvetlen összevetést.

## 11. Következtetések és tanulságok

A Random Forest kiválasztott konfigurációja 87,27%-os tesztpontosságot és 87,12%-os macro F1-score-t ért el. A 13 kísérlet fő tanulsága nem a baseline jelentős javítása, hanem a pontosság és számítási költségek közötti kompromisszum feltárása.

- **K1 – Prediktív teljesítmény:** a Random Forest önálló referenciaeredménye rendelkezésre áll; a többi modellhez viszonyított hipotézis csak a közös végső összehasonlításban értékelhető.
- **K2 – Modellkomplexitás:** a nagyobb ensemble nem adott arányos teljesítménynyereséget; a túl erős regularizáció viszont rontotta a pontosságot.
- **K3 – Számítási hatékonyság:** az RF-03 és RF-12 validációs eredményei kedvezőbb költség–pontosság kompromisszumot mutatnak, mint az RF-10.
- **K4 – Adatreprezentáció:** a vizsgált PCA és HOG konfigurációk nem javították az accuracyt. A PCA gyorsította a modellpredikciót, de a modellméret és az end-to-end költség szempontjából nem igazolt egyértelmű előnyt.
- **K5 – Hiperparaméter-optimalizálás:** a legjobb validációs accuracy 0,08 százalékponttal nőtt, ami 8 többlet helyes validációs predikciónak felel meg. A számítási hatékonyságra gyakorolt hatás ennél számottevőbb volt.

A vizuálisan hasonló felsőruházati kategóriák jelentős részt képviselnek a hibákban. Ez indokolja a konvolúciós architektúrák térbeli jellemzőkinyerési képességének vizsgálatát, de a CNN előnyét még nem igazolja.

## 12. Továbbfejlesztési lehetőségek

- Több véletlenmag vagy ismételt split alkalmazása a validációs különbségek bizonytalanságának becslésére.
- End-to-end inferencia és memóriaigény mérése azonos hardveren, közös mérési protokoll szerint.
- A faensemble csomópontszámának és mélységeloszlásának elemzése, illetve permutation importance alkalmazása a pixel fontosság robusztusabb értelmezésére.
- A feature importance és a CNN feature map vizualizációk kvalitatív összevetése, az eltérő jelentésük figyelembevételével.

Ezek további kutatási lehetőségek, nem részei a most lezárt modellválasztásnak.

## 13. Felhasznált források

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324
2. Scikit-learn 1.6 – RandomForestClassifier: https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html
3. Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. https://doi.org/10.48550/arXiv.1708.07747
4. Projekt: `docs/features/02-02-scientific-background.md`, `02-03-problem-definition.md`, `03-01-eda.md`, `03_02_ml_preprocessing.md`, `03-04-dataset-split.md`, `04-03-rf-cnn-research.md`.
5. Saját mérési eredmények: `notebooks/02_random_forest.ipynb` és a kapcsolódó kísérleti CSV-k.
