# 4.3 – Random Forest és Simple CNN: modellkutatás és kísérlettervezés

## 1. Feladat adatai

| Megnevezés | Érték |
|---|---|
| Felelős | Beni |
| Feature branch | `feature/04-03-rf-cnn-research` |
| Dátum | 2026.11.26. |
| Kapcsolódó feladatok | 2.2, 2.3, 3.1–3.4, 5.3 |

## 2. Feladat célja

A Random Forest és a Simple CNN baseline modelljének, valamint a célzott optimalizálási kísérleteknek a megtervezése. A cél nem kizárólag a prediktív teljesítmény javítása, hanem annak vizsgálata is, hogy az eltérő adatreprezentációk, modellkomplexitás és regularizáció hogyan befolyásolják a pontosságot, az általánosítási képességet és a számítási költségeket. A dokumentum előzetes kísérleti terv, nem mérési eredményeket tartalmaz.

## 3. Elméleti háttér

### Random Forest

A Random Forest bootstrap mintákon tanított döntési fákból álló ensemble módszer. A fák csomópontjaiban véletlenszerűen kiválasztott jellemzők közül keresi a felosztást, ezáltal csökkentve a fák közötti hasonlóságot. A scikit-learn implementációja az egyes fák osztályvalószínűségeinek átlagolásával végzi a predikciót [1, 2].

A módszer előnye a nemlineáris kapcsolatok kezelése, a standardizálással szembeni csekély érzékenység és a párhuzamosítható tanítás. Korlátja, hogy a kilapított képek lokális térbeli szerkezetét nem kezeli explicit módon, továbbá a fák számának és mélységének növekedésével jelentős memória- és futásidőigénye lehet.

### Simple CNN

A konvolúciós neurális hálózatok lokális receptív mezőket és megosztott súlyokat alkalmaznak, amelyek alkalmasak a képi mintázatok hierarchikus megtanulására [3]. A konvolúciós rétegek jellemzőtérképeket hoznak létre, a ReLU nemlinearitást biztosít, a max pooling pedig csökkenti a térbeli dimenziókat. A teljesen összekapcsolt osztályozófej tíz osztályhoz tartozó logitot állít elő.

A Simple CNN-t a ResNet-18-nál kisebb, könnyebben értelmezhető konvolúciós referenciamodellnek tervezzük. A 28×28 pixeles képekből közvetlenül tanul jellemzőket, azonban a teljesítménye függ a hálózat kapacitásától, a tanítási konfigurációtól és a regularizációtól.

## 4. Bemeneti adatok és előfeltételek

- Közös adatbetöltés: `src/preprocessing/dataset.py`, `load_dataset()`.
- Közös, stratifikált split: `data/splits.npz`, `load_split()`, random seed 42.
- Training: 50 000; validation: 10 000; official test: 10 000 kép.
- RF: `build_ml_datasets()`, 784 elemű, [0, 1] tartományú float32 pixelvektorok; opcionális PCA és HOG.
- CNN: `build_dataloaders()`, `[B, 1, 28, 28]` alakú, [0, 1] tartományú float32 tenzorok; opcionális standardizálás és training-only augmentáció.
- A paramétertanulást igénylő előfeldolgozást kizárólag a tanítóhalmazon illesztjük.
- A kiegyensúlyozott osztályeloszlás miatt a baseline modellekhez nem alkalmazunk osztálysúlyozást vagy újramintavételezést.
- Könyvtárak: scikit-learn 1.6.1, PyTorch 2.7.1, Torchvision 0.22.1, NumPy, Matplotlib.

## 5. Megvalósítási terv

Az újrafelhasználható modellkódot külön Python-modulokban, a kísérleteket notebookokban tervezzük megvalósítani. A konfigurációk és mért eredmények CSV-ben, a vizualizációk a `results/figures/` alatt, a mentett modellek lokálisan az `artifacts/` alatt kapnak helyet.

Tervezett fájlok: `src/models/random_forest.py`, `src/models/simple_cnn.py`, `notebooks/02_random_forest.ipynb`, `notebooks/03_simple_cnn.ipynb`, `results/experiments/random_forest_experiments.csv`, `results/experiments/simple_cnn_experiments.csv`, valamint a két modellhez tartozó feature-dokumentációk.

## 6. Baseline konfigurációk

### 6.1. Random Forest

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
| n_jobs | -1 (erőforráskorlát esetén csökkenthető) |
| standardize | False |
| pca_components | None |
| use_hog | False |

A kiindulás a scikit-learn alapértelmezett modellje, rögzített random seeddel és párhuzamosítással.

### 6.2. Simple CNN

| Réteg | Kimeneti alak (batch dimenzió nélkül) |
|---|---|
| Input | 1 × 28 × 28 |
| Conv2d(1, 32, kernel_size=3, padding=1) + ReLU | 32 × 28 × 28 |
| MaxPool2d(2) | 32 × 14 × 14 |
| Conv2d(32, 64, kernel_size=3, padding=1) + ReLU | 64 × 14 × 14 |
| MaxPool2d(2) | 64 × 7 × 7 |
| Flatten | 3136 |
| Linear(3136, 128) + ReLU | 128 |
| Linear(128, 10) | 10 |

Tervezett tanítható paraméterszám: 421 642 (bias paraméterekkel). A tényleges implementációban ellenőrizendő.

| Paraméter | Érték | Indoklás |
|---|---|---|
| Optimizer | Adam | Adaptív gradiensalapú optimalizálás |
| Learning rate | 0.001 | Kiinduló érték |
| Loss | CrossEntropyLoss | Többosztályos osztályozás egész címkékkel és nyers logitokkal |
| Batch size | 64 | Közös DL baseline |
| Epoch | 10 | Kezdeti számítási keret |
| Dropout | 0 | Egyszerű baseline, külön regularizációs kísérlet |
| Weight decay | 0 | Alapkonfiguráció |
| Standardizálás | False | Közös DL baseline |
| Augmentáció | None | Közös DL baseline |
| Random seed | 42 | Modellinicializálás és mintakeverés |

A modell kimenetére nem alkalmazunk softmaxot a CrossEntropyLoss előtt. A predikció a legnagyobb logit indexe. Az epochonkénti tanítási és validációs loss, accuracy és macro F1-score rögzítendő.

## 7. Kísérleti terv

### 7.1. Random Forest

| Run ID | Eltérés a baseline-tól | Hipotézis |
|---|---|---|
| RF-00 | Nincs | Referencia |
| RF-01 | n_estimators=50 | Csökkenhet a tanítási idő és a modellméret |
| RF-02 | n_estimators=200 | Javulhat a prediktív teljesítmény, nagyobb költség mellett |
| RF-03 | max_depth=20 | Csökkenhet a komplexitás és a túlillesztés |
| RF-04 | min_samples_leaf=3 | Mérsékelhető a túlillesztés |
| RF-05 | max_features=log2 | Kisebb csomópontonkénti jellemzőhalmaz, eltérő költség és diverzitás |
| RF-06 | PCA, 100 komponens | Kisebb dimenzió, potenciális információvesztés |
| RF-07 | HOG | A körvonalak és lokális gradiensinformáció javíthatja az osztályozást |
| RF-08 | Ígéretes beállítások kombinációja | További javulás lehetséges, nem garantált |

A PCA és HOG hatását azonos RF-konfiguráció mellett vizsgáljuk. A HOG és a PCA előfeldolgozási időigényét külön is mérjük. Az erőforrások ismeretében a kísérletek száma csökkenthető; az eltéréseket dokumentáljuk.

### 7.2. Simple CNN

| Run ID | Eltérés a baseline-tól | Hipotézis |
|---|---|---|
| CNN-00 | Nincs | Referencia |
| CNN-01 | Dropout=0.3 a rejtett teljesen összekapcsolt réteg után | Mérsékelhető a túlillesztés |
| CNN-02 | Learning rate=0.0003 | Stabilabb, esetleg lassabb konvergencia |
| CNN-03 | Training-only standardizálás | Javulhat az optimalizálás |
| CNN-04 | Training-only RandomAffine(degrees=8, translate=(0.05, 0.05)) | Javulhat az általánosítás |
| CNN-05 | Ígéretes beállítások kombinációja | További javulás lehetséges |
| CNN-06 (opcionális) | 20 epoch a kiválasztott konfigurációval | Vizsgálható a hosszabb tanítás hatása |

Az egyedi kísérletekben egy tényezőt változtatunk, a kombinált kísérletben már több ígéretes beállítást is alkalmazhatunk. Az alapértelmezett 10 epochot és a kísérletek körét a baseline konvergenciája és mért futásideje alapján felülvizsgálhatjuk. Minden ilyen eltérést a kísérleti naplóban rögzítünk.

## 8. Értékelési protokoll

- A baseline és tuning futások azonos mentett train/validation splitet használnak.
- Rögzítendő: training és validation accuracy, macro F1, training time, inference time, modellméret, konfiguráció, seed, hardver és szoftverkörnyezet; CNN esetén epochonkénti loss és learning curve is.
- Javasolt modellválasztási szabály: legmagasabb validation accuracy; egyezéskor macro F1, majd kisebb modellméret. A közös csapatprotokoll ettől eltérő döntése elsőbbséget élvez.
- A teszthalmazon kizárólag az előre kiválasztott, véglegesített konfigurációk értékelése történik; nincs tesztalapú tuning. A közös preprocessing függvény a tesztadatokat is előállíthatja, de az optimalizálás során nem mérünk rajtuk teljesítményt.
- A tanítási és preprocessing idő külön mérendő; a mérési határokat rögzítjük. Az inferenciaidő összehasonlításához közös hardver, azonos mintaszám, dokumentált batch size, bemelegítés és ismétlések szükségesek. GPU-mérésnél szinkronizálni kell az időmérés előtt és után. A pontos közös mérési protokoll egyeztetendő.
- A mentett modellméretet a fájl bájtban mért méretével adjuk meg, a szerializálás módját és a preprocessing tárolását dokumentálva.
- A végső eredményekhez confusion matrix, összehasonlító diagramok, CNN learning curve-ök és kísérleti CSV készülnek.

## 9. Korlátok és következő lépések

A tényleges pontosság, futásidő és memóriaigény még nem ismert. A helyi fejlesztési környezet CPU-alapú; a GPU használata és a közös futásidőmérési környezet egyeztetendő. Elsőként a Random Forest baseline futását ellenőrizzük, majd célzott kísérleteket végzünk, és a Simple CNN implementációjával folytatjuk. Az eredményeket külön modellfejlesztési dokumentációk rögzítik.

## 10. Felhasznált források

1. Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32. https://doi.org/10.1023/A:1010933404324
2. Scikit-learn 1.6.1 – RandomForestClassifier. https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html
3. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 86(11), 2278–2324. https://doi.org/10.1109/5.726791
4. PyTorch – CrossEntropyLoss. https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html
5. Projekt: `docs/features/02-02-scientific-background.md`, `docs/features/02-03-problem-definition.md`, `docs/features/03_02_ml_preprocessing.md`, `docs/features/03-03-dl-preprocessing.md`, `docs/features/03-04-dataset-split.md`.
