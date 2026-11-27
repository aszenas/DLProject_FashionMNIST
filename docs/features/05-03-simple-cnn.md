# 05-03 Simple CNN

## Összefoglalás

A Fashion-MNIST feladathoz egy újrafelhasználható PyTorch-alapú Simple CNN modult és tanítási keretrendszert készült. Az öt lefuttatott kísérlet közül a CNN-03, vagyis a training halmazon meghatározott standardizálást használó modell lett a végleges modell. A kiválasztás programozottan, validation accuracy, azonosság esetén macro F1, majd modellméret szerint történik.

## Elméleti háttér és architektúra

A konvolúciós hálózat lokális térbeli mintázatokat tanul megosztott súlyokkal. A modell két, ReLU-aktivációval követett konvolúciós réteget és max-poolingot, majd 128 egységű teljesen kapcsolt réteget és tíz logitot használ:

`Conv(1,32,3,padding=1) → ReLU → MaxPool(2) → Conv(32,64,3,padding=1) → ReLU → MaxPool(2) → Flatten(3136) → Linear(128) → Dropout → Linear(10)`

A `SimpleCNN` nyers logitokat ad vissza, softmax nélkül. A konvolúciós rétegek külön attribútumként elérhetők a feature map vizualizációhoz. A tanítható paraméterek száma 421 642.

## Implementáció és reprodukálhatóság

Az implementáció a [simple_cnn.py](../../src/models/dl/simple_cnn.py) és a [dl.py](../../src/training/dl.py) modulokban található. A notebook a közös `load_dataset()`, `load_split()` és `build_dataloaders()` interfészeket használja. A felosztás 50 000 training, 10 000 validation és 10 000 test kép. A baseline batch size 64, seed 42, Adam optimalizálóval, 0,001 learning rate-tel, 10 epochon, dropout és augmentáció nélkül készült.

A futtatási környezet Python 3.12.10, PyTorch 2.7.1+cpu és Torchvision 0.22.1+cpu volt. CUDA nem volt elérhető, a futtatás CPU-n történt. A training loss mintaszámmal súlyozott, a macro F1 az epoch összes predikciójából számított. A legjobb checkpoint kiválasztása validation accuracy, macro F1, majd korábbi epoch alapján történt, a tényleges `state_dict` másolatával.

## Kísérletek és learning curve-ök

| Run | Eltérés | Best epoch | Train acc. | Val. acc. | Train F1 | Val. F1 | Accuracy gap | Training idő (s) | Full inference (s) | Modell (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-00 | baseline | 8 | 0,9555 | 0,9251 | 0,9554 | 0,9250 | 0,0304 | 366,22 | 1,56 | 1,612 |
| CNN-01 | dropout=0,3 | 10 | 0,9526 | 0,9266 | 0,9524 | 0,9266 | 0,0260 | 793,48 | 8,04 | 1,612 |
| CNN-02 | learning rate=0,0003 | 10 | 0,9237 | 0,9145 | 0,9235 | 0,9144 | 0,0092 | 911,43 | 7,71 | 1,612 |
| CNN-03 | standardize=True | 10 | 0,9818 | 0,9289 | 0,9818 | 0,9291 | 0,0529 | 974,99 | 6,33 | 1,612 |
| CNN-04 | RandomAffine | 9 | 0,9320 | 0,9225 | 0,9320 | 0,9226 | 0,0095 | 757,70 | 2,27 | 1,612 |

A learning curve-ök és az összehasonlító ábrák a [simple_cnn figures](../../results/figures/simple_cnn/) mappában vannak. A CNN-03 érte el a legmagasabb validation accuracy-t és macro F1-et, ezért ez lett a végleges modell. CNN-05 kombinált konfigurációját és CNN-06 hosszabb, 20 epochos futását CPU-erőforráskorlát miatt elhagytuk; ezekhez nem készült fiktív eredmény.

Az inference időkben a full mérés tartalmazza a DataLoader-iterációt, tensor-transzfert és forward pass-t, míg a forward-only mérés előre betöltött device batch-eken csak a modellszámítást méri. Az értékek CPU-terheléstől és cache-állapottól ingadozhatnak, ezért a konfigurációk közötti időeltérésekből nem vonható le oksági következtetés.

## Végleges test-set értékelés

A tesztértékelés kizárólag a már mentett CNN-00 és CNN-03 checkpointokból történt, újratanítás nélkül. CNN-03 esetén a mentett training-only paraméterek: mean `0,28627583354020947`, std `0,3531853346544167`. A test loader ugyanazt a batch size-t (64) és a checkpoint preprocessing konfigurációját használta.

| Modell | Test loss | Test accuracy | Test macro F1 | Full inference (s) | Forward-only (s) | Paraméter | Méret (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN-00 | 0,23534 | 0,9195 | 0,91920 | 2,4491 | 1,7146 | 421 642 | 1,612 |
| CNN-03 | 0,26554 | 0,9240 | 0,92415 | 2,7332 | 1,5209 | 421 642 | 1,612 |

A class-level precision, recall, F1 és support a [simple_cnn_classification_report.csv](../../results/experiments/simple_cnn_classification_report.csv) fájlban, az összesített metrikák a [simple_cnn_final_metrics.csv](../../results/experiments/simple_cnn_final_metrics.csv) fájlban találhatók.

| Osztály | CNN-00 precision / recall / F1 | CNN-03 precision / recall / F1 |
|---|---|---|
| T-shirt/top | 0,852 / 0,893 / 0,872 | 0,863 / 0,887 / 0,875 |
| Trouser | 0,995 / 0,975 / 0,985 | 0,988 / 0,989 / 0,989 |
| Pullover | 0,908 / 0,836 / 0,870 | 0,903 / 0,861 / 0,881 |
| Dress | 0,918 / 0,922 / 0,920 | 0,921 / 0,925 / 0,923 |
| Coat | 0,831 / 0,927 / 0,877 | 0,887 / 0,887 / 0,887 |
| Sandal | 0,990 / 0,971 / 0,980 | 0,983 / 0,986 / 0,985 |
| Shirt | 0,797 / 0,740 / 0,768 | 0,774 / 0,791 / 0,782 |
| Sneaker | 0,944 / 0,987 / 0,965 | 0,957 / 0,977 / 0,967 |
| Bag | 0,984 / 0,982 / 0,983 | 0,989 / 0,977 / 0,983 |
| Ankle boot | 0,982 / 0,962 / 0,972 | 0,981 / 0,960 / 0,970 |

A confusion matrixek külön képfájlként exportálódtak: [CNN-00](../../results/figures/simple_cnn/simple_cnn_cnn-00_test_confusion_matrix.png) és [CNN-03](../../results/figures/simple_cnn/simple_cnn_cnn-03_test_confusion_matrix.png). A legnehezebb osztály mindkét modellnél a Shirt; a CNN-03-nál a javuló recall mellett továbbra is a T-shirt/top, Pullover és Coat kategóriákkal vannak a legnagyobb átfedések. Az EDA cosine similarity elemzése szintén magas hasonlóságot mutatott a Shirt/Coat, Coat/Pullover és Shirt/Pullover osztálypárok között. Ez leíró összevetés, nem oksági állítás.

## Feature mapek és modellmentés

A [simple_cnn_final_feature_maps.png](../../results/figures/simple_cnn/simple_cnn_final_feature_maps.png) ugyanazon dokumentált validation mintán mutatja a CNN-03 eredeti képét, valamint az első és második konvolúciós réteg ReLU utáni, pooling előtti első nyolc csatornáját. A rétegek egymás után egyre összetettebb lokális mintázatokat emelnek ki, de a vizualizáció kvalitatív.

A végleges kiválasztott checkpoint a `artifacts/dl/simple_cnn.pt`, az eredeti baseline a `artifacts/dl/simple_cnn_baseline.pt`. Mindkettő tartalmazza a modell- és preprocessing-konfigurációt, a checkpoint state dict-jét és a legjobb epochot; CNN-03 a training-only mean/std értékeket is tárolja. A checkpoint visszatöltése és a modellkonfiguráció egyezése ellenőrzött. Az artifactokat a [`.gitignore`](../../.gitignore) kizárja.

## Kutatási kérdések és korlátok

- A Simple CNN megfelelő baseline-t ad a Fashion-MNIST képekhez; a standardizálás kis, de mérhető validation javulást adott a rögzített spliten.
- A dropout és az augmentáció nem javította a CNN-03 eredményét, az alacsonyabb learning rate pedig gyengébb validációs teljesítményt adott.
- Egyetlen split és seed miatt az eredmények bizonytalansága nem becsülhető megbízhatóan; további seedek vagy cross-validation szükségesek lennének.
- A CPU-s időmérések terhelésérzékenyek. A forward-only és full inference mérések szétválasztása javítja az értelmezhetőséget, de nem szünteti meg a rendszerzajt.
- CNN-05 és CNN-06 erőforráskorlát miatt kimaradt, ezért a kombinált preprocessing és a hosszabb tanítás hatásáról nincs eredmény.
- Hosszabb távon több split/seed, célzottabb augmentáció, GPU-s benchmark és a feature mapek kvantitatív elemzése indokolt.

## Ellenőrzések

A notebook JSON-validációja, Ruff linting és formázás, a unittest tesztek, a `git diff --check`, a checkpoint-visszatöltés és az artifact ignore-ellenőrzés lefuttatható. Teljes, clean-kernel notebook-futtatás nem történt, mert az újratanítási cellák ismételt futtatása a feladat szerint tiltott; a végső tesztértékelés külön, kizárólag a meglévő checkpointokból futott.
