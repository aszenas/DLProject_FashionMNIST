# 3.4 – Közös Fashion-MNIST adatfelosztás

A hivatalos 60 000 elemű tanítóhalmaz eredeti sorrendjéből 50 000 tanító és
10 000 validációs indexet választunk. A `train_test_split` a címkék alapján
stratifikál, a véletlenmag 42. A különálló, 10 000 elemű hivatalos teszthalmaz
teljes egészében megmarad a végső értékeléshez.

A repóban tárolt `data/splits.npz` két egész indexlistát tartalmaz:
`train_indices` és `val_indices`. A `load_split()` ellenőrzi a méreteket és azt,
hogy minden eredeti tanítóindex pontosan egyszer szerepeljen. A két modellcsalád
futtatásához a mentett fájlt kell betölteni. Újragenerálásra a repó gyökeréből
van lehetőség:

```bash
python -m src.preprocessing.split
```

Az indexek az eredeti, szűretlen tanítóhalmaz sorrendjére mutatnak. Mindkét
előfeldolgozó ugyanazt a két tömböt kapja; a képek feldolgozását továbbra is a
saját pipeline végzi.

```python
from src.preprocessing.dataset import load_dataset
from src.preprocessing.dl import build_dataloaders
from src.preprocessing.ml import build_ml_datasets
from src.preprocessing.split import load_split

train_dataset, test_dataset = load_dataset()
train_indices, val_indices = load_split()
args = (train_dataset, test_dataset, train_indices, val_indices)

ml_datasets, preprocessor = build_ml_datasets(*args)
dl_loaders = build_dataloaders(*args)
```

A baseline egyezése a validációs mintákon közvetlenül ellenőrizhető, mert a DL
validációs DataLoader megtartja a mentett sorrendet. Az ML-vektort a megfelelő
DL-tenzor kilapított változatával kell összevetni, az osztálycímkével együtt.

Az ellenőrzés a Fashion-MNIST adatokon 50 000 / 10 000 / 10 000
mintát adott. Mind a tíz osztályból 5 000 kép került a tanító- és 1 000 a
validációs halmazba. Az újragenerált indexek egyeztek a mentett indexekkel.
A három halmaz elején, közepén és végén ellenőrzött képek címkéi és
baseline pixelértékei egyeztek a két pipeline kimenetében.
