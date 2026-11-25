# 3.3 – Közös DL-előfeldolgozás

`feature/03-03-dl-preprocessing`

## Cél és bemenet

Az MLP, a CNN és a ResNet-18 ugyanazt a három PyTorch DataLoadert használhatja. A
`build_dataloaders` nyers, egycsatornás PIL-képeket és egész osztálycímkéket adó
tanító- és tesztadatkészletet vár. Ezt a formátumot használja a közös
`load_dataset()` is. A train és a validation indexei a hivatalos tanítóhalmaz
eredeti sorrendjére hivatkoznak, és a 3.4 közös felosztásából érkeznek. A test
loader végigmegy a teljes hivatalos teszthalmazon.

## Beállítások és használat

A következő kód akkor használható a teljes adathalmazon, amikor a közös
betöltő és a 3.4 felosztása elérhető. A `train_indices` és `val_indices`
változókhoz a közös splitből kiolvasott indexeket kell átadni.

```python
from src.preprocessing.dataset import load_dataset
from src.preprocessing.dl import build_dataloaders

DL_SETTINGS = {
    "batch_size": 64,
    "seed": 42,
    "standardize": False,
    "train_augmentation": None,
}

train_dataset, test_dataset = load_dataset()
loaders = build_dataloaders(
    train_dataset, test_dataset, train_indices, val_indices, **DL_SETTINGS
)
images, labels = next(iter(loaders["train"]))
```

A baseline a `ToTensor()` transzformációt használja. Egy batch képei
`[B, 1, 28, 28]` alakú, 0–1 közötti `float32` tenzorok. A címkék
`[B]` alakú `torch.long` tenzorok. Az MLP-nél a lapítás a hálózaton belül
történik. A loader seedje a tanítóminták keverési sorrendjét vezérli. A
split seedjét a 3.4 kezeli. A modell inicializálásához és a véletlen
augmentációhoz a futtató kód állítsa be a megfelelő globális seedeket is.

Kísérlethez a `train_augmentation` lehet például a Torchvision
`RandomAffine(degrees=8, translate=(0.05, 0.05))` transzformációja. Csak a
tanítóképeken fut, a tenzorrá alakítás előtt. A `standardize=True` az eredeti
tanítóképek pixeleiből számolja az átlagot és a szórást, és ugyanezeket
alkalmazza a validációs és tesztképekre is.

## Ellenőrzés

A repó gyökeréből, adatletöltés nélkül futtatható:

```bash
python -m unittest discover -s tests -p "test_dl_preprocessing.py"
```

A közös split bekötése után ellenőrizendő az 50 000 / 10 000 / 10 000
elemszám, valamint hogy ugyanazon eredeti képindexen az ML-vektor és a
kilapított DL-tenzor baseline értékei és címkéi egyeznek.
