
# Fashion-MNIST adathalmaz letöltése és betöltésének ellenőrzése

## 1. Feladat adatai

| Megnevezés | Érték |
|---|---|
| Felelős | Anna |
| Feature branch | `feature/03-00-dataset-loading` |
| Dátum | 2026.11.25. |
| Kapcsolódó fájlok / notebookok | `src/preprocessing/dataset.py`, `scripts/preview_dataset.py`, `requirements.txt` |

## 2. Feladat célja

A feladat célja a Fashion-MNIST adathalmaz letöltése, valamint egy közös, újrafelhasználható adatbetöltő modul kialakítása, amely a projekt során mind a hat ML- és DL-modell számára biztosítja az egységes adatforrást.
További cél az adatbetöltés működésének ellenőrzése, a képek alapvető tulajdonságainak megismerése, valamint néhány véletlenszerűen kiválasztott kép és a hozzájuk tartozó címke vizualizálása.

## 3. Elméleti háttér

A Fashion-MNIST a Zalando Research által létrehozott, nyilvánosan elérhető, címkézett képosztályozási adathalmaz, amely az eredeti MNIST adathalmaz alternatívájaként szolgál.
Az adathalmaz 70 000 darab, 28×28 pixeles, szürkeárnyalatos ruházati képet tartalmaz, amelyek 10 különböző kategóriába tartoznak.

| Címke | Kategória |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Az adatok eredetileg tömörített IDX formátumban érhetők el. A letöltéshez és betöltéshez a Torchvision könyvtár beépített FashionMNIST osztályát alkalmazzuk, amely automatikusan kezeli az adatfájlok letöltését, kibontását, valamint a képek és címkék betöltését.
A megközelítés előnye, hogy nem szükséges saját adatbetöltési logikát implementálni az IDX fájlok feldolgozásához, továbbá minden csapattag azonos módon férhet hozzá az adathalmazhoz.

## 4. Bemeneti adatok és előfeltételek

Az adathalmaz eredeti felosztása:

| Tulajdonság | Érték |
|---|---|
| Tanítóhalmaz mérete | 60 000 kép |
| Teszthalmaz mérete | 10 000 kép |
| Teljes adathalmaz mérete | 70 000 kép |
| Képméret | 28×28 pixel |
| Színcsatornák száma | 1 (grayscale) |
| Nyers képek adattípusa | uint8 |
| Pixelértékek tartománya | 0–255 |
| Osztályok száma | 10 |

Ebben a fejlesztési szakaszban az adathalmaz eredeti train/test felosztását tartjuk meg. A végleges train/validation/test split kialakítása külön feature branch feladata.
Előfeldolgozási műveleteket, normalizálást, flatteninget vagy data augmentationt ezen a ponton még nem alkalmaztunk.

### Felhasznált könyvtárak

| Könyvtár | Verzió | Funkció |
|---|---|---|
| PyTorch | 2.7.1 | Adatok tenzoros reprezentációja |
| Torchvision | 0.22.1 | Fashion-MNIST letöltése és betöltése |
| Matplotlib | 3.10.3 | Mintaképek vizualizálása |
| Ruff | 0.11.13 | Kódminőség és formázás ellenőrzése |

A mintaképek véletlenszerű kiválasztásához, valamint a fájlelérési utak kezeléséhez a Python beépített random és pathlib moduljait használjuk.

## 5. Implementáció

### 5.1. Megvalósítási lépések

1. A szükséges könyvtárak hozzáadása a requirements.txt fájlhoz, rögzített csomagverziókkal.

2. A dataset.py modul létrehozása a src/preprocessing mappában.

3. A közös load_dataset() függvény implementálása, amely a Torchvision FashionMNIST osztályának segítségével letölti és betölti az eredeti tanító- és teszthalmazt.

4. A letöltött adatfájlok lokális tárolása a data/raw/ mappában. A download=True paraméter biztosítja, hogy az adathalmaz csak akkor kerüljön letöltésre, ha még nem érhető el lokálisan.

5. A preview_dataset.py ellenőrző script implementálása, amely megjeleníti az adathalmaz alapvető tulajdonságait, valamint 10 véletlenszerűen kiválasztott tanítóképet a hozzájuk tartozó numerikus címkével és kategórianévvel.

6. A reprodukálható mintavétel biztosítása rögzített random seed (42) alkalmazásával.

### 5.2. Létrehozott vagy módosított fájlok

| Fájl | Funkció / változtatás |
|---|---|
| `src/preprocessing/dataset.py` | Közös adatletöltő és adatbetöltő modul |
| `scripts/preview_dataset.py` | Adatbetöltés ellenőrzése és mintaképek vizualizálása |
| `requirements.txt` | PyTorch, Torchvision és Matplotlib hozzáadása |
| `docs/features/03_00_dataset_loading.md` | A feature branch dokumentációja |

### 5.3. Futtatási útmutató

A parancsokat a repository gyökérkönyvtárából, aktivált virtuális környezetben szükséges futtatni.

Függőségek telepítése:

```bash
python -m pip install -r requirements.txt
```

Adathalmaz letöltése és betöltése:

```bash
python -m src.preprocessing.dataset
```

Adathalmaz vizuális ellenőrzése:

```bash
python -m scripts.preview_dataset
```

Az első futtatás során internetkapcsolat szükséges az adathalmaz letöltéséhez. A további futtatások során a program a lokálisan tárolt adatfájlokat használja.

## 6. Vizualizációk és eredmények

Az adatbetöltés ellenőrzéséhez egy egyszerű vizuális smoke test készült.

A script az alábbi információkat jeleníti meg:

- Tanító- és teszthalmaz elemszáma.
- Képek dimenziója.
- Nyers képek adattípusa.
- Osztályok száma és megnevezése.
- 10 véletlenszerűen kiválasztott tanítókép.
- A kiválasztott képek indexe, numerikus címkéje és kategórianeve.

A mintaképek egy 2×5-ös Matplotlib ábrán jelennek meg, szürkeárnyalatos formában.

## 7. Felhasznált források

### Technikai dokumentációk

1. Zalando Research – Fashion-MNIST:
   https://github.com/zalandoresearch/fashion-mnist

2. Torchvision – FashionMNIST:
   https://docs.pytorch.org/vision/0.22/generated/torchvision.datasets.FashionMNIST.html

3. Matplotlib Documentation:
   https://matplotlib.org/3.10.3/
