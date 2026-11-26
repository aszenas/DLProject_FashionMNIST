
# 1. Probléma relevanciája és adatforrás

## 1.1. Problémafelvetés és gyakorlati jelentőség

A projekt célja a ruházati termékek automatikus képosztályozásának
vizsgálata, valamint három hagyományos gépi tanulási algoritmus
(Logistic Regression, Random Forest, RBF kerneles SVM) és három
mélytanulási architektúra (MLP, Simple CNN, ResNet-18)
teljesítményének összehasonlítása a Fashion-MNIST adathalmazon.

A feladat többosztályos, felügyelt tanulási problémaként
értelmezhető, amelynek során a modelleknek egy ruházati termék
képéből kell meghatározniuk annak kategóriáját.

A ruházati termékek automatikus kategorizálása különösen
az elektronikus kereskedelemben rendelkezik gyakorlati
jelentőséggel, ahol a nagy mennyiségű termékadat manuális
feldolgozása jelentős emberi erőforrást igényelhet.
A képalapú osztályozás támogathatja a termékkatalógusok
rendszerezését, a keresési és szűrési funkciókat, valamint
a termékadatok automatikus címkézését.

Donati és szerzőtársai (2019) az Adidas vállalattal
együttműködésben végzett kutatásuk során demonstrálták
a képfeldolgozási és mélytanulási módszerek alkalmazhatóságát
ruházati termékek vizuális tulajdonságainak automatikus
felismerésében és osztályozásában.

Saját vizsgálatunk célja nem kizárólag a legmagasabb
osztályozási pontosság elérése, hanem annak meghatározása,
hogy az összetettebb modellarchitektúrák milyen mértékű
teljesítményjavulást biztosítanak a számítási erőforrásigény
növekedéséhez képest. Ez különösen olyan gyakorlati
alkalmazások esetében releváns, ahol a prediktív teljesítmény
mellett a tanítási idő, az inferencia sebessége és
a modell tárolási költsége is meghatározó.

## 1.2. A Fashion-MNIST adathalmaz eredete és adatgyűjtése

A Fashion-MNIST adathalmazt Han Xiao, Kashif Rasul és
Roland Vollgraf publikálta 2017-ben a Zalando Research
keretében, gépi tanulási algoritmusok összehasonlító
értékelésének támogatására.

Az adathalmaz a Zalando online kereskedelmi platformján
elérhető 70 000 különböző termék professzionális,
elölnézeti fényképeiből készült. A termékek férfi,
női, gyermek- és uniszex kategóriákból származnak.
A fehér színű termékeket a háttérrel szembeni alacsony
kontraszt miatt kizárták az adathalmazból.

Az eredeti, 762×1000 pixeles JPEG-fényképek 51×73 pixeles
előnézeti változatait többlépéses képfeldolgozási
eljárással alakították át:

1. PNG-formátumra konvertálás.
2. A háttérhez hasonló színű képszélek eltávolítása.
3. A hosszabb képoldal átméretezése 28 pixelre.
4. A képi részletek és körvonalak élesítése.
5. A kép középre igazítása 28×28 pixeles vásznon.
6. A pixelintenzitások invertálása.
7. Konvertálás 8 bites szürkeárnyalatos képpé.

Az eljárás eredményeként egységes méretű és reprezentációjú
képek jöttek létre, amelyek közvetlenül felhasználhatók
gépi tanulási kísérletekhez (Xiao et al., 2017, 2. fejezet,
2. oldal).

### Címkézési stratégia

A képek osztálycímkéi a termékek úgynevezett silhouette
code-ján, vagyis terméktípus-azonosítóján alapulnak.
A címkéket a Zalando divatszakértői manuálisan határozták
meg, majd egy külön ellenőrző csapat felülvizsgálta azokat.

Minden termékhez pontosan egy kategória tartozik, így
az adathalmaz egycímkés, többosztályos osztályozási
feladat megoldására alkalmas.

Az eredeti tanítóhalmazhoz kategóriánként 6000 mintát
választottak ki véletlenszerűen, míg a fennmaradó
1000 minta kategóriánként a teszthalmazba került
(Xiao et al., 2017, 2. fejezet, 2–3. oldal).

### Az adathalmaz főbb jellemzői

| Tulajdonság | Érték |
|---|---|
| Adathalmaz neve | Fashion-MNIST |
| Adatforrás | Zalando Research |
| Publikálás éve | 2017 |
| Minták száma | 70 000 |
| Eredeti tanítóhalmaz | 60 000 kép |
| Eredeti teszthalmaz | 10 000 kép |
| Képméret | 28×28 pixel |
| Színcsatornák száma | 1 |
| Pixelértékek tartománya | 0–255 |
| Nyers adattípus | uint8 |
| Bemeneti jellemzők száma | 784 pixelérték |
| Osztályok száma | 10 |
| Minták száma osztályonként | 7000 |
| Osztályeloszlás | Kiegyensúlyozott |
| Feladat típusa | Felügyelt, többosztályos osztályozás |
| Adatformátum | Tömörített IDX |
| Licenc | MIT |

### Célváltozó

A célváltozó egy 0 és 9 közötti egész szám, amely
a képen szereplő termék kategóriáját azonosítja.

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

A célváltozó nominális kategorikus változó, vagyis
a numerikus címkék között nincs természetes sorrend
vagy mennyiségi kapcsolat.

### Licenc és hozzáférhetőség

Az adathalmaz a Zalando Research hivatalos GitHub
repository-jában MIT licenc alatt érhető el, amely
lehetővé teszi annak felhasználását, módosítását és
terjesztését a szerzői jogi és licencfeltételek
megtartása mellett.

Adatforrás:
https://github.com/zalandoresearch/fashion-mnist

Licenc:
https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE

A projektben az adathalmaz letöltése és betöltése
a Torchvision könyvtár FashionMNIST osztályának
segítségével történik.

### Az adathalmaz korlátai

Bár a Fashion-MNIST valódi kereskedelmi termékek képeiből
származik, az egységes háttér, az alacsony felbontás,
a színinformáció hiánya és a fehér termékek kizárása
korlátozza a valós környezet reprezentációját.

Az adathalmaz nem tartalmazza a természetes környezetben
készült ruházati fényképekre jellemző valamennyi vizuális
változatosságot, például a változó megvilágítást,
összetett hátteret vagy különböző nézőpontokat.

A kiegyensúlyozott osztályeloszlás szintén eltérhet
egy valós termékkatalógus eloszlásától.

Emiatt a kapott eredmények elsősorban az algoritmusok
kontrollált benchmark-környezetben történő összehasonlítására
alkalmasak. Nem következtethetünk belőlük közvetlenül
a modellek valós kereskedelmi környezetben nyújtott
teljesítményére.

## 2. Kutatási kérdések és előzetes hipotézisek

A 2.2. fejezetben bemutatott szakirodalmi eredmények
alapján az alábbi kutatási kérdéseket és hipotéziseket
fogalmazzuk meg.

### K1 – Prediktív teljesítmény

Kutatási kérdés:

Milyen különbségek figyelhetők meg a hagyományos ML-
és DL-modellek osztályozási teljesítményében egységes
adatfelosztási és értékelési feltételek mellett?

H1 – Hipotézis:

A konvolúciós architektúrák (Simple CNN és ResNet-18)
várhatóan magasabb Accuracy és Macro F1-score értékeket
érnek el, mint a hagyományos ML-modellek, valamint
a képeket egydimenziós jellemzővektorként feldolgozó MLP.

A feltételezés alapja a konvolúciós hálózatok térbeli
jellemzőkinyerési képessége és a 2.2. fejezetben bemutatott
korábbi összehasonlító eredmények
(Schulz et al., 2020; Nocentini et al., 2022).

Vizsgálati módszer:

- Accuracy és Macro F1-score összehasonlítása.
- Confusion matrixok elemzése.
- Osztályonkénti teljesítmény vizsgálata.

### K2 – Modellkomplexitás és prediktív teljesítmény

Kutatási kérdés:

Milyen kapcsolat figyelhető meg a modellek komplexitása
és osztályozási teljesítménye között?

H2 – Hipotézis:

A nemlineáris modellek várhatóan jobb osztályozási
teljesítményt nyújtanak, mint a lineáris Logistic Regression.

A hagyományos ML-modellek között az RBF kerneles SVM-től
várjuk a legmagasabb pontosságot.

A mélytanulási modellek közül a ResNet-18-tól várjuk
a legjobb vagy a Simple CNN-nel összemérhető
osztályozási teljesítményt.

He és szerzőtársai (2016) eredményei alapján a reziduális
kapcsolatok elősegítik a mélyebb neurális hálózatok
optimalizálását. Ez azonban nem garantálja, hogy
egy viszonylag kis felbontású képeket tartalmazó
adathalmazon a mélyebb architektúra minden esetben
jobb eredményt ér el.

Vizsgálati módszer:

- Logistic Regression és MLP összehasonlítása.
- Logistic Regression, Random Forest és SVM összehasonlítása.
- Simple CNN és ResNet-18 összehasonlítása.
- Modellméret és DL-paraméterszám vizsgálata.

### K3 – Számítási hatékonyság

Kutatási kérdés:

Milyen kompromisszum figyelhető meg az osztályozási
teljesítmény, a tanítási idő, az inferenciaidő és
a modellméret között?

H3 – Hipotézis:

A Logistic Regression várhatóan az egyik legalacsonyabb
számítási igényű modell lesz, míg a ResNet-18 a vizsgált
DL-architektúrák közül várhatóan a legnagyobb
erőforrásigénnyel rendelkezik.

Az RBF kerneles SVM tanítása a mintaszám növekedésével
jelentős számítási költséget eredményezhet, ezért
nem feltételezzük, hogy minden hagyományos ML-algoritmus
gyorsabb lesz a mélytanulási modelleknél.

A Simple CNN várhatóan kedvező kompromisszumot biztosít
az osztályozási teljesítmény és a modellkomplexitás között
(Preuveneers, 2023; Jiang, 2026).

Vizsgálati módszer:

- Tanítási idő mérése.
- Inferenciaidő mérése azonos mintaszám mellett.
- Mentett modellek méretének összehasonlítása.
- DL-modellek paraméterszámának meghatározása.

A különböző hardverkörnyezetekben kapott futásidőket
a rendelkezésre álló CPU- és GPU-erőforrások
figyelembevételével értelmezzük.

### K4 – Adatreprezentáció és előfeldolgozás

Kutatási kérdés:

Milyen hatással vannak a különböző adatreprezentációk
és előfeldolgozási módszerek a modellek teljesítményére?

H4 – Hipotézis:

A nyers pixelvektorok helyett alkalmazott HOG
jellemzőkinyerés javíthatja a hagyományos ML-modellek
osztályozási teljesítményét azáltal, hogy a képek
lokális gradiensinformációit reprezentálja
(Dalal & Triggs, 2005).

A PCA alkalmazása megfelelő komponensszám mellett
várhatóan csökkentheti a számítási igényt, miközben
az osztályozási teljesítmény jelentős része megőrizhető.

A StandardScaler hatása várhatóan az SVM és Logistic
Regression modelleknél lesz relevánsabb, mint
a Random Forest esetében.

A DL-modellek esetén a training halmazon alkalmazott
mérsékelt data augmentation javíthatja az általánosítási
képességet.

Vizsgálati módszer:

- Baseline és opcionális preprocessing konfigurációk
  összehasonlítása.
- Accuracy és Macro F1-score változásának vizsgálata.
- Feature-dimenzió és tanítási idő összehasonlítása.
- Data augmentation hatásának értékelése.

### K5 – Hiperparaméter-optimalizálás

Kutatási kérdés:

Milyen mértékben javítható az egyes modellek
teljesítménye hiperparaméter-optimalizálással?

H5 – Hipotézis:

A megfelelő hiperparaméter-konfigurációk és
regularizációs módszerek kiválasztásával a baseline
modellek többségének osztályozási teljesítménye
javítható.

A javulás mértéke várhatóan modellenként eltérő lesz,
és nem feltétlenül jár együtt a számítási
erőforrásigény csökkenésével.

Vizsgálati módszer:

- Baseline és optimalizált modellek összehasonlítása.
- Validációs és végső tesztmetrikák elemzése.
- DL learning curve-ök vizsgálata.
- Pontosságjavulás és többlet számítási költség értékelése.

## 3. Modellek várható teljesítménye

A szakirodalomban bemutatott eredmények és a választott
architektúrák tulajdonságai alapján az alábbi előzetes
elvárásokat fogalmazzuk meg.

| Modell | Várható prediktív teljesítmény | Várható számítási igény | Indoklás |
|---|---|---|---|
| Logistic Regression | A legalacsonyabbak között | Alacsony | Lineáris döntési függvény, korlátozott reprezentációs képesség |
| Random Forest | Közepes | Közepes | Nemlineáris összefüggések modellezése döntési fák segítségével |
| SVM (RBF) | Magas az ML-modelleken belül | Magas tanítási költség | Nemlineáris döntési határok és kernelalapú tanulás |
| MLP | Közepes–magas | Közepes | Nemlineáris reprezentáció, explicit térbeli struktúra nélkül |
| Simple CNN | Magas | Közepes–magas | Lokális és térbeli képi jellemzők automatikus tanulása |
| ResNet-18 | Magas, potenciálisan a legjobb | Magas | Mélyebb architektúra és reziduális kapcsolatok |

A táblázat előzetes kutatási hipotéziseket tartalmaz,
nem saját mérési eredményeket.

A feltételezett pontossági sorrend:

Logistic Regression < Random Forest < MLP ≈ SVM
< Simple CNN ≲ ResNet-18

A pontos sorrend az alkalmazott hiperparaméterektől,
előfeldolgozási módszerektől, tanítási feltételektől
és modellarchitektúráktól függően változhat.

A hipotézisek igazolása vagy elvetése kizárólag
a saját kísérletek eredményei alapján történik.

## Felhasznált irodalom

Donati, L., Iotti, E., Mordonini, G., & Prati, A. (2019).
Fashion Product Classification through Deep Learning
and Computer Vision.
Applied Sciences, 9(7), 1385.
https://doi.org/10.3390/app9071385

Xiao, H., Rasul, K., & Vollgraf, R. (2017).
Fashion-MNIST: A Novel Image Dataset for Benchmarking
Machine Learning Algorithms.
arXiv:1708.07747.
https://doi.org/10.48550/arXiv.1708.07747

He, K., Zhang, X., Ren, S., & Sun, J. (2016).
Deep Residual Learning for Image Recognition.
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, 770–778.
https://doi.org/10.1109/CVPR.2016.90

Dalal, N., & Triggs, B. (2005).
Histograms of Oriented Gradients for Human Detection.
IEEE Computer Society Conference on Computer Vision
and Pattern Recognition, 886–893.
https://doi.org/10.1109/CVPR.2005.177

Jiang, N. (2026).
Comparative Analysis and Optimization of Model
Architectures for Fashion-MNIST Image Classification.
Applied and Computational Engineering, 225, 48–54.
https://doi.org/10.54254/2755-2721/2026.CH31978

Schulz, M. A., et al. (2020).
Different scaling of linear models and deep learning
in UKBiobank brain images versus machine-learning datasets.
Nature Communications, 11, 4231.
https://doi.org/10.1038/s41467-020-18037-z

Nocentini, O., Kim, J., Bashir, M. Z., & Cavallo, F. (2022).
Image classification using multiple convolutional neural
networks on the Fashion-MNIST dataset.
Sensors, 22(23), 9544.
https://doi.org/10.3390/s22239544

Preuveneers, D. (2023).
AutoFL: Towards AutoML in a Federated Learning Context.
Applied Sciences, 13(14), 8019.
https://doi.org/10.3390/app13148019

### Technikai források

Zalando Research – Fashion-MNIST:
https://github.com/zalandoresearch/fashion-mnist

Fashion-MNIST MIT License:
https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE

Scikit-learn – Support Vector Classification:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
