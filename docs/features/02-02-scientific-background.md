# 2.2. Tudományos háttér

A képosztályozási feladatok megoldására számos gépi tanulási megközelítés alkalmazható, amelyek az adatok reprezentációjában, a jellemzők kinyerésében és a tanulási mechanizmusokban egyaránt különböznek. A megfelelő modell kiválasztása során az osztályozási pontosság mellett annak komplexitása, általánosítási képessége és számítási erőforrásigénye is meghatározó szempont. Az alábbi szakirodalmi áttekintés célja a hagyományos gépi tanulási és mélytanulási megközelítések összehasonlításával kapcsolatos kutatási eredmények összefoglalása, különös tekintettel a Fashion-MNIST adathalmazon végzett vizsgálatokra.

Xiao és szerzőtársai (2017) a Fashion-MNIST adathalmazt az eredeti, kézírásos számjegyeket tartalmazó MNIST összetettebb alternatívájaként hozták létre, lehetőséget biztosítva különböző osztályozási algoritmusok teljesítményének összehasonlítására. Mukhamediev (2024) összefoglalásában a hagyományos gépi tanulási modellek közül a logisztikus regresszió 84,2%-os, a Random Forest 87,3%-os, míg a nemlineáris tartóvektor-gép 89,7%-os osztályozási pontosságot ért el. Ezek a referenciaeredmények jelzik, hogy a klasszikus algoritmusok is eredményesen alkalmazhatók képosztályozási feladatokra, ugyanakkor a nemlineáris összefüggések modellezése további teljesítményjavulást eredményezhet.

A mélytanulási megközelítések egyik lényeges előnye az adatokból történő automatikus jellemzőkinyerés. A konvolúciós neurális hálózatok az egydimenziós pixelvektorokra épülő megközelítésekkel szemben a képek térbeli struktúráját is közvetlenül képesek hasznosítani. Schulz és szerzőtársai (2020) eredményei alapján a Fashion-MNIST esetében a lineáris modellektől a nemlineáris kernelalapú, majd a mélytanulási modellek felé haladva javult az osztályozási teljesítmény, amelyet a rendelkezésre álló tanítóadatok mennyisége is befolyásolt. Nocentini és szerzőtársai (2022) különböző konvolúciós architektúrák vizsgálata során 94,04%-os pontosságot értek el egy 15 konvolúciós réteget tartalmazó hálózattal. Eredményeik ugyanakkor arra is rámutattak, hogy a hálózat mélységének növelése nem eredményez szükségszerűen jobb teljesítményt, valamint a vizuálisan hasonló ruházati kategóriák elkülönítése továbbra is kihívást jelent.

A magasabb prediktív teljesítmény azonban nagyobb számítási erőforrásigénnyel is együtt járhat. Preuveneers (2023) a Fashion-MNIST adathalmazon végzett referenciakísérleteiben az osztályozási pontosság és az F1-mutató mellett a futásidőt és a memóriaigényt is vizsgálta. Az egyik vizsgált hardverkörnyezetben a konvolúciós neurális hálózat 91,3%-os pontosságot ért el, míg a Random Forest és a többrétegű perceptron rendre 84,9%-os és 86,9%-os pontosság mellett rövidebb futásidőt mutatott. Az eredmények szemléltetik a prediktív teljesítmény és a számítási hatékonyság közötti lehetséges kompromisszumot.

A szakirodalmi eredmények összességében a konvolúciós megközelítések előnyét jelzik a Fashion-MNIST osztályozási feladatában, ugyanakkor a modellválasztás során a pontosságon túlmutató szempontok figyelembevétele is indokolt. Mivel a publikált eredmények eltérő előfeldolgozási, tanítási és hardverkörülmények között születtek, közvetlen összehasonlíthatóságuk korlátozott. Saját vizsgálatunk ezért egységes adatfelosztás és értékelési módszertan mellett hasonlítja össze a választott modelleket, vizsgálva az osztályozási teljesítmény, a modellkomplexitás, a számítási igény és a hiperparaméter-optimalizálás közötti összefüggéseket.

## Felhasznált irodalom

Mukhamediev, R. I. (2024). State-of-the-art results with the Fashion-MNIST dataset. *Mathematics, 12*(20), 3174. https://doi.org/10.3390/math12203174

Nocentini, O., Kim, J., Bashir, M. Z., & Cavallo, F. (2022). Image classification using multiple convolutional neural networks on the Fashion-MNIST dataset. *Sensors, 22*(23), 9544. https://doi.org/10.3390/s22239544

Preuveneers, D. (2023). AutoFL: Towards AutoML in a federated learning context. *Applied Sciences, 13*(14), 8019. https://doi.org/10.3390/app13148019

Schulz, M. A., Yeo, B. T. T., Vogelstein, J. T., Mourao-Miranda, J., Kather, J. N., Kording, K., Richards, B., & Bzdok, D. (2020). Different scaling of linear models and deep learning in UKBiobank brain images versus machine-learning datasets. *Nature Communications, 11*, 4231. https://doi.org/10.1038/s41467-020-18037-z

Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv preprint arXiv:1708.07747*. https://doi.org/10.48550/arXiv.1708.07747
