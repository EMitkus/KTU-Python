# Alzheimerio MRI Klasifikacija Naudojant TensorFlow ir PyTorch

## Aprašymas:
Šis projektas skirtas Alzheimerio ligos MRI vaizdų klasifikacijai, naudojant giluminio mokymosi metodus. Projektas leidžia pasirinkti TensorFlow arba PyTorch modelį ir įvertina treniravimo efektyvumą pagal pasirinktą metodą.

## Funkcionalumas:
Duomenų Apdorojimas:
1.Duomenys įkeliami iš MySQL duomenų bazės.
2.Automatiškai apskaičiuojamas treniravimo ir testavimo duomenų dydis (MB).
3.Duomenys gali būti padalinti į 50 %, 75 % ir 100 % apimties rinkinius treniravimui.

Modeliai:
1.TensorFlow: Naudojamas CNN modelis, sudarytas su Keras API, su dropout ir ankstyvo sustabdymo funkcijomis.
2.PyTorch: Naudojamas ResNet50 modelis, iš anksto apmokytas su ImageNet duomenų rinkiniu. Taip pat įgyvendintos svorių reguliavimo (weight decay) ir ankstyvo sustabdymo funkcijos.

Rezultatų Vizualizacija:
1.Treniravimo ir validacijos nuostolių grafikai, sugeneruoti kiekvienai duomenų rinkinio proporcijai.
2.Tikslumo (accuracy), prisiminimo (recall), ir precizijos (precision) metrikos.

Komandinės Eilutės Sąsaja (CLI):
1.Argumentai leidžia pasirinkti modelį (tensorflow arba pytorch), epohų skaičių, partijos dydį ir kitus parametrus.

## Reikalavimai:
Šio projekto paleidimui reikalingos šios bibliotekos (sąrašas pateiktas requirements.txt faile):
1.torch (PyTorch)
2.torchvision
3.torchmetrics
4.tensorflow
5.mysql-connector-python
6.matplotlib
7.numpy

## Projekto Struktūra:
projekto_katalogas/
│
├── data/
│   ├── __init__.py           # Inicializuoja duomenų valdymo modulį
│   ├── mysql_handler.py      # MySQL duomenų įrašymo/nuskaitymo funkcionalumas
│
├── model/
│   ├── __init__.py           # Inicializuoja modelių modulį
│   ├── cnn.py                # CNN modelis PyTorch pagrindu
│   ├── tensorflow.py         # CNN modelis TensorFlow pagrindu
│
├── utils/
│   ├── __init__.py           # Inicializuoja pagalbinių funkcijų modulį
│   ├── metrics.py            # Metrikų ir duomenų dydžio skaičiavimo funkcijos
│   ├── visualization.py      # Nuostolių grafiko vizualizavimo funkcija
│
├── archive/
│   ├── train/                # Treniravimui skirti duomenys
│   ├── test/                 # Testavimui skirti duomenys
│
├── main.py                   # Pagrindinis skriptas treniravimo procesui paleisti
├── test_project.py           # Testavimo failas funkcijų ir modulio veikimo patikrinimui
├── setup.py                  # Projekto konfigūravimo failas
├── requirements.txt          # Priklausomybių sąrašas
├── README.md                 # Dokumentacija apie projektą ir jo naudojimą
├── .gitignore                # Failai/katalogai, kurie neturėtų būti įtraukti į `git` repozitoriją

## Naudojimas:
1.Projekto paleidimas:
    a.TensorFlow modelio treniravimas:
        python main.py --model tensorflow --epochs 10 --batch_size 32
    b.PyTorch modelio treniravimas:
        python main.py --model pytorch --epochs 10 --batch_size 32
2.Testavimo Failas:
    a.Projekto funkcionalumo testas:
        python test_project.py
3.Rezultatai:
    a.Treniravimo metu sugeneruojami rezultatai (nuostolių grafikai ir tikslumo metrikos), kurie išvedami į terminalą ir vizualizuojami grafikais.

## Duomenys:
Šio projekto duomenų rinkinys pasiekiamas iš Kaggle: https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy

## Versijų Kontrolė:
1.Projekte naudojamas Git. Norint inicijuoti projektą:
    git init
    git add .
    git commit -m "Pradinis projekto įkėlimas"

## Rezultatų Vizualizacija:
1.Nuostolių ir tikslumo grafikai sugeneruojami treniravimo metu.
2.Palyginimas atliekamas tarp TensorFlow ir PyTorch rezultatų.
3.Grafikai pateikiami pagal duomenų rinkinio dydžius (50%, 75%, 100%).

## Pasiekimai:
1.Sėkmingai įgyvendinti TensorFlow ir PyTorch modeliai.
2.Duomenų įkėlimas ir išsaugojimas naudojant MySQL.
3.Ankstyvo sustabdymo (early stopping) ir svorių reguliavimo (weight decay) įdiegimas.
4.Vizualizacijos ir rezultatų analizė pagal duomenų rinkinio dydį.