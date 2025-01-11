# Alzheimerio MRI Klasifikacija Naudojant TensorFlow ir PyTorch

## Aprašymas:
Šis projektas skirtas Alzheimerio ligos MRI vaizdų klasifikacijai, naudojant giluminio mokymosi metodus. Projektas leidžia pasirinkti TensorFlow arba PyTorch modelį ir įvertina treniravimo efektyvumą pagal pasirinktą metodą.

## Funkcionalumas:
Duomenų Apdorojimas:
1.Duomenys įkeliami iš MySQL duomenų bazės.
2.Automatiškai apskaičiuojamas treniravimo ir testavimo duomenų dydis (MB).
3.Duomenys gali būti padalinti į 50 %, 75 % ir 100 % apimties rinkinius treniravimui.

Modeliai:
1.TensorFlow: Naudojamas CNN modelis, sudarytas su Keras API.
2.PyTorch: Naudojamas ResNet50 modelis, iš anksto apmokytas su ImageNet duomenų rinkiniu.

Rezultatų Vizualizacija:
1.Treniravimo ir validacijos nuostolių grafikai.
2.Tikslumo (accuracy), prisiminimo (recall), ir precizijos (precision) metrikos.

Komandinės Eilutės Sąsaja (CLI):
1.Argumentai leidžia pasirinkti modelį (tensorflow arba pytorch), epohų skaičių, partijos dydį, ir kitus parametrus.

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
│   ├── __init__.py
│   ├── mysql_handler.py
│
├── model/
│   ├── __init__.py
│   ├── cnn.py
│   ├── tensorflow.py
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── visualization.py
│
├── archive/
│   ├── train/
│   ├── test/
│
├── main.py
├── test_project.py
├── setup.py
├── requirements.txt
├── README.md
├── .gitignore

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