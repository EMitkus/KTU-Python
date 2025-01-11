# Alzheimer MRI Klasifikacijos Projektas

Šis projektas skirtas Alzheimer MRI vaizdų klasifikacijai naudojant PyTorch ir TensorFlow neuroninius tinklus. Projektas apima treniravimo duomenų paruošimą, modelių kūrimą bei analizę, treniravimo ir validacijos rezultatų vizualizavimą.

---

## Reikalavimai:

Šio projekto paleidimui reikalingos šios bibliotekos (sąrašas pateiktas `requirements.txt` faile):
1. **torch** - Pytorch, naudojamas CNN treniravimui ir modelio kūrimui.
2. **torchvision** - Vaizdų apdorojimo funkcijos.
3. **torchmetrics** - Tikslumo, prisiminimo ir precizijos metrikos.
4. **mysql-connector-python** - MySQL duomenų bazės integracijai.
5. **matplotlib** - Nuostolių ir tikslumo grafikai.
6. **tensorflow** - TensorFlow modelių kūrimui ir treniravimui.
7. **numpy** - Skaičiavimų palengvinimui.

## Įdiekite visas priklausomybes naudodami komandą:
	pip install -r requirements.txt

## Projekto struktūra:
projekto_katalogas/ ├── data/ │ ├── init.py # Inicijuoja duomenų valdymo modulį. │ ├── mysql_handler.py # MySQL duomenų įrašymo/nuskaitymo funkcionalumas. ├── model/ │ ├── init.py # Inicijuoja modelių modulį. │ ├── cnn.py # CNN modelio aprašymas PyTorch pagrindu. │ ├── tensorflow.py # CNN modelio aprašymas TensorFlow pagrindu. ├── utils/ │ ├── init.py # Inicijuoja pagalbinių funkcijų modulį. │ ├── metrics.py # Metrikų ir duomenų dydžio skaičiavimo funkcijos. │ ├── visualization.py # Nuostolių grafiko vizualizavimo funkcija. ├── archive/ # Archyvui arba nenaudojami failai (jei yra). ├── train/ # Treniruojamieji duomenys (įkelti iš Kaggle). ├── test/ # Testavimo duomenys (įkelti iš Kaggle). ├── main.py # Pagrindinis skriptas treniravimo procesui paleisti. ├── test_project.py # Testavimo failas funkcijų ir modulio veikimo patikrinimui. ├── requirements.txt # Priklausomybių sąrašas, reikalingas projekto veikimui. ├── README.md # Dokumentacija apie projektą ir jo naudojimą. ├── .gitignore # Failai/katalogai, kurie neturėtų būti įtraukti į git repozitoriją.

## Naudojimas:
### Projekto Paleidimas:
#### TensorFlow modelio treniravimas:
	python main.py --model tensorflow --epochs 10 --batch_size 32
#### PyTorch modelio treniravimas:
	python main.py --model pytorch --epochs 10 --batch_size 32
### Testavimo Failas:
#### Projekto funkcionalumo testavimas:
	python test_project.py

## Rezultatai:
Treniravimo metu sugeneruojami rezultatai (nuostolių grafikai ir tikslumo metrikos), kurie:
    1. Išvedami į terminalą.
    2. Vizualizuojami grafikuose su matplotlib.

## Naudoti Duomenys:
Šis projektas naudoja Alzheimer MRI duomenų rinkinį iš Kaggle: Alzheimer MRI Dataset (99% Accuracy) https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy

### Duomenų struktūra:
    train/: Treniruojamoji duomenų dalis, suskirstyta į potinklius pagal klases (pvz., "Mild Impairment", "Moderate Impairment").
    test/: Testuojamoji duomenų dalis, suskirstyta taip pat kaip ir treniravimo dalis.
