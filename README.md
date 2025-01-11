# Python Mašininio Mokymosi Projektas

Šiame projekte įgyvendinami mašininio mokymosi metodai naudojant PyTorch ir TensorFlow, taip pat atliekamas duomenų paruošimas bei vizualizacija.

## Funkcionalumas
- Klasifikavimo uždaviniai naudojant PyTorch ir TensorFlow modelius.
- Duomenų paėmimas iš MySQL duomenų bazės.
- Nuostolių ir tikslumo vizualizacija.
- Lygiagrečiai atliekami skaičiavimai naudojant TensorFlow.

## Diegimo instrukcijos
1. Reikalinga įdiegti Python 3.8+ ir šias bibliotekas:
pip install torch torchvision tensorflow mysql-connector-python matplotlib

2. Projekto paleidimas su PyTorch:
python main.py --epochs 5 --batch_size 16

3. Projekto paleidimas su TensorFlow:
python main.py --epochs 5 --batch_size 16 --use_tensorflow


## Rezultatai
Rezultatai pateikiami nuostolių ir tikslumo grafikuose bei išvesties failuose.