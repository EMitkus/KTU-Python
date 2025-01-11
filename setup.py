from setuptools import setup, find_packages

# Pagrindinė setup.py struktūra, skirta Python paketo konfigūracijai ir įdiegimui
setup(
    name="alzheimer_mri_project",  # Paketo pavadinimas
    version="1.0.0",  # Versijos numeris
    packages=find_packages(),  # Automatiškai suranda visus paketus projekte
    install_requires=[  # Bibliotekos, reikalingos projekto veikimui
        "torch",  # PyTorch - CNN treniravimui ir modelio kūrimui
        "torchvision",  # Vaizdų apdorojimo funkcijos
        "torchmetrics",  # Metrikų skaičiavimas (tikslumas, prisiminimas, precizija)
        "tensorflow",  # TensorFlow - alternatyviam modelio kūrimui
        "mysql-connector-python",  # MySQL integracijai
        "matplotlib",  # Grafikai nuostolių ir tikslumo vizualizavimui
        "pandas",  # Duomenų analizė ir manipuliavimas
    ],
    entry_points={  # Komandinės eilutės (CLI) sąsajos nustatymai
        'console_scripts': [
            'train_model=main:main',  # Sukuriama CLI komanda `train_model`, nukreipta į `main()` funkciją faile `main.py`
        ],
    },
    description="Paketas Alzheimer MRI klasifikacijai naudojant CNN ir TensorFlow modelius",
    url="https://github.com/jusu-projektas",  # Projekto GitHub ar kitos versijų kontrolės sistemos URL
    python_requires=">=3.8",  # Minimalus reikalaujamas Python versijos numeris
)