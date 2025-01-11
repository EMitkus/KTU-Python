import matplotlib.pyplot as plt
import os

def plot_losses(train_losses, val_losses, title="Treniravimo ir validacijos nuostoliai", save_path=None):
    """
    Vizualizuoja treniravimo ir validacijos nuostolius.

    Args:
        train_losses (list): Treniruotės nuostoliai.
        val_losses (list): Validacijos nuostoliai.
        title (str): Grafiko pavadinimas.
        save_path (str): Jei nurodyta, išsaugo grafiką į nurodytą kelią.
    """
    plt.figure()
    plt.plot(train_losses, label="Treniravimo nuostoliai")
    plt.plot(val_losses, label="Validacijos nuostoliai")
    plt.xlabel("Epohos")
    plt.ylabel("Nuostoliai")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        # Sukuria katalogą, jei jo nėra
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Grafikas išsaugotas į {save_path}")
    
    plt.show()

def plot_metrics(metric_values, metric_name, title="Metrikų vizualizacija", save_path=None):
    """
    Vizualizuoja metrikų reikšmes per epochas.

    Args:
        metric_values (list): Metrikos reikšmės.
        metric_name (str): Metrikos pavadinimas (pvz., "Tikslumas").
        title (str): Grafiko pavadinimas.
        save_path (str): Jei nurodyta, išsaugo grafiką į nurodytą kelią.
    """
    plt.figure()
    plt.plot(metric_values, label=f"{metric_name}")
    plt.xlabel("Epohos")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        # Sukuria katalogą, jei jo nėra
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Grafikas išsaugotas į {save_path}")
    
    plt.show()