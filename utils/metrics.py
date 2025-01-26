import os
import torch
from torchmetrics.classification import Accuracy, Precision, Recall

def calculate_dataset_size(directory):
    """
    Apskaičiuoja bendrą katalogo dydį (MB).
    :param directory: Katalogo kelias.
    :return: Dydis MB.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size / (1024 * 1024)  # Konvertuoja į MB

def calculate_metrics(preds, labels, num_classes):
    """
    Apskaičiuoja tikslumą, preciziją ir prisiminimą PyTorch modeliams.

    Args:
        preds (torch.Tensor): Numatymai (predictions) iš modelio.
        labels (torch.Tensor): Tikrieji žymenys.
        num_classes (int): Klasifikacijos klasių skaičius.

    Returns:
        dict: Žodynas su metrikų reikšmėmis (accuracy, precision, recall).
    """
    # Inicializuojamos metrikos su multiclass nustatymu
    metrics = {
        'accuracy': Accuracy(task='multiclass', num_classes=num_classes, average='macro'),
        'precision': Precision(task='multiclass', num_classes=num_classes, average='macro'),
        'recall': Recall(task='multiclass', num_classes=num_classes, average='macro')
    }
    # Metrikų skaičiavimas
    results = {name: metric(preds, labels).item() for name, metric in metrics.items()}
    return results