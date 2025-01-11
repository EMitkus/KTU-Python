import torch
from torchmetrics.classification import Accuracy, Precision, Recall

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