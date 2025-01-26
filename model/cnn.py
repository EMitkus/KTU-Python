import torch
import torch.nn as nn

class Net(nn.Module):
    """
    CNN modelis Alzheimer MRI klasifikacijai su 128x128 įvestimi.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Išvestis: [batch_size, 32, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Išvestis: [batch_size, 32, 64, 64]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Išvestis: [batch_size, 64, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Išvestis: [batch_size, 64, 32, 32]
            nn.Flatten(),  # Išvestis: [batch_size, 64 * 32 * 32 = 65536]
            nn.Dropout(p=0.5)  # Dropout, kad sumažintume persimokymą
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),  # Tarpinis sluoksnis
            nn.ReLU(),
            nn.Dropout(p=0.7),  # Aukštesnis dropout koeficientas klasifikatoriuje
            nn.Linear(128, num_classes)  # Išvestis: [batch_size, num_classes]
        )
        self.printed = False  # Inicializuojame atributą, kad išvengtume klaidų

    def forward(self, x):
        # Patikriname, ar įvestis tinkama
        if x.ndim != 4 or x.size(1) != 3 or x.size(2) != 128 or x.size(3) != 128:
            raise ValueError("Įvesties dimensijos turi būti [batch_size, 3, 128, 128]")

        x = self.feature_extractor(x)  # Funkcijų ištraukimo žingsnis
        if not self.printed:  # Spausdiname dimensijas tik pirmą kartą
            print(f"Išvesties dimensijos po feature_extractor: {x.shape}")
            self.printed = True
        x = self.classifier(x)  # Klasifikavimo žingsnis
        return x

# Testavimo kodas
if __name__ == "__main__":
    num_classes = 4  # Klasifikavimo klasių skaičius
    model = Net(num_classes=num_classes)  # Sukuriamas modelis
    dummy_input = torch.rand(16, 3, 128, 128)  # Dummy duomenys su [batch_size, channels, height, width]
    output = model(dummy_input)  # Modelio išvestis
    print(f"Išvesties dimensijos: {output.shape}")  # Tikimasi [16, num_classes]