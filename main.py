import argparse
import time
import tracemalloc
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet50
from data.mysql_handler import load_from_mysql
from utils.visualization import plot_losses
from tensorflow_model import TensorFlowModel


def main():
    # Komandinės eilutės argumentai
    parser = argparse.ArgumentParser(description="Alzheimer MRI klasifikacija naudojant CNN")
    parser.add_argument('--epochs', type=int, default=10, help="Epohų skaičius treniravimo metu")
    parser.add_argument('--batch_size', type=int, default=32, help="Partijos dydis treniravimo metu")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Mokymosi koeficientas")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Įrenginys: 'cuda' arba 'cpu'")
    args = parser.parse_args()

    # Parametrai
    device = torch.device(args.device)
    print(f"Naudojamas įrenginys: {device}")

    # 1. Duomenų įkėlimas iš MySQL
    print("Įkeliami treniravimo duomenys iš MySQL...")
    X_train, y_train = load_from_mysql("train_images")
    print("Įkeliami testavimo duomenys iš MySQL...")
    X_test, y_test = load_from_mysql("test_images")

    # 2. TensorFlow duomenų formatavimas
    X_train_tf = X_train.numpy().transpose(0, 2, 3, 1) / 255.0  # Formatas keičiamas į NHWC
    y_train_tf = y_train.numpy()
    X_test_tf = X_test.numpy().transpose(0, 2, 3, 1) / 255.0
    y_test_tf = y_test.numpy()

    # 3. TensorFlow modelio treniravimas
    print("Treniravimas su TensorFlow...")
    tf_model = TensorFlowModel(input_shape=(128, 128, 3), num_classes=4)
    start_time_tf = time.time()
    history = tf_model.train(X_train_tf, y_train_tf, X_test_tf, y_test_tf, epochs=args.epochs, batch_size=args.batch_size)
    end_time_tf = time.time()

    # TensorFlow rezultatų apdorojimas ir išsaugojimas
    tf_results = {
        'train_losses': history.history['loss'],
        'val_losses': history.history['val_loss'],
        'accuracy': history.history['val_accuracy'][-1],
        'time': end_time_tf - start_time_tf
    }

    tf_model.model.save("tensorflow_model.h5")
    print("TensorFlow modelis išsaugotas kaip 'tensorflow_model.h5'")
    print("\n=== TensorFlow Rezultatai ===")
    print(f"Treniravimo laikas: {tf_results['time']:.2f}s")
    print(f"Galutinis tikslumas: {tf_results['accuracy']:.4f}")
    plot_losses(tf_results['train_losses'], tf_results['val_losses'], title="TensorFlow Nuostoliai")

    # 4. PyTorch treniravimo ir validacijos logika
    print("\nTreniravimas su PyTorch...")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset_sizes = [0.5, 0.75, 1.0]
    results = {}

    for size in dataset_sizes:
        train_size = int(len(X_train) * size)
        _, train_subset = random_split(list(zip(X_train, y_train)), [len(X_train) - train_size, train_size])
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=args.batch_size, shuffle=False)

        net = resnet50(pretrained=True)
        net.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(net.fc.in_features, 4)
        )
        net = net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss()

        train_losses, val_losses = [], []
        tracemalloc.start()
        start_time = time.time()

        for epoch in range(args.epochs):
            net.train()
            epoch_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))

            net.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_losses.append(val_loss / len(test_loader))
            print(f"[{size*100:.0f}% duomenų] Epoha {epoch + 1}/{args.epochs}, "
                  f"Treniravimo nuostoliai: {train_losses[-1]:.4f}, Validacijos nuostoliai: {val_losses[-1]:.4f}")
            scheduler.step()

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[size] = {
            'time': end_time - start_time,
            'memory': peak / 10**6,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        # PyTorch modelio išsaugojimas
        torch.save(net.state_dict(), f"pytorch_model_{int(size*100)}.pth")
        print(f"PyTorch modelis ({size*100:.0f}% duomenų) išsaugotas kaip 'pytorch_model_{int(size*100)}.pth'")

    print("\n=== PyTorch Rezultatai ===")
    for size, stats in results.items():
        print(f"{size*100:.0f}% duomenų:")
        print(f"Treniravimo laikas: {stats['time']:.2f}s")
        print(f"Didžiausias atminties naudojimas: {stats['memory']:.2f} MB")
        plot_losses(stats['train_losses'], stats['val_losses'], title=f"PyTorch Nuostoliai ({size*100:.0f}% duomenų)")

    print("TensorFlow ir PyTorch rezultatai sugeneruoti")


if __name__ == "__main__":
    main()