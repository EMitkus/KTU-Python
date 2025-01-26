import argparse
import time
import tracemalloc
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet50
from data.mysql_handler import load_from_mysql
from utils.visualization import plot_losses
from utils.metrics import calculate_dataset_size
from model.tensorflow import TensorFlowModel

def main():
    # Komandinės eilutės argumentai
    parser = argparse.ArgumentParser(description="Alzheimer MRI klasifikacija naudojant CNN")
    parser.add_argument('--model', type=str, choices=['tensorflow', 'pytorch'], required=True,
                        help="Modelio pasirinkimas: 'tensorflow' arba 'pytorch'")
    parser.add_argument('--epochs', type=int, default=10, help="Epohų skaičius treniravimo metu")
    parser.add_argument('--batch_size', type=int, default=32, help="Partijos dydis treniravimo metu")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Mokymosi koeficientas")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Įrenginys: 'cuda' arba 'cpu'")
    parser.add_argument('--data_dir', type=str, default='archive/', help="Kelias į duomenų katalogą")
    args = parser.parse_args()

    # Parametrai
    device = torch.device(args.device)
    print(f"Naudojamas įrenginys: {device}")

    # 1. Duomenų dydžių apskaičiavimas
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    train_size_mb = calculate_dataset_size(train_dir)
    test_size_mb = calculate_dataset_size(test_dir)
    print(f"Treniravimo duomenų dydis: {train_size_mb:.2f} MB")
    print(f"Testavimo duomenų dydis: {test_size_mb:.2f} MB")

    # 2. Duomenų įkėlimas iš MySQL
    print("Įkeliami treniravimo duomenys iš MySQL...")
    X_train, y_train = load_from_mysql("train_images")
    print("Įkeliami testavimo duomenys iš MySQL...")
    X_test, y_test = load_from_mysql("test_images")

    if args.model == 'tensorflow':
        # TensorFlow duomenų formatavimas
        X_train_tf = X_train.reshape(-1, 128, 128, 3) / 255.0
        y_train_tf = y_train
        X_test_tf = X_test.reshape(-1, 128, 128, 3) / 255.0
        y_test_tf = y_test

        # TensorFlow modelio treniravimas
        print("Treniravimas su TensorFlow...")
        tf_model = TensorFlowModel(input_shape=(128, 128, 3), num_classes=4)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        start_time_tf = time.time()
        history = tf_model.train(X_train_tf, y_train_tf, X_test_tf, y_test_tf, epochs=args.epochs, batch_size=args.batch_size, callbacks=[early_stopping])
        end_time_tf = time.time()

        # TensorFlow rezultatų apdorojimas
        tf_results = {
            'train_losses': history.history['loss'],
            'val_losses': history.history['val_loss'],
            'accuracy': history.history['val_accuracy'][-1],
            'time': end_time_tf - start_time_tf
        }

        print("\n=== TensorFlow Rezultatai ===")
        print(f"Treniravimo laikas: {tf_results['time']:.2f}s")
        print(f"Galutinis tikslumas: {tf_results['accuracy']:.4f}")
        plot_losses(tf_results['train_losses'], tf_results['val_losses'], title="TensorFlow Nuostoliai")

    elif args.model == 'pytorch':
        # PyTorch treniravimas ir validacija
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
                torch.nn.Dropout(p=0.7),  # Aukštesnis dropout reguliavimas
                torch.nn.Linear(net.fc.in_features, 4)
            )
            net = net.to(device)

            optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)  # Pridėtas weight decay
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            criterion = torch.nn.CrossEntropyLoss()

            train_losses, val_losses = [], []
            tracemalloc.start()
            start_time = time.time()

            best_val_loss = float('inf')
            patience_counter = 0
            patience_limit = 3  # Ankstyvas sustabdymas po 3 epohų be pagerėjimo

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

                val_loss /= len(test_loader)
                val_losses.append(val_loss)
                print(f"[{size*100:.0f}% duomenų] Epoha {epoch + 1}/{args.epochs}, "
                      f"Treniravimo nuostoliai: {train_losses[-1]:.4f}, Validacijos nuostoliai: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        print("Ankstyvas sustabdymas pasiektas.")
                        break

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

        print("\n=== PyTorch Rezultatai ===")
        for size, stats in results.items():
            print(f"{size*100:.0f}% duomenų:")
            print(f"Treniravimo laikas: {stats['time']:.2f}s")
            print(f"Didžiausias atminties naudojimas: {stats['memory']:.2f} MB")
            plot_losses(stats['train_losses'], stats['val_losses'], title=f"PyTorch Nuostoliai ({size*100:.0f}% duomenų)")

    print("Rezultatai sugeneruoti")

if __name__ == "__main__":
    main()