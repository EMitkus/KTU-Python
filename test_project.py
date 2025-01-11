import torch
import tensorflow as tf
from model import Net
from data import save_to_mysql, load_from_mysql
from utils import calculate_metrics, plot_losses
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow_model import TensorFlowModel

# 1. Testuojamas PyTorch modelio sukūrimas
print("Testuojamas PyTorch modelio sukūrimas...")
pytorch_net = Net(num_classes=4)  # Sukuriamas CNN modelis su 4 klasėmis
print(pytorch_net)

# 2. Testuojamas TensorFlow modelio sukūrimas
print("Testuojamas TensorFlow modelio sukūrimas...")
tf_model = TensorFlowModel(input_shape=(128, 128, 3), num_classes=4)  # Sukuriamas TensorFlow modelis
tf_model.model.summary()

# 3. Testuojamas duomenų išsaugojimas ir nuskaitymas iš MySQL
print("Testuojamas duomenų išsaugojimas ir nuskaitymas iš MySQL...")
dummy_data = [torch.rand(3, 128, 128) for _ in range(5)]  # Sukuriami atsitiktiniai duomenys (5 paveikslėliai 128x128)
dummy_labels = [0, 1, 2, 3, 0]  # Sukuriamos atsitiktinės klasės
dummy_dataset = list(zip(dummy_data, dummy_labels))  # Sudaromas porų sąrašų duomenų rinkinys

# Išsaugojimas į MySQL
save_to_mysql(dummy_dataset, "test_images")  # Duomenys išsaugomi į MySQL lentelę "test_images"

# Nuskaitymas iš MySQL
X, y = load_from_mysql("test_images")  # Duomenys nuskaitomi iš MySQL
print(f"Įkelti duomenys: {len(X)}, Žymenys: {len(y)}")

# 4. Testuojamas TensorFlow modelio treniravimas
print("Testuojamas TensorFlow modelio treniravimas...")
X_tf = tf.random.uniform((10, 128, 128, 3))  # Sukuriami atsitiktiniai TensorFlow duomenys
y_tf = tf.random.uniform((10,), minval=0, maxval=4, dtype=tf.int32)  # Sukuriamos atsitiktinės TensorFlow klasės
y_tf_categorical = to_categorical(y_tf, num_classes=4)

history = tf_model.train(X_tf[:8], y_tf_categorical[:8], X_tf[8:], y_tf_categorical[8:], epochs=3, batch_size=2)
print("TensorFlow treniravimas baigtas.")

# 5. Testuojamas PyTorch metrikų skaičiavimas
print("Testuojamas metrikų skaičiavimas su PyTorch...")
dummy_preds = torch.tensor([0, 1, 2, 3, 0])  # Sukuriami atsitiktiniai numatymai
dummy_labels = torch.tensor([0, 1, 2, 3, 0])  # Sukuriamos atsitiktinės klasės
metrics = calculate_metrics(dummy_preds, dummy_labels, num_classes=4)  # Skaičiuojamos metrikos
print(f"Metrikos: {metrics}")  # Tikslumo, prisiminimo ir precizijos reikšmės

# 6. Testuojamas nuostolių vizualizavimas
print("Testuojamas vizualizavimas...")
plot_losses([0.8, 0.6, 0.4], [0.9, 0.7, 0.5])  # Vizualizuojami treniravimo ir validacijos nuostoliai

# 7. TensorFlow modelio testavimas
print("Testuojamas TensorFlow modelio įvertinimas...")
tf_results = tf_model.evaluate(X_tf[8:], y_tf[8:])
print(f"TensorFlow modelio įvertinimo rezultatai: Nuostoliai - {tf_results[0]}, Tikslumas - {tf_results[1]}")