import mysql.connector
import numpy as np
import torch

# Prisijungimas prie MySQL
def connect_to_mysql():
    """
    Sukuria prisijungimą prie MySQL duomenų bazės.
    Grąžina: mysql.connector jungtis.
    """
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="MySQLPass123",
        database="image_data"
    )
    return conn

# Duomenų įrašymas į MySQL
def save_to_mysql(table_name, data, labels):
    """
    Išsaugo duomenų rinkinį MySQL lentelėje.
    :param table_name: Lentelės pavadinimas.
    :param data: Duomenų rinkinys (PyTorch tensoriai).
    :param labels: Duomenų žymenys (klasės).
    """
    conn = connect_to_mysql()
    cursor = conn.cursor()

    # SQL užklausa, skirta įrašyti duomenis
    query = f"INSERT INTO {table_name} (image_data, label) VALUES (%s, %s)"

    for img, label in zip(data, labels):
        # Konvertuoja PyTorch tensorių į NumPy masyvą ir tada į baitus
        img_bytes = img.numpy().astype(np.uint8).tobytes()  # Konvertuojama į uint8
        cursor.execute(query, (img_bytes, int(label)))  # Įrašoma į MySQL

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Įrašyta {len(data)} įrašų į lentelę '{table_name}'.")

# Duomenų nuskaitymas iš MySQL
def load_from_mysql(table_name):
    conn = connect_to_mysql()
    cursor = conn.cursor()

    # SQL užklausa norint gauti visus duomenis
    cursor.execute(f"SELECT image_data, label FROM {table_name}")
    rows = cursor.fetchall()

    images, labels = [], []
    for row in rows:
        # Debug: Patikrinkite atminties dydį
        print(f"Reading raw data size: {len(row[0])} bytes, label: {row[1]}")

        try:
            # Pabandykite nuskaityti ir pertvarkyti duomenis
            img = np.frombuffer(row[0], dtype=np.uint8)  # Skaitykite kaip uint8
            if img.size == 128 * 128 * 3:  # Tikėtinas dydis
                img = img.reshape(128, 128, 3)
            else:
                raise ValueError(f"Unexpected image size: {img.size} bytes")

            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalizavimas
            images.append(img)
            labels.append(row[1])
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    cursor.close()
    conn.close()

    return torch.stack(images), torch.tensor(labels)