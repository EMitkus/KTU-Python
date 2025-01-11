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
    Įrašo vaizdų duomenis ir jų klases į MySQL lentelę.
    
    Argumentai:
    - table_name (str): Lentelės, į kurią įrašomi duomenys, pavadinimas.
    - data (list): Vaizdų duomenys kaip numpy masyvai.
    - labels (list): Kiekvienam vaizdui priskirtos klasės.
    """
    conn = connect_to_mysql()
    cursor = conn.cursor()
    
    # SQL užklausa su apsauga nuo SQL injekcijų
    query = f'INSERT INTO {table_name} (image_data, label) VALUES (%s, %s)'
    
    for img, label in zip(data, labels):
        cursor.execute(query, (img.tobytes(), int(label)))
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Įrašyta {len(data)} įrašų į lentelę '{table_name}'.")

# Duomenų nuskaitymas iš MySQL
def load_from_mysql(table_name):
    """
    Nuskaito vaizdų duomenis ir jų klases iš MySQL lentelės.
    
    Argumentai:
    - table_name (str): Lentelės, iš kurios duomenys nuskaitomi, pavadinimas.
    
    Grąžina:
    - images (torch.Tensor): Tensor formato vaizdai (N x C x H x W).
    - labels (torch.Tensor): Tensor formato žymės.
    """
    conn = connect_to_mysql()
    cursor = conn.cursor()
    cursor.execute(f'SELECT image_data, label FROM {table_name}')
    rows = cursor.fetchall()
    
    images, labels = [], []
    for row in rows:
        # Binarinio formato dekodavimas ir normalizacija
        img = np.frombuffer(row[0], dtype=np.uint8).reshape(128, 128, 3) / 255.0
        images.append(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1))
        labels.append(row[1])
    
    cursor.close()
    conn.close()
    
    print(f"Nuskaityti {len(images)} įrašai iš lentelės '{table_name}'.")
    return torch.stack(images), torch.tensor(labels)