import tensorflow as tf

class TensorFlowModel:
    def __init__(self, input_shape, num_classes):
        # Inicializuojamas TensorFlow modelis su įvesties forma ir klasių skaičiumi
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        # Sukuriamas CNN modelis TensorFlow
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=input_shape),  # Normalizavimas
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')  # Išvestis su softmax aktyvacija
        ])
        model.compile(optimizer='adam',  # Naudojamas optimizatorius "adam"
                      loss='sparse_categorical_crossentropy',  # Nuostolio funkcija
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])  # Tikslumo metrikos
        return model

    def train(self, train_data, train_labels, val_data, val_labels, epochs, batch_size):
        # Ankstyvo sustabdymo funkcija
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        # Modelio treniravimo funkcija
        history = self.model.fit(train_data, train_labels,
                                 validation_data=(val_data, val_labels),
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=[callback])
        return history

    def evaluate(self, test_data, test_labels):
        # Modelio testavimo funkcija
        results = self.model.evaluate(test_data, test_labels, verbose=0)
        return results

    def save_model(self, filepath):
        # Modelio išsaugojimas
        self.model.save(filepath)

    def load_model(self, filepath):
        # Modelio įkėlimas
        self.model = tf.keras.models.load_model(filepath)