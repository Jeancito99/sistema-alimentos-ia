import pandas as pd
import numpy as np
import cv2
import os
from keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIGURACIÓN
# =========================
BASE_DIR = r"D:\Machin Learning\dataset\Train"
CSV_PATH = os.path.join("data", "dataset_alimentos.csv")

# =========================
# CARGAR CSV (IMPORTANTE sep=";")
# =========================
df = pd.read_csv(CSV_PATH, sep=";")

# limpiar nombres de columnas
df.columns = df.columns.str.strip()

print("Columnas:", df.columns.tolist())

# =========================
# FUNCIÓN CARGA IMAGEN
# =========================
def load_and_preprocess_image(path):
    # limpiar separadores (CRÍTICO)
    path = path.replace("\\", "/")

    # unir correctamente
    full_path = os.path.join(BASE_DIR, path)

    print("DEBUG:", full_path)

    # verificar existencia
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"NO EXISTE: {full_path}")

    img = cv2.imread(full_path)

    if img is None:
        raise FileNotFoundError(f"OpenCV no pudo leer: {full_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    return img / 255.0

# =========================
# CARGAR IMÁGENES
# =========================
images = np.array([
    load_and_preprocess_image(p)
    for p in df["path_imagen"]
])

# =========================
# DATOS NUMÉRICOS
# =========================
numerical_data = df[["temp", "humedad"]].values

labels = df["dias_vida_util"].values

# escalar datos
scaler = StandardScaler()
numerical_data = scaler.fit_transform(numerical_data)

# =========================
# MODELO MULTIMODAL
# =========================

# Rama imagen
input_img = Input(shape=(128, 128, 3), name="img_input")
x = layers.Conv2D(32, (3,3), activation="relu")(input_img)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)

# Rama numérica
input_num = Input(shape=(2,), name="num_input")
y = layers.Dense(16, activation="relu")(input_num)

# fusión
combined = layers.concatenate([x, y])
z = layers.Dense(64, activation="relu")(combined)
output = layers.Dense(1, activation="linear")(z)

model = models.Model(inputs=[input_img, input_num], outputs=output)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# =========================
# SPLIT DATA
# =========================
X_train_img, X_test_img, X_train_num, X_test_num, y_train, y_test = train_test_split(
    images, numerical_data, labels, test_size=0.2, random_state=42
)

# =========================
# ENTRENAMIENTO
# =========================
model.fit(
    [X_train_img, X_train_num],
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1
)

# =========================
# GUARDAR MODELO
# =========================
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

model.save(os.path.join( "models", "food_model.h5"))

print("Modelo guardado correctamente")