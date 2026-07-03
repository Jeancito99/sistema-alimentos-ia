import numpy as np
from PIL import Image
import io
import joblib

# Cargamos un escalador para los datos numéricos (Temp/Humedad)
# Nota: Debes guardar este scaler durante el entrenamiento
def load_scaler():
    try:
        return joblib.load('models/scaler.pkl')
    except:
        return None

def preprocess_image(image_bytes):
    """
    Convierte bytes de imagen a formato compatible con CNN (128x128x3).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0  # Normalización de píxeles [0, 1]
    return np.expand_dims(img_array, axis=0)

def prepare_iot_data(temp, hum):
    """
    Escala los datos de sensores IoT antes de meterlos a la red neuronal.
    """
    data = np.array([[temp, hum]])
    scaler = load_scaler()
    if scaler:
        return scaler.transform(data)
    return data # Retorna sin escalar si no hay scaler (usar solo en pruebas rápidas)