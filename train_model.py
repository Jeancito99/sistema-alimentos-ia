import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, concatenate

# 1. Simulación de datos
def crear_dataset_dummy(n=1000):
    data = {
        'temp': np.random.uniform(15, 30, n),
        'humedad': np.random.uniform(50, 90, n),
        'vida_util': [] # Target
    }
    # Fórmula simple: a mayor temp/humedad, menor vida útil
    for t, h in zip(data['temp'], data['humedad']):
        vida = 10 - (t * 0.2) - (h * 0.05) + np.random.normal(0, 0.5)
        data['vida_util'].append(max(0, vida))
    return pd.DataFrame(data)

# 2. Arquitectura Multimodal
def build_multimodal_model():
    # Rama de Imagen (CNN simplificada)
    img_input = Input(shape=(128, 128, 3), name="imagen")
    x = Conv2D(32, (3,3), activation='relu')(img_input)
    x = Flatten()(x)
    
    # Rama de Datos IoT (Temp, Humedad)
    iot_input = Input(shape=(2,), name="iot_data")
    y = Dense(16, activation='relu')(iot_input)
    
    # Fusión
    combined = concatenate([x, y])
    z = Dense(64, activation='relu')(combined)
    output = Dense(1, activation='linear', name="prediccion")(z)
    
    model = Model(inputs=[img_input, iot_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# --- EJECUCIÓN PRINCIPAL ---

# A. Crear datos
df = crear_dataset_dummy(n=1000)

# B. Preprocesamiento (Scaler)
# Crear carpeta models si no existe
if not os.path.exists('models'):
    os.makedirs('models')

scaler = StandardScaler()
# Ajustar el scaler con los datos de entrenamiento
scaler.fit(df[['temp', 'humedad']])

# Guardar el escalador
joblib.dump(scaler, 'models/scaler.pkl')
print("Scaler guardado. Ahora tu API usará la misma lógica.")

# C. Construir y guardar modelo
model = build_multimodal_model()
print("Modelo Multimodal estructurado exitosamente.")

# Guardar el modelo
model.save('models/food_model.h5')
print("Modelo guardado en models/food_model.h5")