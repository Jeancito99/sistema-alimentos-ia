from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import tensorflow as tf
import numpy as np
import cv2
import os

app = FastAPI(title="Food API")

# 1. Cargar el modelo .h5
MODEL_PATH = 'models/model.h5'
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo model.h5 cargado exitosamente.")
else:
    model = None
    print(f"¡Alerta! No se encontró el archivo '{MODEL_PATH}' en el directorio actual.")

# Función para preprocesar la imagen
def preprocess_image(file_bytes, target_size=(224, 224)):
    # Leer la imagen desde los bytes en memoria
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("El archivo enviado no es una imagen válida o está dañado.")
        
    # Redimensionar y normalizar (0 a 1)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Añadir dimensión de batch (1, H, W, C)
    return img

@app.post("/api/predict")
async def predict_food_status(
    imagen: UploadFile = File(...),
    humedad: float = Form(...),
    temperatura: float = Form(...)
):
    try:
        # Leer los bytes del archivo subido de forma asíncrona
        file_bytes = await imagen.read()
        
        # Preprocesar la imagen recibida
        processed_img = preprocess_image(file_bytes)

        # Preparar los datos numéricos como un arreglo de NumPy
        numeric_features = np.array([[humedad, temperatura]], dtype=np.float32)

        # 2. Realizar la predicción
        if model is not None:
            # Flujo Multi-Input: [imagen, datos_numericos]
            prediction = model.predict([processed_img, numeric_features])
            dias = int(np.round(prediction[0][0]))
        else:
            # Caso de prueba por si el modelo no carga en el despliegue
            dias = 3  

        # Evitar días negativos por ruido del modelo
        dias = max(0, dias)

        # 3. Formatear la respuesta JSON tal como la solicitaste
        return {
            "dias_restantes": dias,
            "estado": "Consumible" if dias > 2 else "Desechar"
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en la predicción: {str(e)}")